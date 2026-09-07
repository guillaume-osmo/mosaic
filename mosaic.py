"""Mosaic: a lossy music codec.

An original implementation of established transform-coding techniques. The
codec core invokes no existing audio codec; NumPy and SciPy provide the
transform and array primitives, and the standard library provides only integer
and byte handling.

Signal path:

1. Sine-windowed MDCT, 1024-sample window, 512 hop, folded into an orthonormal
   DCT-IV. Window length is selected per file from the low-frequency energy
   fraction.
2. Auditory-scale bands, each floored to a minimum bin count, with per-band
   left/right or mid/side stereo chosen on correlation.
3. Per band, per frame, per channel, the quantized ``log2(band RMS)`` is
   transmitted. Both sides derive the quantizer step from it identically:
   ``step = global * max(rms, past_masked_rms) * EMPH[band]``, floored. The step
   is therefore proportional to band energy — constant relative error — and the
   decoder knows how much energy each band should have.
4. Bounded reverse water-filling across six frequency buckets. Each bucket's
   actual coding load is measured at a data-derived reference step, and buckets
   spending more bits per bin than average take a bounded coarser multiplier
   (0.8–1.25x, geometric-mean renormalised to 1). The six multipliers are
   quantized and transmitted, so the decoder reads them rather than recomputing
   them: a poor allocation costs quality, never a broken stream. Buckets with
   too little measurable content are pinned to 1.0 and left alone.
5. Dead-zone scalar quantization, then a context-modelled adaptive binary range
   coder — zero-flag hierarchy, significance and greater-than-one bits, and an
   Exp-Golomb remainder, all causally contexted.
6. Parametric noise substitution. The decoder compares the energy it
   reconstructed per band against the transmitted band energy and fills the
   deficit into bins that quantized to zero with deterministic, energy-matched
   noise. An unaffordable band is reconstructed with the correct spectral
   envelope instead of as silence, and a band that survived quantization
   receives almost nothing, so the fill is self-limiting. It is disabled below
   250 Hz, where content is tonal and noise would damage structure.
7. Groups of 64 frames are coded independently, with a bisection search on a
   global step multiplier to meet the byte budget.

Encoding targets a byte budget rather than a bitrate, because that is the only
basis on which codecs compare fairly — see RESULTS.md.

    encode(samples, target_bytes) -> bytes, len(result) <= target_bytes
    decode(payload)               -> float32 (n, channels)

`samples` is float32 (n, channels), channels in (1, 2), 48 kHz, within [-1, 1].
`decode` returns exactly the sample count that was encoded.
"""
from __future__ import annotations

import argparse
import io
import math
import os
import struct
import sys
import tempfile
import wave
import zlib
from pathlib import Path

import numpy as np
from scipy import fft

try:
    import lzma
except ImportError:  # pragma: no cover - stdlib almost always has it
    lzma = None

try:
    import bz2
except ImportError:  # pragma: no cover - stdlib almost always has it
    bz2 = None

VERSION = 3

RATE = 48000
HOP = 512
GROUP = 64
MAX_SAMPLES = RATE * 600
MAGIC = b"MSC3"
HEADER = struct.Struct("<4sIBHQ")
CRC = struct.Struct("<I")
N_BUCKET = 6
ALLOC_LO, ALLOC_HI = 0.75, 1.35
PACKET_FIELDS = struct.Struct(f"<HfIIB{N_BUCKET}B")
PACKET = struct.Struct(f"<HfIIB{N_BUCKET}BI")

MIN_BAND_BINS = 3
THETA = 0.35                 # dead-zone quantizer offset; zero bin is +-0.65 step
RECON = 0.5 - THETA          # reconstruction point inside a nonzero cell
LOW_SHELF_HZ = 900.0         # above this, no bass correction at all
LOW_MIN_HZ = 50.0            # the ramp saturates at its floor by here
EMPH_LOW_FLOOR = 0.6         # finest allowed step scaling, at the very bottom
EMPH_TOP_MAX = 1.8           # coarsening ceiling, when a frame's treble is near-silent
EMPH_TOP_MIN = 1.15          # coarsening floor, when treble carries real energy
TREBLE_ALPHA = 6.0           # sensitivity of that swing to measured treble share
MASK_FRAC = 0.25             # within-band post-masking reference, -12 dB
DECAY = 0.65                 # per-frame decay of that reference
FLOOR_FRAC = 0.08            # frame-level absolute floor, from the local percentile
FILL_LOW_HZ = 250.0          # no noise substitution below here
LZMA_PRESET = 0
BISECT_ITERS = 12
MAX_STEP = 1.0e6

CODEC_LZMA, CODEC_BZ2, CODEC_ZLIB = 0, 1, 2

K_TOP = 1 << 24
K_BITS = 11
K_MOVE = 5
PROB_ONE = 1 << K_BITS
PROB_INIT = PROB_ONE >> 1


class CodecError(ValueError):
    """Invalid audio, unsupported format, corrupt bitstream or impossible rate."""


def _pack_alloc(mult):
    """Quantize 6 bucket multipliers into bytes for the packet header."""
    codes = np.clip(
        np.round((np.asarray(mult, dtype=np.float64) - ALLOC_LO)
                 / (ALLOC_HI - ALLOC_LO) * 255.0), 0, 255
    ).astype(np.uint8)
    return tuple(int(c) for c in codes)


def _unpack_alloc(codes):
    arr = np.asarray(codes, dtype=np.float32)
    return (np.float32(ALLOC_LO) + arr * np.float32((ALLOC_HI - ALLOC_LO) / 255.0))


NEUTRAL_ALLOC_CODES = _pack_alloc(np.ones(N_BUCKET, dtype=np.float32))


def _compress_best(raw: bytes):
    """Try every available stdlib compressor, return (codec_id, smallest bytes).

    Still used for the small side-information blob (M/S flags and scale-index
    deltas), where LZ matching across the frame axis does well and the volume
    is too low to justify a second context model.
    """
    best_id, best = CODEC_ZLIB, zlib.compress(raw, level=9)
    if bz2 is not None:
        candidate = bz2.compress(raw, compresslevel=9)
        if len(candidate) < len(best):
            best_id, best = CODEC_BZ2, candidate
    if lzma is not None:
        candidate = lzma.compress(raw, format=lzma.FORMAT_ALONE, preset=LZMA_PRESET)
        if len(candidate) < len(best):
            best_id, best = CODEC_LZMA, candidate
    return best_id, best


def _decompress(data: bytes, codec: int) -> bytes:
    if codec == CODEC_LZMA:
        if lzma is None:
            raise CodecError("Payload needs lzma, unavailable in this interpreter.")
        worker = lambda blob: lzma.decompress(blob, format=lzma.FORMAT_ALONE)
    elif codec == CODEC_BZ2:
        if bz2 is None:
            raise CodecError("Payload needs bz2, unavailable in this interpreter.")
        worker = bz2.decompress
    elif codec == CODEC_ZLIB:
        worker = zlib.decompress
    else:
        raise CodecError("Unknown compressor id in packet.")
    try:
        return worker(data)
    except Exception as error:  # noqa: BLE001 - any coder failure is a corrupt stream
        raise CodecError("Invalid entropy stream.") from error


def _build_edges(min_bins: int = MIN_BAND_BINS) -> np.ndarray:
    """Auditory-scale band edges in MDCT bins, each band at least min_bins wide."""
    hz_edges = np.r_[0, np.geomspace(80, RATE / 2, 30)]
    raw = np.clip(np.rint(hz_edges * (2 * HOP / RATE)), 0, HOP).astype(int)
    raw = np.unique(raw)
    if raw[-1] != HOP:
        raw = np.append(raw, HOP)
    edges = [int(raw[0])]
    for edge in raw[1:-1]:
        if edge - edges[-1] >= min_bins:
            edges.append(int(edge))
    edges.append(int(raw[-1]))
    edges = np.array(sorted(set(edges)), dtype=int)
    while len(edges) > 2 and edges[-1] - edges[-2] < min_bins:
        edges = np.delete(edges, -2)
    return edges


EDGES = _build_edges()
BANDS = len(EDGES) - 1
WIDTHS = np.diff(EDGES).astype(np.float32)
BIN_BANDS = np.repeat(np.arange(BANDS), np.diff(EDGES))
WINDOW = np.sin(np.pi / (2 * HOP) * (np.arange(2 * HOP) + 0.5)).astype(np.float32)

EDGES_LIST = EDGES.tolist()

BAND_BUCKET = np.minimum((np.arange(BANDS) * N_BUCKET) // BANDS, N_BUCKET - 1)
BIN_BUCKET = BAND_BUCKET[BIN_BANDS]
BAND_BUCKET_LIST = BAND_BUCKET.tolist()
BIN_BUCKET_LIST = BIN_BUCKET.tolist()

CTX_SIG = 0                          # N_BUCKET * 5
CTX_GT1 = CTX_SIG + N_BUCKET * 5     # N_BUCKET * 3
CTX_EG = CTX_GT1 + N_BUCKET * 3      # 8 prefix positions
CTX_BAND = CTX_EG + 8                # N_BUCKET * 2
N_CTX = CTX_BAND + N_BUCKET * 2

_BIN_HZ = RATE / (2.0 * HOP)
_LOW_HZ = np.maximum(EDGES[:-1] * _BIN_HZ, 20.0)
_HIGH_HZ = np.maximum(EDGES[1:] * _BIN_HZ, 40.0)
CENTER_HZ = np.sqrt(_LOW_HZ * _HIGH_HZ)

_low_ramp = np.clip(
    np.log2(LOW_SHELF_HZ / np.maximum(CENTER_HZ, LOW_MIN_HZ))
    / np.log2(LOW_SHELF_HZ / LOW_MIN_HZ),
    0.0, 1.0,
).astype(np.float32)
EMPH_BASE = (1.0 - np.float32(1.0 - EMPH_LOW_FLOOR) * _low_ramp).astype(np.float32)

TOP_MASK = CENTER_HZ > 4000.0
TOP_RAMP_SUB = np.zeros(int(np.count_nonzero(TOP_MASK)), dtype=np.float32)
if np.any(TOP_MASK):
    TOP_RAMP_SUB = np.clip(
        np.log2(CENTER_HZ[TOP_MASK] / 4000.0) / np.log2(5.0), 0.0, 1.0
    ).astype(np.float32)

FILL = np.clip(np.log2(CENTER_HZ / FILL_LOW_HZ) / 2.0, 0.0, 1.0).astype(np.float32)


def _alloc_bins(mult):
    return mult[BIN_BUCKET]


class RangeEncoder:
    """Carry-less binary range encoder, 11-bit adaptive probabilities."""

    __slots__ = ("low", "span", "cache", "cache_size", "out")

    def __init__(self):
        self.low = 0
        self.span = 0xFFFFFFFF
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()

    def _shift_low(self):
        low = self.low
        if low < 0xFF000000 or low > 0xFFFFFFFF:
            carry = low >> 32
            temp = self.cache
            out = self.out
            while True:
                out.append((temp + carry) & 0xFF)
                temp = 0xFF
                self.cache_size -= 1
                if self.cache_size == 0:
                    break
            self.cache = (low >> 24) & 0xFF
        self.cache_size += 1
        self.low = (low << 8) & 0xFFFFFFFF

    def encode_bit(self, probs, index, bit):
        prob = probs[index]
        bound = (self.span >> K_BITS) * prob
        if bit:
            self.low += bound
            self.span -= bound
            probs[index] = prob - (prob >> K_MOVE)
        else:
            self.span = bound
            probs[index] = prob + ((PROB_ONE - prob) >> K_MOVE)
        while self.span < K_TOP:
            self._shift_low()
            self.span <<= 8

    def encode_bypass(self, bit):
        self.span >>= 1
        if bit:
            self.low += self.span
        while self.span < K_TOP:
            self._shift_low()
            self.span <<= 8

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class RangeDecoder:
    """Mirror of RangeEncoder; reads zeros past the end rather than failing."""

    __slots__ = ("data", "size", "pos", "span", "code")

    def __init__(self, data):
        self.data = data
        self.size = len(data)
        self.span = 0xFFFFFFFF
        code = 0
        for index in range(1, 5):
            code = (code << 8) | (data[index] if index < self.size else 0)
        self.code = code
        self.pos = 5

    def _next(self):
        pos = self.pos
        self.pos = pos + 1
        return self.data[pos] if pos < self.size else 0

    def decode_bit(self, probs, index):
        prob = probs[index]
        bound = (self.span >> K_BITS) * prob
        if self.code < bound:
            self.span = bound
            probs[index] = prob + ((PROB_ONE - prob) >> K_MOVE)
            bit = 0
        else:
            self.code -= bound
            self.span -= bound
            probs[index] = prob - (prob >> K_MOVE)
            bit = 1
        while self.span < K_TOP:
            self.code = ((self.code << 8) | self._next()) & 0xFFFFFFFF
            self.span <<= 8
        return bit

    def decode_bypass(self):
        self.span >>= 1
        if self.code >= self.span:
            self.code -= self.span
            bit = 1
        else:
            bit = 0
        while self.span < K_TOP:
            self.code = ((self.code << 8) | self._next()) & 0xFFFFFFFF
            self.span <<= 8
        return bit


def _new_probs():
    return [PROB_INIT] * N_CTX


def mdct(frames):
    """Sine-windowed MDCT via folding and an orthonormal DCT-IV."""
    windowed = frames * WINDOW
    half = HOP // 2
    folded = np.concatenate(
        (
            -windowed[..., HOP:HOP + half][..., ::-1] - windowed[..., HOP + half:],
            windowed[..., :half] - windowed[..., half:HOP][..., ::-1],
        ),
        axis=-1,
    )
    return fft.dct(folded, type=4, norm="ortho", axis=-1, workers=1)


def imdct(coefficients):
    """Inverse MDCT contributions, ready for 50% overlap-add."""
    folded = fft.dct(coefficients, type=4, norm="ortho", axis=-1, workers=1)
    half = HOP // 2
    return np.concatenate(
        (folded[..., half:], -folded[..., ::-1], -folded[..., :half]), axis=-1) * WINDOW


def _local_floor(frame_rms):
    """Past-biased local low percentile of the frame RMS envelope."""
    padded = np.pad(frame_rms, (5, 2), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 8)
    return np.percentile(windows, 25.0, axis=1).astype(np.float32)


def _smooth(values, radius=4):
    """Centered moving average, computable by the decoder from the whole group."""
    if len(values) <= 1:
        return values
    kernel = np.ones(2 * radius + 1, dtype=np.float32) / np.float32(2 * radius + 1)
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def derive_steps(scales):
    """Rebuild band RMS and per-bin quantizer steps from the transmitted scales.

    Run identically by encoder and decoder -- everything here comes out of the
    uint8 scale bytes that are actually in the bitstream, so the noise-fill
    target energy and the step size never disagree between the two sides.
    """
    count, channels, _ = scales.shape
    rms = np.exp2((scales.astype(np.float32) - np.float32(128.0)) / np.float32(8.0))
    rms = np.where(scales > 0, rms, np.float32(0.0)).astype(np.float32)

    reference = rms.copy()
    decay = np.float32(DECAY)
    for index in range(1, count):
        np.maximum(rms[index], reference[index - 1] * decay, out=reference[index])

    power = (rms * rms) * WIDTHS
    total_power = power.sum(axis=(1, 2))
    frame_rms = np.sqrt(
        total_power / np.float32(channels * HOP) + np.float32(1e-30)
    ).astype(np.float32)
    floor = _local_floor(frame_rms) * np.float32(FLOOR_FRAC)

    high_power = power[:, :, TOP_MASK].sum(axis=(1, 2))
    treble_share = _smooth(
        np.clip(high_power / (total_power + np.float32(1e-30)), 0.0, 1.0).astype(np.float32)
    )
    top_target = np.float32(EMPH_TOP_MAX) - np.float32(EMPH_TOP_MAX - EMPH_TOP_MIN) * np.clip(
        treble_share * np.float32(TREBLE_ALPHA), 0.0, 1.0
    )
    emph = np.broadcast_to(EMPH_BASE, (count, BANDS)).copy()
    if TOP_RAMP_SUB.size:
        emph[:, TOP_MASK] = 1.0 + (top_target[:, None] - 1.0) * TOP_RAMP_SUB[None, :]

    base = np.maximum(rms, reference * np.float32(MASK_FRAC)) * emph[:, None, :]
    base = np.maximum(base, floor[:, None, None])
    base = np.maximum(base, np.float32(1e-9)).astype(np.float32)
    return rms, base[..., BIN_BANDS]


def prepare_group(coefficients):
    """Select L/R or M/S per band, measure band RMS, normalise by derived steps."""
    count, channels, _ = coefficients.shape
    transformed = coefficients.copy()
    modes = np.zeros((count, BANDS), dtype=np.uint8)
    scales = np.zeros((count, channels, BANDS), dtype=np.uint8)
    for band in range(BANDS):
        left, right = int(EDGES[band]), int(EDGES[band + 1])
        block = transformed[..., left:right]
        if channels == 2:
            first, second = block[:, 0].copy(), block[:, 1].copy()
            energy = np.sum(first ** 2, axis=-1) * np.sum(second ** 2, axis=-1)
            correlation = np.abs(np.sum(first * second, axis=-1)) / np.sqrt(energy + 1e-30)
            use_ms = correlation > 0.3
            modes[:, band] = use_ms
            block[use_ms, 0] = (first[use_ms] + second[use_ms]) * np.float32(2 ** -0.5)
            block[use_ms, 1] = (first[use_ms] - second[use_ms]) * np.float32(2 ** -0.5)
        rms = np.sqrt(np.mean(block.astype(np.float32) ** 2, axis=-1))
        index = np.rint(np.log2(np.maximum(rms, 1e-30)) * 8.0) + 128.0
        index = np.clip(index, 1.0, 255.0)
        index = np.where(rms < 2.0 ** -15.0, 0.0, index)
        scales[:, :, band] = index.astype(np.uint8)

    _, base = derive_steps(scales)
    normalized = transformed / base
    mode_bytes = np.packbits(modes, bitorder="little").tobytes() if channels == 2 else b""
    delta = np.diff(scales.astype(np.int16), axis=0,
                    prepend=np.zeros_like(scales[:1], dtype=np.int16))
    scale_bytes = delta.astype(np.uint8).transpose(1, 2, 0).tobytes()
    return normalized, mode_bytes + scale_bytes


def _side_size(count, channels):
    mode_size = (count * BANDS + 7) // 8 if channels == 2 else 0
    return mode_size + count * channels * BANDS


def _quantize_ints(normalized, step):
    """Dead-zone scalar quantization to signed integers."""
    scaled = normalized / np.float32(step)
    magnitude = np.floor(np.abs(scaled) + np.float32(THETA))
    magnitude = np.clip(magnitude, 0.0, 32000.0)
    return np.where(scaled < 0, -magnitude, magnitude).astype(np.int32)


def _bucket_multipliers(normalized):
    """Bounded reverse water-filling across the 6 frequency buckets.

    Measures each bucket's coding load at a data-derived reference step (the
    median magnitude of this group's own normalized coefficients, so a quiet
    or loud group does not degenerate to all-zero or all-saturated), using the
    real dead-zone quantizer -- not a proxy like crest factor. A bucket
    spending more bits per bin than average gets a bounded coarser multiplier,
    one spending fewer gets a bounded finer one, and the reliable set is
    renormalized so the outer step search still targets the right total rate.
    Buckets without enough nonzero content to measure honestly are pinned to
    exactly 1.0, so this cannot invent detail in a near-silent band.
    """
    ref_step = float(np.median(np.abs(normalized))) + 1e-6
    ref = np.abs(_quantize_ints(normalized, ref_step))
    cost = np.zeros(N_BUCKET, dtype=np.float64)
    bins = np.zeros(N_BUCKET, dtype=np.float64)
    reliable = np.zeros(N_BUCKET, dtype=bool)
    for bucket in range(N_BUCKET):
        sub = ref[:, :, BIN_BUCKET == bucket]
        n = sub.size
        if n == 0:
            continue
        bins[bucket] = n
        nonzero = sub > 0
        share = float(nonzero.mean())
        if share < 0.01:
            continue
        reliable[bucket] = True
        share = min(share, 1.0 - 1e-6)
        entropy2 = -share * math.log2(share) - (1.0 - share) * math.log2(1.0 - share)
        mean_mag = float(sub[nonzero].mean())
        cost[bucket] = entropy2 + share * (1.0 + math.log2(mean_mag + 1.0))

    mult = np.ones(N_BUCKET, dtype=np.float64)
    if np.count_nonzero(reliable) < 2:
        return mult.astype(np.float32)
    average = np.average(cost[reliable], weights=bins[reliable])
    average = max(average, 1e-6)
    ratio = np.clip(cost[reliable] / average, 0.3, 3.0)
    scaled = np.clip(ratio ** 0.3, 0.8, 1.25)
    scaled *= math.exp(-np.average(np.log(scaled), weights=bins[reliable]))
    mult[reliable] = np.clip(scaled, ALLOC_LO, ALLOC_HI)
    return mult.astype(np.float32)


def _contexts(ints):
    """Causal context inputs shared by the coder and the cost estimator."""
    mags = np.abs(ints)
    nonzero = mags > 0
    band_nz = np.add.reduceat(nonzero.astype(np.int32), EDGES[:-1], axis=2) > 0
    capped = np.minimum(mags, 2)
    left = np.zeros_like(capped)
    left[:, :, 1:] = capped[:, :, :-1]
    earlier = np.zeros_like(capped)
    earlier[1:] = capped[:-1]
    return mags, nonzero, band_nz, left + earlier


def _ctx_bits(ctx, bits, n_ctx):
    """Ideal per-context entropy of a binary decision sequence, in bits."""
    if ctx.size == 0:
        return 0.0
    index = ctx.astype(np.int64) * 2 + bits.astype(np.int64)
    hist = np.bincount(index, minlength=n_ctx * 2).astype(np.float64).reshape(-1, 2)
    total = hist.sum(axis=1)
    good = total > 0
    if not np.any(good):
        return 0.0
    counts = hist[good]
    share = counts / total[good][:, None]
    cost = np.where(counts > 0.0, -np.log2(np.maximum(share, 1e-12)), 0.0)
    return float(np.sum(counts * cost))


def _estimate_bits(ints):
    """Vectorized cost of the coefficient stream under the coder's own model.

    This is what the rate search bisects on. It is the model's ideal code
    length rather than the coder's actual output, so a running calibration
    factor and a step-up retry in encode_group turn it into a hard guarantee.
    """
    mags, nonzero, band_nz, near = _contexts(ints)
    flags = band_nz.astype(np.int32)
    previous = np.zeros_like(flags)
    previous[1:] = flags[:-1]
    bits = _ctx_bits((BAND_BUCKET[None, None, :] * 2 + previous).ravel(),
                     flags.ravel(), 2 * N_BUCKET)
    active = band_nz[:, :, BIN_BANDS]
    if np.any(active):
        bits += _ctx_bits((BIN_BUCKET[None, None, :] * 5 + near)[active],
                          nonzero[active], 5 * N_BUCKET)
        chosen = active & nonzero
        if np.any(chosen):
            picked = mags[chosen]
            bits += _ctx_bits(
                (BIN_BUCKET[None, None, :] * 3 + np.minimum(near, 2))[chosen],
                picked > 1, 3 * N_BUCKET)
            bits += float(picked.size)
            big = picked[picked > 1] - 2
            if big.size:
                order = np.floor(np.log2(big.astype(np.float64) + 1.0))
                bits += 2.0 * float(np.sum(order)) + float(big.size)
    return bits + 40.0 + 8.0 * N_CTX


def _encode_coeffs(ints, enc, probs):
    """Write the quantized coefficients through the context-modelled coder."""
    count, channels, _ = ints.shape
    mags, _, band_nz, near = _contexts(ints)
    neighbour = near.tolist()
    flags = band_nz.astype(np.int32).tolist()
    magnitude = mags.tolist()
    negative = (ints < 0).tolist()

    edges = EDGES_LIST
    bucket = BIN_BUCKET_LIST
    bbucket = BAND_BUCKET_LIST
    put = enc.encode_bit
    raw = enc.encode_bypass
    history = [[0] * BANDS for _ in range(channels)]
    for frame in range(count):
        for channel in range(channels):
            row_flags = flags[frame][channel]
            row_mag = magnitude[frame][channel]
            row_neg = negative[frame][channel]
            row_near = neighbour[frame][channel]
            seen = history[channel]
            for band in range(BANDS):
                flag = row_flags[band]
                put(probs, CTX_BAND + bbucket[band] * 2 + seen[band], flag)
                seen[band] = flag
                if not flag:
                    continue
                for slot in range(edges[band], edges[band + 1]):
                    value = row_mag[slot]
                    hint = row_near[slot]
                    place = bucket[slot]
                    if value == 0:
                        put(probs, place * 5 + hint, 0)
                        continue
                    put(probs, place * 5 + hint, 1)
                    capped = hint if hint < 2 else 2
                    gt1 = CTX_GT1 + place * 3 + capped
                    if value == 1:
                        put(probs, gt1, 0)
                    else:
                        put(probs, gt1, 1)
                        rest = value - 2
                        order = 0
                        while rest >= (1 << order):
                            put(probs, CTX_EG + (order if order < 8 else 7), 1)
                            rest -= 1 << order
                            order += 1
                        put(probs, CTX_EG + (order if order < 8 else 7), 0)
                        while order:
                            order -= 1
                            raw((rest >> order) & 1)
                    raw(1 if row_neg[slot] else 0)


def _decode_coeffs(dec, probs, count, channels):
    """Read back exactly what _encode_coeffs wrote, rebuilding contexts causally."""
    edges = EDGES_LIST
    bucket = BIN_BUCKET_LIST
    bbucket = BAND_BUCKET_LIST
    get = dec.decode_bit
    raw = dec.decode_bypass
    history = [[0] * BANDS for _ in range(channels)]
    past = [[0] * HOP for _ in range(channels)]
    rows = []
    for frame in range(count):
        for channel in range(channels):
            seen = history[channel]
            earlier = past[channel]
            current = [0] * HOP
            values = [0] * HOP
            left = 0
            for band in range(BANDS):
                flag = get(probs, CTX_BAND + bbucket[band] * 2 + seen[band])
                seen[band] = flag
                if not flag:
                    left = 0
                    continue
                for slot in range(edges[band], edges[band + 1]):
                    hint = left + earlier[slot]
                    place = bucket[slot]
                    if not get(probs, place * 5 + hint):
                        left = 0
                        continue
                    capped = hint if hint < 2 else 2
                    if get(probs, CTX_GT1 + place * 3 + capped):
                        base = 0
                        order = 0
                        while get(probs, CTX_EG + (order if order < 8 else 7)):
                            base += 1 << order
                            order += 1
                            if order > 24:
                                raise CodecError("Corrupt magnitude prefix.")
                        suffix = 0
                        for _ in range(order):
                            suffix = (suffix << 1) | raw()
                        value = base + suffix + 2
                        if value > 32000:
                            value = 32000
                    else:
                        value = 1
                    values[slot] = -value if raw() else value
                    left = 2 if value > 2 else value
                    current[slot] = left
            past[channel] = current
            rows.append(values)
    return np.array(rows, dtype=np.int32).reshape(count, channels, HOP)


def _silent_packet(count, channels):
    """A valid, tiny packet decoding to silence -- the last-resort budget escape."""
    codec_id, side_blob = _compress_best(bytes(_side_size(count, channels)))
    enc = RangeEncoder()
    _encode_coeffs(np.zeros((count, channels, HOP), dtype=np.int32), enc, _new_probs())
    blob = enc.finish()
    fields = PACKET_FIELDS.pack(count, 1.0, len(side_blob), len(blob), codec_id,
                                *NEUTRAL_ALLOC_CODES)
    return fields + CRC.pack(zlib.crc32(fields + side_blob + blob)) + side_blob + blob


def encode_group(coefficients, budget, calib):
    """Water-fill bucket multipliers, bisect a global step, run the coder once."""
    count, channels, _ = coefficients.shape
    normalized, side_raw = prepare_group(coefficients)
    alloc = _bucket_multipliers(normalized)
    alloc_codes = _pack_alloc(alloc)
    normalized = normalized / _alloc_bins(alloc)[None, None, :]
    codec_id, side_blob = _compress_best(side_raw)
    coef_budget = budget - len(side_blob)
    if coef_budget < 48:
        return _silent_packet(count, channels), None

    limit = max(96.0, coef_budget * 8.0 * 0.995 / calib)
    low, high = 0.01, 1.0
    while _estimate_bits(_quantize_ints(normalized, high)) > limit and high < 65536.0:
        low, high = high, high * 2.0
    for _ in range(BISECT_ITERS):
        middle = math.sqrt(low * high)
        if _estimate_bits(_quantize_ints(normalized, middle)) <= limit:
            high = middle
        else:
            low = middle
    step = float(np.float32(high))

    for _ in range(14):
        ints = _quantize_ints(normalized, step)
        estimate = _estimate_bits(ints)
        enc = RangeEncoder()
        _encode_coeffs(ints, enc, _new_probs())
        blob = enc.finish()
        if len(blob) <= coef_budget:
            fields = PACKET_FIELDS.pack(count, step, len(side_blob), len(blob), codec_id,
                                        *alloc_codes)
            packet = (fields + CRC.pack(zlib.crc32(fields + side_blob + blob))
                      + side_blob + blob)
            return packet, (len(blob) * 8.0) / max(estimate, 1.0)
        step = float(np.float32(min(step * 1.14, MAX_STEP)))
        if step >= MAX_STEP:
            break
    return _silent_packet(count, channels), None


def minimum_size(count: int, channels: int) -> int:
    """Smallest stream that can represent `count` frames: header plus empty packets."""
    frame_count = (count + HOP - 1) // HOP + 1
    total = HEADER.size + CRC.size
    for start in range(0, frame_count, GROUP):
        total += len(_silent_packet(min(GROUP, frame_count - start), channels))
    return total


def encode(samples, target_bytes: int) -> bytes:
    """Encode float32 (n, channels) PCM into at most `target_bytes` bytes.

    Raises `CodecError` if `target_bytes` is below `minimum_size`, rather than
    returning an oversized stream: the byte budget is a hard contract, and a
    caller comparing codecs at matched size needs a failure, not a silent
    overrun.
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] not in (1, 2):
        raise CodecError("Expected [samples, 1 or 2 channels].")
    if not 0 < len(samples) <= MAX_SAMPLES or not np.isfinite(samples).all():
        raise CodecError("Expected finite, nonempty audio of at most 10 minutes.")
    if np.max(np.abs(samples)) > 1:
        raise CodecError("Input PCM must be within [-1, 1].")
    count, channels = samples.shape
    floor = minimum_size(count, channels)
    if target_bytes < floor:
        raise CodecError(
            f"Byte budget {target_bytes} is below the {floor}-byte minimum for "
            f"{count} samples in {channels} channel(s).")
    header = HEADER.pack(MAGIC, RATE, channels, HOP, count)
    output = bytearray(header + CRC.pack(zlib.crc32(header)))
    frame_count = (count + HOP - 1) // HOP + 1
    padded = np.pad(samples, ((HOP, (frame_count + 1) * HOP - count - HOP), (0, 0)))
    frames_left = frame_count
    calib = 1.05
    for start in range(0, frame_count, GROUP):
        group_count = min(GROUP, frame_count - start)
        segment = padded[start * HOP:(start + group_count + 1) * HOP]
        frames = np.lib.stride_tricks.sliding_window_view(segment, 2 * HOP, axis=0)[::HOP]
        remaining = target_bytes - len(output)
        budget = (remaining * group_count) // frames_left - PACKET.size
        packet, ratio = encode_group(mdct(frames), budget, calib)
        if len(packet) > remaining:
            packet = _silent_packet(group_count, channels)
            ratio = None
        if len(packet) > remaining:
            break
        output.extend(packet)
        if ratio is not None:
            calib = 0.65 * calib + 0.35 * min(1.6, max(0.85, ratio))
        frames_left -= group_count
    return bytes(output)


def _noise_fill(coefficients, signed, band_rms, seed):
    """Replace zero-quantized bins with noise matching the transmitted energy."""
    generator = np.random.default_rng((0x9E3779B9 ^ int(seed)) & 0xFFFFFFFF)
    noise = generator.standard_normal(coefficients.shape).astype(np.float32)
    for band in range(BANDS):
        weight = float(FILL[band])
        if weight <= 0.0:
            continue
        left, right = int(EDGES[band]), int(EDGES[band + 1])
        block = coefficients[:, :, left:right]
        zeros = signed[:, :, left:right] == 0
        decoded = np.sum(block * block, axis=-1)
        target = np.float32(right - left) * (band_rms[:, :, band] ** 2)
        deficit = np.maximum(target - decoded, np.float32(0.0)) * np.float32(weight)
        shaped = noise[:, :, left:right] * zeros
        energy = np.sum(shaped * shaped, axis=-1)
        usable = (energy > np.float32(1e-30)) & (deficit > np.float32(0.0))
        gain = np.zeros_like(energy)
        gain[usable] = np.sqrt(deficit[usable] / energy[usable])
        block += shaped * gain[..., None]


def decode_group(side_blob, coef_blob, count, channels, step, codec, seed, alloc_codes):
    mode_size = (count * BANDS + 7) // 8 if channels == 2 else 0
    scale_size = count * channels * BANDS
    raw = _decompress(side_blob, codec)
    if len(raw) != mode_size + scale_size:
        raise CodecError("Invalid side-information length.")
    modes = np.zeros((count, BANDS), dtype=bool)
    if channels == 2:
        modes = np.unpackbits(np.frombuffer(raw[:mode_size], dtype=np.uint8),
                              bitorder="little")[:count * BANDS].reshape(count, BANDS).astype(bool)
    deltas = np.frombuffer(raw, dtype=np.uint8, count=scale_size,
                           offset=mode_size).reshape(channels, BANDS, count).transpose(2, 0, 1)
    scales = np.cumsum(deltas, axis=0, dtype=np.uint32).astype(np.uint8)
    band_rms, base = derive_steps(scales)
    alloc = _unpack_alloc(alloc_codes)
    steps = base * _alloc_bins(alloc)[None, None, :] * np.float32(step)

    signed = _decode_coeffs(RangeDecoder(coef_blob), _new_probs(), count, channels)
    magnitude = np.abs(signed).astype(np.float32)
    amplitude = np.where(signed != 0, magnitude + np.float32(RECON), np.float32(0.0))
    coefficients = (np.sign(signed).astype(np.float32) * amplitude * steps).astype(np.float32)
    _noise_fill(coefficients, signed, band_rms, seed)

    if channels == 2:
        for band in range(BANDS):
            left, right = int(EDGES[band]), int(EDGES[band + 1])
            selected = modes[:, band]
            if not np.any(selected):
                continue
            mid = coefficients[selected, 0, left:right].copy()
            side = coefficients[selected, 1, left:right].copy()
            coefficients[selected, 0, left:right] = (mid + side) * np.float32(2 ** -0.5)
            coefficients[selected, 1, left:right] = (mid - side) * np.float32(2 ** -0.5)
    return imdct(coefficients)


def decode(payload: bytes):
    """Decode to float32 (n, channels); validates framing, lengths, checksums."""
    data = payload
    if len(data) < HEADER.size + CRC.size:
        raise CodecError("Truncated header.")
    magic, rate, channels, hop, count = HEADER.unpack_from(data)
    if CRC.unpack_from(data, HEADER.size)[0] != zlib.crc32(data[:HEADER.size]):
        raise CodecError("Header checksum mismatch.")
    if magic != MAGIC or rate != RATE or hop != HOP or channels not in (1, 2):
        raise CodecError("Unsupported Mosaic format.")
    if not 0 < count <= MAX_SAMPLES:
        raise CodecError("Invalid sample count.")
    frame_count = (count + HOP - 1) // HOP + 1
    offset = HEADER.size + CRC.size
    packets = []
    for start in range(0, frame_count, GROUP):
        if len(data) - offset < PACKET.size:
            break
        unpacked = PACKET.unpack_from(data, offset)
        frames, step, side_len, coef_len, codec = unpacked[:5]
        alloc_codes = unpacked[5:5 + N_BUCKET]
        checksum = unpacked[5 + N_BUCKET]
        fields = data[offset:offset + PACKET_FIELDS.size]
        offset += PACKET.size
        if (frames != min(GROUP, frame_count - start) or not math.isfinite(step)
                or not 0.005 <= step <= 2.0e6
                or codec not in (CODEC_LZMA, CODEC_BZ2, CODEC_ZLIB)):
            raise CodecError("Invalid packet parameters.")
        span = frames * channels * (BANDS + HOP) * 4 + 8192
        if side_len > span or coef_len > span or offset + side_len + coef_len > len(data):
            raise CodecError("Invalid packet length.")
        side_blob = data[offset:offset + side_len]
        coef_blob = data[offset + side_len:offset + side_len + coef_len]
        if zlib.crc32(fields + side_blob + coef_blob) != checksum:
            raise CodecError("Packet checksum mismatch.")
        packets.append((start, frames, step, codec, side_blob, coef_blob, alloc_codes))
        offset += side_len + coef_len
    if offset != len(data):
        raise CodecError("Trailing bytes after final packet.")
    output = np.zeros(((frame_count + 1) * HOP, channels), dtype=np.float32)
    for start, frames, step, codec, side_blob, coef_blob, alloc_codes in packets:
        contributions = decode_group(side_blob, coef_blob, frames, channels, step, codec,
                                     start, alloc_codes)
        output[start * HOP:(start + frames) * HOP] += (
            contributions[..., :HOP].transpose(0, 2, 1).reshape(-1, channels))
        output[(start + 1) * HOP:(start + frames + 1) * HOP] += (
            contributions[..., HOP:].transpose(0, 2, 1).reshape(-1, channels))
    result = output[HOP:HOP + count]
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def read_wav(path):
    """48 kHz 16-bit PCM WAV -> float32 (n, channels) in [-1, 1]."""
    with wave.open(str(path), "rb") as audio:
        if (audio.getframerate() != RATE or audio.getsampwidth() != 2
                or audio.getnchannels() not in (1, 2)):
            raise CodecError("Input must be 48 kHz, 16-bit PCM WAV, mono or stereo.")
        if not 0 < audio.getnframes() <= MAX_SAMPLES:
            raise CodecError("Input must be nonempty and at most 10 minutes.")
        frames, channels = audio.getnframes(), audio.getnchannels()
        raw = audio.readframes(frames)
    if len(raw) != frames * channels * 2:
        raise CodecError("Truncated WAV data.")
    return np.frombuffer(raw, dtype="<i2").reshape(-1, channels).astype(np.float32) / 32768


def pcm16(samples):
    return np.clip(np.rint(samples * 32768), -32768, 32767).astype("<i2")


def publish(destination, data):
    """Write a complete file without replacing an existing output."""
    destination = Path(destination).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".mosaic-",
                                     delete=False) as staging:
        staged = Path(staging.name)
        try:
            staging.write(data)
            staging.flush()
            os.link(staged, destination)
        finally:
            staged.unlink(missing_ok=True)


def write_wav(destination, samples, rate=RATE):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(samples.shape[1])
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(samples.tobytes())
    publish(destination, buffer.getvalue())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Mosaic v2 encoder/decoder (.ms2).")
    commands = parser.add_subparsers(dest="command", required=True)

    encoder = commands.add_parser("encode", help="Encode PCM WAV to a new .ms2 file.")
    encoder.add_argument("input", type=Path)
    encoder.add_argument("output", type=Path)
    budget = encoder.add_mutually_exclusive_group(required=True)
    budget.add_argument("--bitrate", type=float,
                        help="Total target kb/s across channels, including all overhead.")
    budget.add_argument("--bytes", type=int, dest="byte_budget",
                        help="Exact byte budget for the whole file.")

    decoder = commands.add_parser("decode", help="Decode .ms2 to PCM WAV.")
    decoder.add_argument("input", type=Path)
    decoder.add_argument("output", type=Path, nargs="?")
    decoder.add_argument("--null", action="store_true",
                         help="Decode to PCM16 without writing a file.")

    args = parser.parse_args(argv)
    try:
        if args.command == "encode":
            if args.output.exists():
                raise CodecError(f"Output exists: {args.output}")
            samples = read_wav(args.input)
            if args.byte_budget is not None:
                target = args.byte_budget
            else:
                if not math.isfinite(args.bitrate) or not 16 <= args.bitrate <= 1024:
                    raise CodecError("Bitrate must be in [16, 1024] kb/s.")
                target = math.floor(args.bitrate * 1000 * len(samples) / RATE / 8)
            publish(args.output, encode(samples, target))
        else:
            if not args.null and args.output is None:
                raise CodecError("Specify a WAV output or --null.")
            if args.output is not None and args.output.exists():
                raise CodecError(f"Output exists: {args.output}")
            if args.input.stat().st_size > MAX_SAMPLES * 6:
                raise CodecError("Input exceeds the format size limit.")
            samples = pcm16(decode(args.input.read_bytes()))
            if not args.null:
                write_wav(args.output, samples)
    except (OSError, CodecError, wave.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
