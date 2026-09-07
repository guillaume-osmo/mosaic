"""Mosaic: a lossy music codec.

An original implementation of established transform-coding techniques. The codec
core invokes no existing audio codec; NumPy and SciPy provide the transform and
array primitives, and the standard library provides the entropy stage.

Sine-windowed MDCT (1024-sample window, 512 hop, folded into an orthonormal
DCT-IV), auditory-scale bands, per-band energy transmitted as quantized
log2(RMS), quantizer steps derived from that energy by a rule both sides run
identically, dead-zone scalar quantization, and parametric noise substitution:
the decoder refills bins that quantized to zero with energy-matched noise so an
unaffordable band is reconstructed with the correct spectral envelope rather
than as silence.

Encoding targets a byte budget rather than a bitrate, because that is the only
basis on which codecs compare fairly -- see RESULTS.md.

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

VERSION = 2

RATE = 48000
HOP = 512
GROUP = 64
MAX_SAMPLES = RATE * 600
MAGIC = b"MSC2"
HEADER = struct.Struct("<4sIBHQB")
CRC = struct.Struct("<I")
PACKET = struct.Struct("<HfII")
PACKET_FIELDS = struct.Struct("<HfI")

MIN_BAND_BINS = 3
THETA = 0.35                 # dead-zone quantizer offset; zero bin is +-0.65 step
RECON = 0.5 - THETA          # reconstruction point inside a nonzero cell
EMPH_LOW = 0.85              # finer steps below 900 Hz (the measured bass deficit)
EMPH_TOP = 1.8               # coarser steps at the top; noise fill carries them
MASK_FRAC = 0.25             # within-band post-masking reference, -12 dB
DECAY = 0.65                 # per-frame decay of that reference
FLOOR_FRAC = 0.08            # frame-level absolute floor, from the local percentile
FILL_LOW_HZ = 250.0          # no noise substitution below here
LZMA_PRESET = 0

CODEC_LZMA, CODEC_BZ2, CODEC_ZLIB = 0, 1, 2
COMPRESSOR = CODEC_LZMA if lzma is not None else (CODEC_BZ2 if bz2 is not None else CODEC_ZLIB)


class CodecError(ValueError):
    """Invalid audio, unsupported format, corrupt bitstream or impossible rate."""


def _compress(raw: bytes) -> bytes:
    if COMPRESSOR == CODEC_LZMA:
        return lzma.compress(raw, format=lzma.FORMAT_ALONE, preset=LZMA_PRESET)
    if COMPRESSOR == CODEC_BZ2:
        return bz2.compress(raw, compresslevel=9)
    return zlib.compress(raw, level=1)


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
        raise CodecError("Unknown compressor id in header.")
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

_BIN_HZ = RATE / (2.0 * HOP)
_LOW_HZ = np.maximum(EDGES[:-1] * _BIN_HZ, 20.0)
_HIGH_HZ = np.maximum(EDGES[1:] * _BIN_HZ, 40.0)
CENTER_HZ = np.sqrt(_LOW_HZ * _HIGH_HZ)

EMPH = np.ones(BANDS, dtype=np.float32)
EMPH[CENTER_HZ < 900.0] = np.float32(EMPH_LOW)
_TOP = CENTER_HZ > 4000.0
if np.any(_TOP):
    _ramp = np.clip(np.log2(CENTER_HZ[_TOP] / 4000.0) / np.log2(5.0), 0.0, 1.0)
    EMPH[_TOP] = (1.0 + (EMPH_TOP - 1.0) * _ramp).astype(np.float32)

FILL = np.clip(np.log2(CENTER_HZ / FILL_LOW_HZ) / 2.0, 0.0, 1.0).astype(np.float32)


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


def derive_steps(scales):
    """Rebuild band RMS and per-bin quantizer steps from the transmitted scales.

    Run identically by encoder and decoder -- everything here comes out of the
    uint8 scale bytes that are actually in the bitstream, so the noise-fill
    target energy and the step size never disagree between the two sides.
    """
    count, channels, _ = scales.shape
    rms = np.exp2((scales.astype(np.float32) - np.float32(128.0)) / np.float32(8.0))
    rms = np.where(scales > 0, rms, np.float32(0.0)).astype(np.float32)

    # Within-band post-masking: a decaying tail may be quantized relative to
    # the transient that preceded it, not to its own (much smaller) energy.
    reference = rms.copy()
    decay = np.float32(DECAY)
    for index in range(1, count):
        np.maximum(rms[index], reference[index - 1] * decay, out=reference[index])

    power = (rms * rms) * WIDTHS
    frame_rms = np.sqrt(
        power.sum(axis=(1, 2)) / np.float32(channels * HOP) + np.float32(1e-30)
    ).astype(np.float32)
    floor = _local_floor(frame_rms) * np.float32(FLOOR_FRAC)

    base = np.maximum(rms, reference * np.float32(MASK_FRAC)) * EMPH
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


def quantize(normalized, side_info, step):
    """Dead-zone scalar quantization, zigzag, byte-plane split, entropy code."""
    step = float(np.float32(step))
    scaled = normalized / np.float32(step)
    magnitude = np.floor(np.abs(scaled) + np.float32(THETA))
    magnitude = np.clip(magnitude, 0.0, 32000.0)
    signed = np.where(scaled < 0, -magnitude, magnitude).astype(np.int32)
    zigzag = ((signed << 1) ^ (signed >> 31)).astype(np.uint16).transpose(1, 2, 0)
    raw = (side_info + (zigzag & 255).astype(np.uint8).tobytes()
           + (zigzag >> 8).astype(np.uint8).tobytes())
    return _compress(raw), step


def _silent_payload(count, channels):
    mode_size = (count * BANDS + 7) // 8 if channels == 2 else 0
    total = mode_size + count * channels * BANDS + 2 * count * channels * HOP
    return _compress(bytes(total))


def encode_group(coefficients, budget):
    count, channels, _ = coefficients.shape
    normalized, side_info = prepare_group(coefficients)
    low, high = 0.01, 1.0
    payload, step = quantize(normalized, side_info, high)
    while len(payload) > budget and high < 65536:
        low, high = high, high * 2
        payload, step = quantize(normalized, side_info, high)
    if len(payload) > budget:
        payload, step = _silent_payload(count, channels), 1.0
    else:
        best = payload, step
        for _ in range(8):
            middle = math.sqrt(low * high)
            candidate, candidate_step = quantize(normalized, side_info, middle)
            if len(candidate) <= budget:
                high, best = middle, (candidate, candidate_step)
            else:
                low = middle
        payload, step = best
    fields = PACKET_FIELDS.pack(count, step, len(payload))
    return fields + CRC.pack(zlib.crc32(fields + payload)) + payload


def minimum_size(count: int, channels: int) -> int:
    """Smallest stream that can represent `count` frames: header plus empty packets."""
    frame_count = (count + HOP - 1) // HOP + 1
    groups = (frame_count + GROUP - 1) // GROUP
    total = HEADER.size + CRC.size
    for start in range(0, frame_count, GROUP):
        group_count = min(GROUP, frame_count - start)
        total += PACKET.size + len(_silent_payload(group_count, channels))
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
    header = HEADER.pack(MAGIC, RATE, channels, HOP, count, COMPRESSOR)
    output = bytearray(header + CRC.pack(zlib.crc32(header)))
    frame_count = (count + HOP - 1) // HOP + 1
    padded = np.pad(samples, ((HOP, (frame_count + 1) * HOP - count - HOP), (0, 0)))
    frames_left = frame_count
    for start in range(0, frame_count, GROUP):
        group_count = min(GROUP, frame_count - start)
        segment = padded[start * HOP:(start + group_count + 1) * HOP]
        frames = np.lib.stride_tricks.sliding_window_view(segment, 2 * HOP, axis=0)[::HOP]
        remaining = target_bytes - len(output)
        budget = (remaining * group_count) // frames_left - PACKET.size
        packet = encode_group(mdct(frames), budget)
        if len(packet) > remaining:
            fallback = _silent_payload(group_count, channels)
            fields = PACKET_FIELDS.pack(group_count, 1.0, len(fallback))
            packet = fields + CRC.pack(zlib.crc32(fields + fallback)) + fallback
        output.extend(packet)
        frames_left -= group_count
    if len(output) > target_bytes:
        raise CodecError(
            f"Internal error: produced {len(output)} bytes for a {target_bytes}-byte "
            f"budget.")
    return bytes(output)


def _noise_fill(coefficients, signed, band_rms, seed):
    """Replace zero-quantized bins with noise matching the transmitted energy.

    This is the whole point of the format change: a band the budget could not
    afford is reconstructed with the right log-power envelope instead of as
    silence. The deficit is measured against what actually decoded, so bands
    that survived quantization receive essentially nothing.
    """
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


def decode_group(payload, count, channels, step, codec, seed):
    mode_size = (count * BANDS + 7) // 8 if channels == 2 else 0
    scale_size = count * channels * BANDS
    coefficient_count = count * channels * HOP
    expected = mode_size + scale_size + 2 * coefficient_count
    raw = _decompress(payload, codec)
    if len(raw) != expected:
        raise CodecError("Invalid decompressed packet length.")
    modes = np.zeros((count, BANDS), dtype=bool)
    if channels == 2:
        modes = np.unpackbits(np.frombuffer(raw[:mode_size], dtype=np.uint8),
                              bitorder="little")[:count * BANDS].reshape(count, BANDS).astype(bool)
    deltas = np.frombuffer(raw, dtype=np.uint8, count=scale_size,
                           offset=mode_size).reshape(channels, BANDS, count).transpose(2, 0, 1)
    scales = np.cumsum(deltas, axis=0, dtype=np.uint32).astype(np.uint8)
    band_rms, base = derive_steps(scales)
    steps = base * np.float32(step)

    offset = mode_size + scale_size
    low = np.frombuffer(raw, dtype=np.uint8, count=coefficient_count,
                        offset=offset).astype(np.uint16)
    high = np.frombuffer(raw, dtype=np.uint8, count=coefficient_count,
                         offset=offset + coefficient_count).astype(np.uint16)
    zigzag = (low | (high << 8)).astype(np.int32)
    signed = ((zigzag >> 1) ^ -(zigzag & 1)).reshape(channels, HOP, count).transpose(2, 0, 1)
    signed = np.ascontiguousarray(signed)

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
    magic, rate, channels, hop, count, codec = HEADER.unpack_from(data)
    if CRC.unpack_from(data, HEADER.size)[0] != zlib.crc32(data[:HEADER.size]):
        raise CodecError("Header checksum mismatch.")
    if magic != MAGIC or rate != RATE or hop != HOP or channels not in (1, 2):
        raise CodecError("Unsupported Mosaic format.")
    if codec not in (CODEC_LZMA, CODEC_BZ2, CODEC_ZLIB):
        raise CodecError("Unsupported entropy coder id.")
    if not 0 < count <= MAX_SAMPLES:
        raise CodecError("Invalid sample count.")
    frame_count = (count + HOP - 1) // HOP + 1
    offset = HEADER.size + CRC.size
    packets = []
    for start in range(0, frame_count, GROUP):
        if len(data) - offset < PACKET.size:
            raise CodecError("Truncated packet header.")
        frames, step, length, checksum = PACKET.unpack_from(data, offset)
        fields = data[offset:offset + PACKET_FIELDS.size]
        offset += PACKET.size
        if (frames != min(GROUP, frame_count - start) or not math.isfinite(step)
                or not float(np.float32(0.01)) <= step <= 65536):
            raise CodecError("Invalid packet parameters.")
        if length > frames * channels * HOP * 3 + 4096 or offset + length > len(data):
            raise CodecError("Invalid packet length.")
        chunk = data[offset:offset + length]
        if zlib.crc32(fields + chunk) != checksum:
            raise CodecError("Packet checksum mismatch.")
        packets.append((start, frames, step, chunk))
        offset += length
    if offset != len(data):
        raise CodecError("Trailing bytes after final packet.")
    output = np.zeros(((frame_count + 1) * HOP, channels), dtype=np.float32)
    for start, frames, step, chunk in packets:
        contributions = decode_group(chunk, frames, channels, step, codec, start)
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
