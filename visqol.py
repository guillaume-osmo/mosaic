"""ViSQOL v3 audio mode, in NumPy. Derived from google/visqol (Apache-2.0).

A gammatone spectrogram compared patch-by-patch under NSIM, mapped to MOS-LQO
in [1, 5] by the nu-SVR that ships with the reference implementation
(`models/libsvm_nu_svr_model.txt`). Higher is better; a file scored against
itself gives 4.7321, which is the ceiling.

Validated against `src/include/conformance.h` from the reference repository --
the published output of Google's own binary on its shipped test pairs --
over nine degraded pairs spanning MOS 1.39 to 4.73:

    Spearman rho 0.9833, mean absolute error 0.0310 MOS, bias +0.022.
    The six real-codec conditions (AAC 24/48/64/128/256, MP3 96, Opus 128)
    agree within 0.01 MOS.

Not ported: speech mode, the TFLite lattice mapper, VAD patch creation, and
per-patch realignment beyond the integer frame search.

Alignment is load-bearing, not a detail: Opus and MP3 reconstruct late, while a
codec that trims to the exact original length comes out sample-aligned. Scoring
without compensating charges the reference codecs for a delay nobody can hear.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import hilbert, lfilter

SR = 48000
N_BANDS = 32
MIN_FREQ = 50.0
WINDOW_S = 0.08
OVERLAP = 0.25
PATCH_FRAMES = 30
SEARCH_RADIUS = 60
FLOOR_ABS_DB = -45.0
FLOOR_REL_DB = 45.0
NSIM_C1 = 0.01 ** 2
NSIM_C3 = 0.03 ** 2 / 2.0
SPL_REF = 2e-5
MODEL_PATH = Path(__file__).resolve().parent / "models/libsvm_nu_svr_model.txt"

NSIM_KERNEL = np.array([
    [0.0113033910173052, 0.0838251475442633, 0.0113033910173052],
    [0.0838251475442633, 0.619485845753726, 0.0838251475442633],
    [0.0113033910173052, 0.0838251475442633, 0.0113033910173052],
])


def read_wav_mono(path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{path}: 16-bit PCM only")
        channels = handle.getnchannels()
        raw = handle.readframes(handle.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return data.reshape(-1, channels).mean(axis=1) if channels > 1 else data


def erb_center_freqs(low: float, high: float, n: int) -> np.ndarray:
    ear_q, min_bw = 9.26449, 24.7
    d = high + ear_q * min_bw
    e = (-np.log(high + ear_q * min_bw) + np.log(low + ear_q * min_bw)) / n
    return -(ear_q * min_bw) + np.exp(np.arange(1, n + 1) * e) * d


def erb_filters(sr: int = SR, n: int = N_BANDS, low: float = MIN_FREQ, high=None):
    """Slaney gammatone coefficients: (denominator, [4 numerators]) per band."""
    high = sr / 2.0 if high is None else min(high, sr / 2.0)
    cf = erb_center_freqs(low, high, n)
    erb = cf / 9.26449 + 24.7
    bw = 1.019 * 2 * np.pi * erb
    t = 1.0 / sr
    exp_bt = np.exp(bw * t)
    b1 = -2.0 * np.cos(2.0 * cf * np.pi * t) / exp_bt
    b2 = np.exp(-2.0 * bw * t)
    gt_b = np.sin(2.0 * cf * np.pi * t) * t
    b_pos = gt_b * 2.0 * np.sqrt(3.0 + 2.0 ** 1.5)
    b_neg = gt_b * 2.0 * np.sqrt(3.0 - 2.0 ** 1.5)
    a = np.cos(2.0 * cf * np.pi * t) * 2.0 * t
    numerators = [-(a / exp_bt + b_pos / exp_bt) / 2.0, -(a / exp_bt - b_pos / exp_bt) / 2.0,
                  -(a / exp_bt + b_neg / exp_bt) / 2.0, -(a / exp_bt - b_neg / exp_bt) / 2.0]

    j = 1j
    s1, s2 = np.sqrt(3.0 - 2.0 ** 1.5), np.sqrt(3.0 + 2.0 ** 1.5)
    x_exp = np.exp(4.0 * j * cf * np.pi * t)
    x01 = -2.0 * x_exp * t
    x02 = 2.0 * np.exp(-(bw * t) + 2.0 * j * cf * np.pi * t) * t
    cos_t, sin_t = np.cos(2.0 * cf * np.pi * t), np.sin(2.0 * cf * np.pi * t)
    x5 = (-2.0 / np.exp(2 * bw * t)) - 2 * x_exp + 2 * (1 + x_exp) / np.exp(bw * t)
    gain = np.abs(np.prod([x01 + x02 * (cos_t + sign * scale * sin_t)
                           for sign, scale in ((-1, s1), (1, s1), (-1, s2), (1, s2))],
                          axis=0) / x5 ** 4)

    bands = []
    for i in range(n):
        den = np.array([1.0, b1[i], b2[i]])
        nums = [np.array([t / gain[i], numerators[0][i] / gain[i], 0.0])]
        nums += [np.array([t, numerators[k][i], 0.0]) for k in (1, 2, 3)]
        bands.append((den, nums))
    return bands[::-1]


def gammatone_spectrogram(signal: np.ndarray, sr: int = SR, bands=None) -> np.ndarray:
    """(n_bands, n_frames) per-band RMS. Filter state resets each frame."""
    bands = erb_filters(sr) if bands is None else bands
    size = int(round(sr * WINDOW_S))
    hop = int(size * OVERLAP)
    if signal.size <= size:
        raise ValueError(f"too few samples ({signal.size}) for a {size}-sample window")
    n_frames = 1 + (signal.size - size) // hop
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(size) / (size - 1))
    frames = sliding_window_view(signal, size)[::hop][:n_frames] * window
    out = np.empty((len(bands), n_frames))
    for i, (den, nums) in enumerate(bands):
        y = frames
        for num in nums:
            y = lfilter(num, den, y, axis=1)
        out[i] = np.sqrt(np.mean(y * y, axis=1))
    return out


def globally_align(reference: np.ndarray, degraded: np.ndarray):
    """Remove one integer lag, matched on the upper envelope."""
    n = min(reference.size, degraded.size)
    ref_env = np.abs(hilbert(reference[:n]))
    deg_env = np.abs(hilbert(degraded[:n]))
    size = 1 << int(np.ceil(np.log2(2 * n)))
    corr = np.fft.irfft(np.fft.rfft(ref_env - ref_env.mean(), size)
                        * np.conj(np.fft.rfft(deg_env - deg_env.mean(), size)), size)
    lag = int(np.argmax(corr))
    if lag > size // 2:
        lag -= size
    if lag == 0 or abs(lag) > n / 2.0:
        return reference, degraded, 0
    if lag < 0:
        degraded = degraded[-lag:]
    else:
        reference = reference[lag:]
    m = min(reference.size, degraded.size)
    return reference[:m], degraded[:m], lag


def _prepare(ref: np.ndarray, deg: np.ndarray):
    to_db = lambda x: 10.0 * np.log10(np.maximum(np.abs(x), np.finfo(np.float64).eps))
    ref, deg = to_db(ref), to_db(deg)
    ref = np.maximum(ref, FLOOR_ABS_DB)
    deg = np.maximum(deg, FLOOR_ABS_DB)
    k = min(ref.shape[1], deg.shape[1])
    floor = np.maximum(ref[:, :k].max(axis=0), deg[:, :k].max(axis=0)) - FLOOR_REL_DB
    ref[:, :k] = np.maximum(ref[:, :k], floor[None, :])
    deg[:, :k] = np.maximum(deg[:, :k], floor[None, :])
    lowest = min(ref.min(), deg.min())
    return ref - lowest, deg - lowest


def _conv_same(a: np.ndarray) -> np.ndarray:
    view = sliding_window_view(np.pad(a, 1), NSIM_KERNEL.shape)
    return np.einsum("ijkl,kl->ij", view, NSIM_KERNEL)


def patch_nsim(ref_patch: np.ndarray, deg_patch: np.ndarray):
    """NSIM is SSIM without the contrast term: luminance times structure."""
    mu_r, mu_d = _conv_same(ref_patch), _conv_same(deg_patch)
    mu_r_sq, mu_d_sq, mu_rd = mu_r * mu_r, mu_d * mu_d, mu_r * mu_d
    var_r = _conv_same(ref_patch * ref_patch) - mu_r_sq
    var_d = _conv_same(deg_patch * deg_patch) - mu_d_sq
    cov = _conv_same(ref_patch * deg_patch) - mu_rd
    intensity = (2.0 * mu_rd + NSIM_C1) / (mu_r_sq + mu_d_sq + NSIM_C1)
    product = var_r * var_d
    denom = np.where(product < 0.0, NSIM_C3, np.sqrt(np.maximum(product, 0.0)) + NSIM_C3)
    per_band = (intensity * (cov + NSIM_C3) / denom).mean(axis=1)
    return float(per_band.mean()), per_band


def measure(reference: np.ndarray, degraded: np.ndarray, sr: int = SR,
            search_radius: int = SEARCH_RADIUS, bands=None):
    """Per-band NSIM (32,), its mean, and the lag removed."""
    ref_spl = 20.0 * np.log10(np.sqrt(np.mean(reference ** 2)) / SPL_REF)
    deg_spl = 20.0 * np.log10(np.sqrt(np.mean(degraded ** 2)) / SPL_REF)
    degraded = degraded * 10.0 ** ((ref_spl - deg_spl) / 20.0)
    reference, degraded, lag = globally_align(reference, degraded)
    bands = erb_filters(sr) if bands is None else bands
    ref_spec = gammatone_spectrogram(reference, sr, bands)
    deg_spec = gammatone_spectrogram(degraded, sr, bands)
    k = min(ref_spec.shape[1], deg_spec.shape[1])
    ref_spec, deg_spec = _prepare(ref_spec[:, :k], deg_spec[:, :k])

    init = PATCH_FRAMES // 2
    if k < PATCH_FRAMES + init:
        raise ValueError(f"too short ({k} frames) for a {PATCH_FRAMES}-frame patch")
    limit = k - PATCH_FRAMES if init < k - PATCH_FRAMES else init + 1
    per_patch = []
    for start in range(init, limit, PATCH_FRAMES):
        ref_patch = ref_spec[:, start:start + PATCH_FRAMES]
        best, best_bands = -np.inf, None
        for offset in range(max(0, start - search_radius),
                            min(k - PATCH_FRAMES, start + search_radius) + 1):
            score, per_band = patch_nsim(ref_patch, deg_spec[:, offset:offset + PATCH_FRAMES])
            if score > best:
                best, best_bands = score, per_band
        per_patch.append(best_bands)
    fvnsim = np.mean(np.stack(per_patch, axis=0), axis=0)
    return fvnsim, float(fvnsim.mean()), lag


class NuSVR:
    """libsvm nu-SVR with an RBF kernel, over the 32 per-band NSIM values."""

    def __init__(self, gamma: float, rho: float, coef: np.ndarray, sv: np.ndarray):
        self.gamma, self.rho, self.coef, self.sv = gamma, rho, coef, sv

    @classmethod
    def load(cls, path=MODEL_PATH) -> "NuSVR":
        gamma = rho = None
        rows, n_features, in_sv = [], 0, False
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if not in_sv:
                if line == "SV":
                    in_sv = True
                elif line.startswith("gamma "):
                    gamma = float(line.split()[1])
                elif line.startswith("rho "):
                    rho = float(line.split()[1])
                elif line.startswith("kernel_type ") and line.split()[1] != "rbf":
                    raise ValueError(f"only the RBF kernel is implemented: {line}")
                continue
            parts = line.split()
            values = {}
            for token in parts[1:]:
                index, _, value = token.partition(":")
                values[int(index)] = float(value)
                n_features = max(n_features, int(index))
            rows.append((float(parts[0]), values))
        if gamma is None or rho is None or not rows:
            raise ValueError(f"{path} is not a usable libsvm model")
        sv = np.zeros((len(rows), n_features))
        coef = np.empty(len(rows))
        for i, (c, values) in enumerate(rows):
            coef[i] = c
            for index, value in values.items():
                sv[i, index - 1] = value
        return cls(gamma, rho, coef, sv)

    def predict(self, x: np.ndarray, clamp: bool = True) -> float:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != self.sv.shape[1]:
            raise ValueError(f"expected {self.sv.shape[1]} features, got {x.size}")
        d2 = ((self.sv - x[None, :]) ** 2).sum(axis=1)
        value = float(self.coef @ np.exp(-self.gamma * d2) - self.rho)
        return float(min(5.0, max(1.0, value))) if clamp else value


def mos(reference_path, degraded_path, model=None, bands=None) -> float:
    """MOS-LQO in [1, 5] for a pair of 48 kHz 16-bit WAV files."""
    model = NuSVR.load() if model is None else model
    fvnsim, _, _ = measure(read_wav_mono(reference_path),
                           read_wav_mono(degraded_path), bands=bands)
    return model.predict(fvnsim)
