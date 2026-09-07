"""Codec tests: transform, round trips, budget compliance, bitstream validation."""
from __future__ import annotations

import numpy as np
import pytest

import mosaic


def tone(seconds=1.0, freq=440.0, channels=2, amplitude=0.3, rate=mosaic.RATE):
    t = np.arange(int(seconds * rate)) / rate
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    if channels == 1:
        return signal.astype(np.float32)[:, None]
    return np.stack([signal, 0.8 * signal], axis=1).astype(np.float32)


def noise(seconds=1.0, channels=2, seed=0, rate=mosaic.RATE):
    rng = np.random.default_rng(seed)
    return (0.2 * rng.standard_normal((int(seconds * rate), channels))).astype(np.float32)


def budget_for(samples, kbps):
    return kbps * 1000 * len(samples) // mosaic.RATE // 8


def test_mdct_reconstructs_before_quantization():
    """Overlap-add of the inverse transform must cancel time-domain aliasing."""
    rng = np.random.default_rng(1)
    signal = rng.standard_normal(mosaic.HOP * 8).astype(np.float32)
    padded = np.pad(signal, (mosaic.HOP, mosaic.HOP * 2))
    frames = np.lib.stride_tricks.sliding_window_view(
        padded, 2 * mosaic.HOP, axis=0)[::mosaic.HOP]
    contributions = mosaic.imdct(mosaic.mdct(frames))
    output = np.zeros(len(padded), dtype=np.float64)
    for i, contribution in enumerate(contributions):
        output[i * mosaic.HOP:i * mosaic.HOP + 2 * mosaic.HOP] += contribution
    recovered = output[mosaic.HOP:mosaic.HOP + len(signal)]
    assert np.allclose(recovered, signal, atol=1e-4)


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("kbps", [64, 96, 160])
def test_round_trip_shape_and_budget(channels, kbps):
    samples = tone(channels=channels)
    budget = budget_for(samples, kbps)
    payload = mosaic.encode(samples, budget)
    assert len(payload) <= budget
    restored = mosaic.decode(payload)
    assert restored.shape == samples.shape
    assert np.isfinite(restored).all()


def test_round_trip_preserves_odd_length():
    """The decoder must return the exact original sample count, not a frame multiple."""
    samples = tone(seconds=0.5)[:23_457]
    payload = mosaic.encode(samples, budget_for(samples, 128))
    assert mosaic.decode(payload).shape == samples.shape


def test_tonal_signal_is_reconstructed_accurately():
    samples = tone(seconds=2.0)
    payload = mosaic.encode(samples, budget_for(samples, 160))
    restored = mosaic.decode(payload)
    error = np.mean((restored - samples) ** 2)
    assert 10 * np.log10(np.mean(samples ** 2) / error) > 15.0


def test_noise_fill_restores_band_energy():
    """Parametric substitution should keep total energy near the original.

    A codec that zeroes unaffordable bands loses their energy outright; this one
    refills them, so broadband input at a low budget should retain most of its
    power rather than collapsing towards silence.
    """
    samples = noise(seconds=2.0)
    payload = mosaic.encode(samples, budget_for(samples, 64))
    restored = mosaic.decode(payload)
    ratio = np.mean(restored ** 2) / np.mean(samples ** 2)
    assert 0.5 < ratio < 2.0


def test_silence_round_trips():
    samples = np.zeros((mosaic.RATE, 2), dtype=np.float32)
    payload = mosaic.encode(samples, budget_for(samples, 64))
    restored = mosaic.decode(payload)
    assert restored.shape == samples.shape
    assert np.abs(restored).max() < 1e-3


def test_transient_at_packet_boundary():
    """A click placed on a group boundary must not desynchronise the stream."""
    samples = np.zeros((mosaic.HOP * mosaic.GROUP * 2, 2), dtype=np.float32)
    samples[mosaic.HOP * mosaic.GROUP] = 0.9
    payload = mosaic.encode(samples, budget_for(samples, 128))
    assert mosaic.decode(payload).shape == samples.shape


@pytest.mark.parametrize("samples", [
    np.zeros((10, 3), dtype=np.float32),
    np.zeros(10, dtype=np.float32),
    np.zeros((0, 2), dtype=np.float32),
])
def test_rejects_malformed_input(samples):
    with pytest.raises(mosaic.CodecError):
        mosaic.encode(samples, 4096)


def test_rejects_out_of_range_and_nonfinite():
    for bad in (np.full((1000, 2), 2.0, dtype=np.float32),
                np.full((1000, 2), np.nan, dtype=np.float32)):
        with pytest.raises(mosaic.CodecError):
            mosaic.encode(bad, 4096)


def test_rejects_impossible_budget():
    with pytest.raises(mosaic.CodecError):
        mosaic.encode(tone(), 8)


def test_rejects_corrupt_and_truncated_streams():
    samples = tone()
    payload = mosaic.encode(samples, budget_for(samples, 96))
    with pytest.raises(mosaic.CodecError):
        mosaic.decode(b"NOPE" + payload[4:])
    with pytest.raises(mosaic.CodecError):
        mosaic.decode(payload[:len(payload) // 2])
    with pytest.raises(mosaic.CodecError):
        mosaic.decode(payload + b"\x00" * 8)
    flipped = bytearray(payload)
    flipped[-1] ^= 0xFF
    with pytest.raises(mosaic.CodecError):
        mosaic.decode(bytes(flipped))


def test_decode_is_deterministic():
    """Noise substitution is seeded from the bitstream, so decoding must repeat."""
    samples = noise(seconds=1.0)
    payload = mosaic.encode(samples, budget_for(samples, 64))
    assert np.array_equal(mosaic.decode(payload), mosaic.decode(payload))
