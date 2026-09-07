"""Compare Mosaic against Opus and MP3 at matched file size.

Codecs are compared at equal BYTES per excerpt, not at equal nominal bitrate.
Opus is VBR and MP3 carries framing, so `-b:a 96k` does not produce 96 kb/s: on
the corpus in RESULTS.md it produces 116.1 kb/s. Lining up "Mosaic 96" against
"Opus 96" therefore scores a 96 kb/s codec against a 116 kb/s one. Here the
reference codec is encoded first, its actual output size is measured, and Mosaic
is held to exactly that budget.

    python benchmark.py REFERENCE.wav [REFERENCE.wav ...] --bitrate 96

Requires ffmpeg on PATH for the reference codecs only; the Mosaic path does not
use it. Inputs must be 48 kHz 16-bit PCM WAV.
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import mosaic
import visqol

FFMPEG = shutil.which("ffmpeg")


def _run(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-400:]}")


def encode_reference(source: Path, destination: Path, codec: str, kbps: int,
                     complexity: int) -> tuple[int, float]:
    encoder = "libopus" if codec == "opus" else "libmp3lame"
    started = time.perf_counter()
    _run([FFMPEG, "-y", "-v", "error", "-i", str(source), "-c:a", encoder,
          "-b:a", f"{kbps}k", "-compression_level", str(complexity),
          "-ar", "48000", str(destination)])
    return destination.stat().st_size, time.perf_counter() - started


def decode_reference(source: Path, destination: Path, channels: int) -> float:
    started = time.perf_counter()
    _run([FFMPEG, "-y", "-v", "error", "-i", str(source), "-c:a", "pcm_s16le",
          "-ar", "48000", "-ac", str(channels), str(destination)])
    return time.perf_counter() - started


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--bitrate", type=int, default=96,
                        help="reference codec target kb/s; the byte budget comes "
                             "from what it actually produces")
    parser.add_argument("--codec", default="opus", choices=["opus", "mp3"])
    parser.add_argument("--complexity", type=int, default=10,
                        help="reference encoder effort; 10 is Opus at its best")
    parser.add_argument("--keep", type=Path, default=None,
                        help="directory to keep decoded WAVs in, for listening")
    args = parser.parse_args(argv)

    if FFMPEG is None:
        sys.exit("ffmpeg is required on PATH for the reference codecs")

    model = visqol.NuSVR.load()
    bands = visqol.erb_filters()
    work = Path(tempfile.mkdtemp(prefix="mosaic-bench-"))
    keep = args.keep
    if keep is not None:
        keep.mkdir(parents=True, exist_ok=True)

    print(f"  reference: {args.codec} {args.bitrate}k complexity {args.complexity}, "
          f"matched bytes per excerpt")
    print(f"\n{'excerpt':<20}{'bytes':>9}{'kb/s':>7}{'mosaic':>9}{args.codec:>9}"
          f"{'delta':>8}{'enc s':>7}{'dec s':>7}")

    rows = []
    for source in args.inputs:
        samples = mosaic.read_wav(source)
        seconds = len(samples) / mosaic.RATE
        channels = samples.shape[1]

        packed = work / f"{source.stem}.{'opus' if args.codec == 'opus' else 'mp3'}"
        budget, _ = encode_reference(source, packed, args.codec, args.bitrate,
                                     args.complexity)
        reference_wav = (keep or work) / f"{source.stem}.{args.codec}.wav"
        decode_reference(packed, reference_wav, channels)

        started = time.perf_counter()
        payload = mosaic.encode(samples, budget)
        encode_s = time.perf_counter() - started
        started = time.perf_counter()
        restored = mosaic.decode(payload)
        decode_s = time.perf_counter() - started
        if restored.shape != samples.shape:
            sys.exit(f"{source}: decode returned {restored.shape}, expected {samples.shape}")
        if len(payload) > budget:
            sys.exit(f"{source}: {len(payload)} bytes exceeds the {budget}-byte budget")
        mosaic_wav = (keep or work) / f"{source.stem}.mosaic.wav"
        mosaic.write_wav(mosaic_wav, mosaic.pcm16(restored))

        ours = visqol.mos(source, mosaic_wav, model=model, bands=bands)
        theirs = visqol.mos(source, reference_wav, model=model, bands=bands)
        rows.append((source.stem, len(payload), budget, ours, theirs, encode_s, decode_s))
        print(f"{source.stem:<20}{len(payload):>9}{len(payload) * 8 / seconds / 1000:>7.1f}"
              f"{ours:>9.4f}{theirs:>9.4f}{ours - theirs:>+8.4f}{encode_s:>7.2f}{decode_s:>7.2f}")

    deltas = [r[3] - r[4] for r in rows]
    ahead = sum(1 for d in deltas if d > 0)
    print(f"\n  mosaic mean MOS {statistics.mean(r[3] for r in rows):.4f}   "
          f"{args.codec} {statistics.mean(r[4] for r in rows):.4f}   "
          f"delta {statistics.mean(deltas):+.4f}   ahead on {ahead}/{len(rows)}")
    if len(rows) > 2:
        rng = np.random.default_rng(0)
        d = np.array(deltas)
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(20000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  paired bootstrap over excerpts: 95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  -> {'excludes' if lo > 0 else 'INCLUDES'} zero")
    print(f"\n  MOS-LQO is an objective metric, ceiling 4.7321. It is not a "
          f"listening test.")
    if keep is not None:
        print(f"  decoded WAVs kept in {keep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
