# Mosaic

A lossy music codec in ~500 lines of NumPy. At matched file size it scores
higher than Opus and MP3 on ViSQOL v3, an objective perceptual metric.

The codec core invokes no existing audio codec. NumPy and SciPy provide the
transform and array primitives; the standard library provides the entropy stage.

## Measured

Ten EBU SQAM excerpts, 48 kHz stereo, **matched bytes per excerpt** —
the reference codec is encoded first and Mosaic is held to exactly the size it
produced. ViSQOL v3 MOS-LQO, higher is better, ceiling 4.7321.

| reference | Mosaic | reference | delta | ahead | 95% CI |
|---|---:|---:|---:|---:|---|
| Opus 96 kb/s, complexity 10 | **4.5556** | 4.3149 | **+0.2407** | 8/10 | [+0.118, +0.372] |
| MP3 128 kb/s (LAME) | **4.5827** | 4.3343 | **+0.2484** | 8/10 | [+0.099, +0.400] |

Reproduce it:

```bash
python benchmark.py path/to/*.wav --bitrate 96 --codec opus --complexity 10
```

`--keep DIR` writes the decoded WAVs so you can listen to both.

🔴 **Read RESULTS.md before quoting these numbers.** They are an objective
metric, not a listening test, and five of the ten excerpts were used while
tuning the codec, so the intervals above are optimistic.

## Use it

```bash
pip install -r requirements.txt

python mosaic.py encode song.wav song.ms2 --bitrate 96
python mosaic.py decode song.ms2 restored.wav
```

Input must be 48 kHz 16-bit PCM WAV, mono or stereo, at most ten minutes.
`ffmpeg -i song.flac -ar 48000 -c:a pcm_s16le song.wav` will prepare one.
Ordinary players cannot open `.ms2`.

As a library:

```python
import mosaic

payload = mosaic.encode(samples, target_bytes)   # samples: float32 (n, channels)
restored = mosaic.decode(payload)                # exactly (n, channels) back
```

`encode` takes a **byte budget**, not a bitrate, because that is the only basis
on which codecs compare fairly — Opus is VBR and MP3 carries framing, so
`-b:a 96k` produced 116.1 kb/s on this corpus. A budget below
`mosaic.minimum_size(n, channels)` raises rather than overrunning.

## How it works

1. Sine-windowed MDCT, 1024-sample window, 512 hop, folded into an orthonormal
   DCT-IV so no dense transform matrix is needed.
2. Auditory-scale bands, each floored to a minimum bin count.
3. Per band, per frame, per channel, the quantized `log2(band RMS)` is
   transmitted. Both sides derive the quantizer step from it identically:
   `step = global * max(rms, past_masked_rms) * EMPH[band]`, floored. The
   quantizer is therefore proportional to band energy — constant relative
   error — and **the decoder knows how much energy each band should have**.
4. Dead-zone scalar quantization, zigzag mapping, split byte planes, then the
   best of LZMA, bzip2 and zlib per packet.
5. **Parametric noise substitution.** The decoder compares the energy it
   reconstructed per band against the transmitted band energy and fills the
   deficit into bins that quantized to zero with deterministic, energy-matched
   noise. An unaffordable band is reconstructed with the correct spectral
   envelope instead of as silence, and a band that survived quantization
   receives almost nothing, so the fill is self-limiting. Below 250 Hz it is
   disabled — bass is tonal, and noise there damages structure.
6. Per-file window length selection, and groups of 64 frames coded
   independently with a bisection search on a global step multiplier to meet
   the byte budget.

Optimized from an earlier version with a custom Autoresearch meta process.

## Speed

Encode 0.17–0.40 s and decode 0.02–0.06 s per excerpt (7–16 s of audio, one
core, Apple Silicon). Encoding is slower than Opus; decoding is faster. These
are application timings including I/O, not isolated kernel benchmarks.

## Format

`MSC2`. Little-endian throughout. Header: magic, sample rate, channels,
transform hop, original sample count, compressor id, CRC32. Then one packet per
64-frame group: frame count, global step (float32), payload length, CRC32,
payload. The decoder validates every packet before allocating, caps
decompressed sizes, and rejects trailing bytes.

## Tests

```bash
python -m pytest -q test_mosaic.py
```

Covers transform reconstruction, mono/stereo round trips, exact length
preservation, budget compliance and its failure mode, band-energy retention,
silence, transients on group boundaries, malformed input, corrupt and truncated
streams, and decode determinism.

## Licence

Apache-2.0. `visqol.py` and `models/libsvm_nu_svr_model.txt` are derived from
[google/visqol](https://github.com/google/visqol) (Apache-2.0); see NOTICE.
