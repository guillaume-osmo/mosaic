# Mosaic

A lossy music codec in ~900 lines of NumPy. At matched file size it scores
higher than Opus and MP3 on ViSQOL v3, an objective perceptual metric.

The codec core invokes no existing audio codec and no general-purpose
compressor. NumPy and SciPy provide the transform and array primitives; the
entropy stage is a context-modelled adaptive binary range coder implemented
here.

## Measured

Ten EBU SQAM excerpts, 48 kHz stereo, **matched bytes per excerpt** —
the reference codec is encoded first and Mosaic is held to exactly the size it
produced. ViSQOL v3 MOS-LQO, higher is better, ceiling 4.7321.

| reference | Mosaic | reference | delta | ahead | 95% CI |
|---|---:|---:|---:|---:|---|
| Opus 96 kb/s, complexity 10 | **4.6224** | 4.3149 | **+0.3075** | 9/10 | [+0.161, +0.474] |
| MP3 128 kb/s (LAME) | **4.6388** | 4.3343 | **+0.3045** | 9/10 | [+0.165, +0.447] |

**None of these ten excerpts was used while tuning the codec.** Sign test
p = 0.011; both intervals exclude zero.

Reproduce it:

```bash
python benchmark.py path/to/*.wav --bitrate 96 --codec opus --complexity 10
```

`--keep DIR` writes the decoded WAVs so you can listen to both.

🔴 **Read RESULTS.md before quoting these numbers.** They are an objective
metric, not a listening test — nobody has heard these files. RESULTS.md also
documents a case where the metric rewards this codec for work that is
inaudible.

## Use it

```bash
pip install -r requirements.txt

python mosaic.py encode song.wav song.ms3 --bitrate 96
python mosaic.py decode song.ms3 restored.wav
```

Input must be 48 kHz 16-bit PCM WAV, mono or stereo, at most ten minutes.
`ffmpeg -i song.flac -ar 48000 -c:a pcm_s16le song.wav` will prepare one.
Ordinary players cannot open `.ms3`.

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
4. Bounded reverse water-filling across six frequency buckets: each bucket's
   real coding load is measured at a data-derived reference step, and the
   resulting multipliers (0.8–1.25x, renormalised) are transmitted rather than
   recomputed, so a poor allocation costs quality but never a broken stream.
5. Dead-zone scalar quantization, then a context-modelled adaptive binary range
   coder — zero-flag hierarchy, significance and greater-than-one bits, and an
   Exp-Golomb remainder, all causally contexted.
6. **Parametric noise substitution.** The decoder compares the energy it
   reconstructed per band against the transmitted band energy and fills the
   deficit into bins that quantized to zero with deterministic, energy-matched
   noise. An unaffordable band is reconstructed with the correct spectral
   envelope instead of as silence, and a band that survived quantization
   receives almost nothing, so the fill is self-limiting. Below 250 Hz it is
   disabled — bass is tonal, and noise there damages structure.
7. Per-file window length selection, and groups of 64 frames coded
   independently with a bisection search on a global step multiplier to meet
   the byte budget.

Optimized from an earlier version with a custom Autoresearch meta process.

## Speed

Encode 0.38–1.09 s and decode 0.16–0.41 s per excerpt (7–16 s of audio, one
core, Apple Silicon). Both are slower than Opus, and the context-modelled range
coder is most of the cost: an earlier version using general-purpose byte
compressors decoded in 0.02–0.06 s but scored 0.067 MOS lower. This codec trades
speed for quality. These are application timings including I/O, not isolated
kernel benchmarks.

## Format

`MSC3`. Little-endian throughout. Header: magic, sample rate, channels,
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
