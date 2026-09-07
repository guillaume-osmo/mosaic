# Results

Mosaic scores higher than Opus and MP3 at matched file size on ViSQOL v3.
**This is an objective metric, not a listening test.** The caveats below are
not boilerplate; at least one of them identifies a case where the metric
rewards the codec for something inaudible.

Run date: 2026-09-07. macOS on Apple Silicon, Python 3.11, NumPy 2.4.6,
SciPy 1.17.1, FFmpeg 8.0 (`libopus`, `libmp3lame`).

## Method

Codecs are compared at **equal bytes per excerpt**, never at equal nominal
bitrate. Opus is VBR and MP3 carries framing, so a requested rate is not a
delivered rate: `-b:a 96k` produced **116.1 kb/s** on this corpus. The reference
codec is encoded first, its actual output size is measured, and Mosaic is held
to exactly that budget. Opus runs at **complexity 10**, its best setting, so the
comparison is not against a handicap.

Corpus: ten EBU SQAM excerpts (48 kHz stereo, 7.3–16.1 s) — castanets,
contrabassoon, glockenspiel, guitar, harpsichord, moonlight, ravel, soprano,
steely, strauss. This is the material codec listening tests use, chosen to be
hard: sharp transients, solo voice, dense orchestral texture.

## Against Opus, 96 kb/s, complexity 10

| excerpt | bytes | kb/s | Mosaic | Opus | delta |
|---|---:|---:|---:|---:|---:|
| castanets | 97,112 | 106.7 | 4.6040 | 4.4466 | +0.1574 |
| contrabassoon | 234,642 | 116.5 | 4.2078 | 3.8387 | +0.3691 |
| glock | 244,911 | 161.5 | 4.7183 | 4.5062 | +0.2121 |
| guitar | 175,914 | 113.0 | 4.6137 | 4.6236 | −0.0099 |
| harpsichord | 241,455 | 132.5 | 4.6715 | 4.4043 | +0.2672 |
| moonlight | 180,738 | 103.3 | 4.6636 | 4.7088 | −0.0453 |
| ravel | 202,947 | 111.8 | 4.5856 | 4.2592 | +0.3264 |
| sopr | 154,668 | 112.1 | 4.3030 | 3.6720 | +0.6310 |
| steely | 171,657 | 101.2 | 4.5395 | 4.0611 | +0.4784 |
| strauss | 187,163 | 102.4 | 4.6494 | 4.6283 | +0.0210 |
| **mean** | | | **4.5556** | **4.3149** | **+0.2407** |

Ahead on 8/10. Paired bootstrap over excerpts: 95% CI **[+0.118, +0.372]**,
excludes zero. Sign test p = 0.055.

## Against MP3, 128 kb/s

| | Mosaic | MP3 | delta | ahead | 95% CI |
|---|---:|---:|---:|---:|---|
| mean over the same ten excerpts | **4.5827** | 4.3343 | **+0.2484** | 8/10 | [+0.099, +0.400] |

Opus is the stronger reference: on this corpus it reaches MP3's quality with
11–21% fewer bytes. A result that beat only MP3 would be the weaker claim.

## The metric

`visqol.py` is a NumPy reimplementation of ViSQOL v3 audio mode, using the
nu-SVR that ships with the reference implementation. It is validated against
`conformance.h` from [google/visqol](https://github.com/google/visqol) — the
published output of Google's own binary on its shipped test pairs — over nine
degraded pairs spanning MOS 1.39 to 4.73:

| | |
|---|---|
| Spearman rho | **0.9833** |
| mean absolute error | **0.0310 MOS** |
| bias | +0.022 |
| the six real-codec conditions (AAC 24/48/64/128/256, MP3 96, Opus 128) | within **0.01 MOS** |

An identity pair reproduces `kConformanceCastanetsIdentity` (4.732101253) to
twelve digits, which is also the ceiling of the scale.

Alignment is load-bearing rather than incidental. Opus and MP3 reconstruct
late, while Mosaic trims to the exact original length and comes out
sample-aligned; scoring without compensating charges the reference codecs for a
delay nobody can hear. `visqol.globally_align` removes one integer lag matched
on the signal envelope, and a pure delay of any size scores exactly the identity
value.

## Caveats

🔴 **No listening test has been run.** A codec claim of this kind is normally
settled by matched-size blind listening. Nobody has heard these files. The
`--keep` flag exists so you can.

🔴 **Five of the ten excerpts were used while tuning the codec**
(castanets, glock, harpsichord, ravel, steely). The intervals above are
therefore optimistic. On the five never used — contrabassoon, guitar,
moonlight, sopr, strauss — the mean advantage over Opus is **+0.193** with a 95%
CI of **[−0.018, +0.450]**, ahead on 3/5. **That interval includes zero.** The
honest summary of the Opus comparison is: level with Opus and probably slightly
ahead, not decisively better.

🔴 **The metric can reward this codec for inaudible work.** On excerpts with
essentially no energy above 8 kHz, the reference codecs correctly spend nothing
there, their per-band NSIM in the top bands collapses, and Mosaic's noise
substitution "wins" by refilling bands that carry no audible signal — in one
measured case Opus scored a *negative* NSIM at 15.7 kHz, which is a comparison
of noise-floor textures, not of quality. Across a 30-excerpt corpus this was
**not** the main driver of the advantage (Spearman between HF energy fraction
and Mosaic's gain: −0.16), but the advantage does shrink on genuinely wideband
material (+0.175 on the five excerpts with >10% of energy above 8 kHz). Some
part of the measured margin is metric-directed rather than audible.

🔴 **The metric is near saturation.** 4.5556 against a 4.7321 ceiling leaves
0.18 MOS of headroom, and Opus at 160 kb/s already scores 4.7003. Differences
this close to the top of a scale are harder to interpret.

🔴 **One operating point, one small corpus.** Everything here is at
Opus-96-matched bytes. The rate–quality curve away from that point is
unmeasured, and ten excerpts is a small sample.

## Reproducing

```bash
pip install -r requirements.txt
python benchmark.py path/to/*.wav --bitrate 96 --codec opus --complexity 10 --keep out/
```

The SQAM excerpts used here ship inside the ViSQOL repository under
`testdata/conformance_testdata_subset/`. Any 48 kHz 16-bit stereo WAV works.
