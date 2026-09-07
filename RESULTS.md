# Results

Mosaic scores higher than Opus and MP3 at matched file size on ViSQOL v3.
**This is an objective metric, not a listening test.** The caveats below are
not boilerplate; at least one of them identifies a case where the metric
rewards the codec for something inaudible.

Run date: 2026-09-07 (Mosaic v3, format MSC3). macOS on Apple Silicon, Python 3.11, NumPy 2.4.6,
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
| castanets | 97,056 | 106.7 | 4.6609 | 4.4466 | +0.2143 |
| contrabassoon | 234,382 | 116.4 | 4.3365 | 3.8387 | +0.4978 |
| glock | 243,102 | 160.3 | 4.7244 | 4.5062 | +0.2182 |
| guitar | 175,909 | 113.0 | 4.6638 | 4.6236 | +0.0402 |
| harpsichord | 241,338 | 132.4 | 4.6888 | 4.4043 | +0.2845 |
| moonlight | 180,661 | 103.2 | 4.6935 | 4.7088 | −0.0153 |
| ravel | 202,892 | 111.8 | 4.6314 | 4.2592 | +0.3722 |
| sopr | 154,655 | 112.0 | 4.5268 | 3.6720 | +0.8548 |
| steely | 171,565 | 101.1 | 4.6121 | 4.0611 | +0.5510 |
| strauss | 187,109 | 102.3 | 4.6859 | 4.6283 | +0.0576 |
| **mean** | | | **4.6224** | **4.3149** | **+0.3075** |

Ahead on 9/10. Sign test p = **0.011**. Paired bootstrap over excerpts: 95% CI
**[+0.161, +0.474]**, excludes zero.

**None of these ten excerpts was used while tuning the codec.** The tuning was
done on twenty unrelated excerpts from a different source (a CC-BY orchestral
film score); all ten SQAM excerpts were held out and scored once, at the end.

## Against MP3, 128 kb/s

| | Mosaic | MP3 | delta | ahead | 95% CI |
|---|---:|---:|---:|---:|---|
| mean over the same ten excerpts | **4.6388** | 4.3343 | **+0.3045** | 9/10 | [+0.165, +0.447] |

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

🔴 **This codec descends from a predecessor that WAS tuned on five of these
ten excerpts** (castanets, glock, harpsichord, ravel, steely). The search that
produced the current version never saw any SQAM excerpt, but it started from
that predecessor, so the lineage is not perfectly clean. On the five excerpts
never used at any point — contrabassoon, guitar, moonlight, sopr, strauss —
the advantage over Opus is **+0.287**, 95% CI **[+0.021, +0.620]**, ahead on
4/5. That interval excludes zero, though with n=5 the sign test alone
(p = 0.19) would not.

For reference, the predecessor scored **4.5556** on these ten (+0.2407 over
Opus, 8/10). The current version improves on it on **10/10** excerpts by
+0.0667 mean, sign test p = 0.001, CI [+0.035, +0.109].

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

🔴 **The metric is near saturation.** 4.6224 against a 4.7321 ceiling leaves
0.11 MOS of headroom, and Opus at 160 kb/s already scores 4.7003. Differences
this close to the top of a scale are harder to interpret, and further gains on
this metric should be treated with more suspicion, not less.

🔴 **Decoding is ~7x slower than the predecessor** (0.16–0.41 s against
0.02–0.06 s per excerpt), almost entirely from the range coder. The quality
gain is real and so is the cost.

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
