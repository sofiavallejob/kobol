# Kobol Expander VCO: results (DAFx draft numbers)

All numbers are on held-out settings. Bootstrap 95% CI on the median in brackets.
CV schemes: random 5-fold (rand5); whole interior waveform columns held out (col, 4 interleaved folds);
whole interior CV rows held out (row, 4 interleaved folds). Grey-box variants: 1500 steps per fold.

## Pitch (RMS cents, interior range; pitch_cv.csv, pitch_nooff.py)
| method | rand5 | col | row |
|---|---|---|---|
| physical + residual | 0.84 | 1.12 | 4.80 |
| physical (exp. law + waveform detune + per-step CV offsets) | 1.37 | 1.37 | 4.95 |
| physical, no CV offsets (interior fit) | 4.02 | 3.95 | 4.58 |
| physical, no waveform detune | 3.79 | 3.24 | 5.94 |
| neural pitch unit (v2) | 2.83 | 2.58 | 8.78 |
| thesis periodic unit (trained on all data, original labels) | 517 | 517 | 517 |
- 1.501 oct/V. Reset-time term: no effect (removed).
- Per-step offsets are shared by all 19 waveforms -> CV source (MCV24). They cannot be predicted for unseen CV levels (row scheme).
- Thesis PU: normalisation verified (matches mid range); fails because original labels are wrong at both ends and omega is regressed linearly.

## Waveform (median harmonic SNR dB; summary_cv.csv)
| method | rand5 | col | row |
|---|---|---|---|
| grey-box final | 17.4 [16.8, 17.7] | 13.9 [13.0, 14.6] | 17.6 [16.9, 18.4] |
| grey-box final + residual | 18.6 [17.7, 19.4] | 13.3 [12.3, 14.6] | 21.5 [20.0, 22.4] |
| grey-box (v3 structure) | 16.4 [16.0, 17.4] | 12.3 [11.2, 13.0] | 17.4 [16.6, 17.9] |
| grey-box (v3) + residual | 18.7 [17.7, 19.6] | 11.8 | 21.4 [19.7, 22.4] |
| black-box (v2) | 12.7 [12.1, 13.2] | 11.1 [10.3, 11.7] | 13.2 [12.4, 14.0] |
| black-box (v2) + residual | 14.6 | 11.7 | 16.1 |
| linear interpolation of neighbours | 13.6 [12.9, 14.3] | 12.0 [11.3, 12.6] | 17.1 [15.8, 18.9] |
Final = v3 structure + L1 on stage gains + second lowpass pole + comparator edge time.
Learned globals (full data): chain highpass 11.3 Hz, second pole 10.8 kHz, comparator edges 43 / 39 us.

## Seed variability (rand5, final variant, 3 seeds)
SNR 17.37 / 17.18 / 17.38 -> 17.31 +- 0.11 dB; with residual 18.42 +- 0.14 dB.

## Ablations (rand5, SNR / LSD dB)
full 16.4 / 3.9; -lowpass 15.4 / 6.0; -chain HP 15.3 / 4.7; -comp B 16.0 / 4.7; -triangle 16.3 / 5.6;
+second pole 16.8 / 3.4; +edge slew 16.5 / 4.0; +core reset 16.3 / 3.9; +L1 16.5 / 4.3; +all three 17.1 (median of folds); final 17.4 / 3.6.

## By family and register (rand5, SNR dB)
grey-box final: pulse 12.2, saw 16.2, square 22.8, triangle 18.5 | low 17.0, mid 19.0, high 14.3
black-box: pulse 9.7, saw 11.7, square 15.0, triangle 14.5 | low 10.0, mid 14.9, high 13.6

## Waveform columns held out (col_breakdown.csv)
Grey-box wins by 5 to 9 dB in continuous morph regions (triangle to saw, columns 1-5; square to pulse, 12-15)
and loses where the waveshaper switches structure (columns 7, 10, 17), where neighbours are nearly identical.

## Stochastic part
Jitter 0.4-0.9 cents rms (audio range), amplitude wobble < 0.2 %, noise floor about -68 dBFS (53 dB below signal).

## Figures (figs/)
fig_pitch: deviation from 1.5 oct/V and per-step CV offsets. fig_waveshaper_params: final model, effective rms contribution of each stage, active comparator duty, output lowpass vs waveform CV.
fig_cv: CV boxplots. fig_col: SNR per held-out waveform column. fig_cycles: held-out cycles, recording vs grey-box vs black-box.

## Against the raw WAVs (rand5, 0.5 s segments, multi-resolution STFT spectral convergence dB, lower is better; raw_cv_rand5_final.csv)
| method | all | low | mid | high |
|---|---|---|---|---|
| grey-box final + residual (physical pitch + residual) | -13.4 | -11.4 | -20.8 | -7.5 |
| grey-box final (physical pitch) | -12.3 | -11.0 | -18.3 | -7.8 |
| black-box (v2) + residual | -11.8 | -9.1 | -17.5 | -6.8 |
| black-box (v2) | -10.1 | -8.6 | -15.3 | -5.9 |
| linear interpolation of neighbours | -9.8 | -7.9 | -15.7 | -6.5 |
| ceiling: same recording, other segment | -20.9 | -16.1 | -29.6 | -16.5 |

## Reproducibility
kobol_vco_pipeline.ipynb runs the whole pipeline. Verified in a clean directory (all 13 stages pass, reduced steps);
re-estimated f0 identical to the reported one (max difference 4e-13 cents). Full run about 2 h on one CPU core.

## Limitations to state
No information above 18 kHz. No dynamics (all recordings at fixed CVs). Per-step CV offsets cannot be predicted for unmeasured CV levels.
Low-register jitter estimates unreliable (about 20 cycles per file). Structural switches of the waveshaper (columns 7, 10, 17) are
better served by plain interpolation. Expander schematics are third-party redraws; the waveshaper itself is not documented, so its
structure is inferred.
