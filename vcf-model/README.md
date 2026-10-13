# Kobol VCF model (SSM2040), learned from Kobol_VCO_static

Despite its name, `Kobol_VCO_static` is the **VCF** dataset from Álvaro Crespo's thesis
("Modelling the VCF from the RSF Kobol", 2024): 4 VCO notes (B2–B5, saw) × 8 resonance × 8 cutoff
values sent from a Doepfer MCV24 (MIDI 15…127, about 1.18…10 V), 3 s each at 48 kHz, gated note at 0.51–2.8 s.

## What the thesis tried, and why it failed
| attempt | what happened | root cause |
|---|---|---|
| Mod-extraction LSTM (Mitcheltree et al.) on the dynamic subset | never trained: 23 min/epoch on a laptop CPU, overheating | model size versus hardware |
| PedalNet (WaveNet) + cutoff/resonance conditioning (`model4.py`) | validation loss stuck at 0.94–0.98 (ESR-type: about the same as outputting silence), RMSE ~2100 (unnormalised int16) | **no sample-aligned input/output pairs.** The static subset has only the filter output, with a free-running VCO and ~44 ms of MIDI/USB latency. A sample-level loss cannot be learned when the input phase is unknown, so the net converges to near-zero output. Also: only 4 notes of one waveform, filter states at self-oscillation, and a short receptive field for a 62 Hz input at 44.1 kHz. |

The dataset itself is fine. The problem was asking a black-box network to learn a waveform mapping that the data
does not pin down. A pure black-box model also could not generalise from 4 sawtooth notes to other inputs.

## What this folder does instead: neural grey-box
The schematics and parts lists (main board: CI4 = **SSM2040**, CI5 = CA3080 in the resonance loop driven by T3,
CI8/CI9 = SSM2050 ADS, output VCA = CA3080 with linear Iabc from T2) give the circuit structure. The data gives
everything else:

1. `extract.py`: f0 per file and 200 harmonic phasors on the steady segment (1.0–2.5 s); self-oscillation peak and level from the residual.
2. `fit_linear.py`: joint fit of an unknown per-note source spectrum and a linear N-pole filter `g/((1+s/wc)^N + K)` on the 192 non-oscillating files.
   **4 poles: 0.32 dB RMS, 0.08 dB median** over 5,683 harmonics (2 poles 3.45, 3 poles 1.81, 6 poles 0.62), so the SSM2040 topology is confirmed by the data.
   K grows linearly with the resonance step (0, 0.17, 1.0, 1.9, 2.9, 3.9) and crosses 4 between r95 and r111, exactly where the recordings start to self-oscillate.
3. `source.py`: reconstructs the VCO saw (amplitudes + phases) at the filter input. It is identical across the 4 notes, which also makes sample-aligned dry/wet pairs possible for future black-box work.
4. `../dsp/kobol_vcf.h`: nonlinear SSM2040: four tanh OTA cells plus a **saturating CA3080 resonance feedback**, ZDF/TPT with linearised tanh (2 iterations).
   Cell-only saturation pulled the oscillation ~30 % below fc. The real filter oscillates at fc, which pointed to (and the schematic confirms) the OTA in the feedback loop.
5. `fit_osc.py`: on the 64 oscillating files, global drive = 0.1 and feedback knee q = 18 (grid search), with per-file fc, K and gain fitted by simulating the C++ filter.
   Median oscillation error 0.08 cents and 0.01 dB; input-harmonic suppression within 0.13 dB median.
6. `law.py`: **control law** (cutoff V, resonance V) → (fc, K, gain): physical law (exponential converter with top compression, linear resonance current with a dead zone) **plus a neural residual** (ensemble of 5 MLPs 2-24-24-3, NumPy/Adam).
   Held-out RMS cutoff error (cents), whole rows / whole columns / random points:

   | law | row | col | rand |
   |---|---|---|---|
   | pure MLP | 24 | **320** | 53 |
   | physical | 31 | 48 | 25 |
   | interpolation | 18 | 128 | 129 |
   | **hybrid (shipped)** | 20 | **36** | **14** |

7. `eval_audio.py`: end to end. The law is trained without the held-out setting, rendered through the C++ filter, and compared with the recording:

   | scheme | harmonics (dB, median/p90) | waveform ESR (dB, median/p90) | osc pitch (cents, median/p90) | osc level (dB, median) |
   |---|---|---|---|---|
   | rand holdout, hybrid | 0.32 / 1.15 | **−25.9 / −12.9** | 8 / 17 | 0.7 |
   | col holdout, hybrid | 0.30 / 1.90 | −27.2 / −13.0 | 11 / 76 | 0.6 |
   | row holdout, hybrid | 0.61 / 2.44 | −20.0 / −5.7 | 11 / 39 | 1.9 |
   | rand holdout, interpolation | 0.28 / 5.20 | −30.0 / −7.7 | 9 / **304** | 0.7 |

   For comparison, the thesis PedalNet reached ESR ≈ 0.95 (−0.2 dB). The hardest settings are r95 (K ≈ 3.9, on the edge of self-oscillation), where a tiny K error moves the resonant peak a lot.
8. `export_law.py` writes `../dsp/kobol_vcf_law.h` (C++ matches Python to 1e-4). `export_js.py` writes `../prototype/vcf.js` (JS filter equals C++ to 1e-12).

## Plots, notebook, listening
- `kobol_vcf_pipeline.ipynb`: the whole pipeline top to bottom, with every plot inline and an A/B audio player.
- `figs.py`: all figures, PNG in `figs/`, PDF in `../paper-latex/figs/`. They cover the dataset, the linear fit and model order, the reconstructed source, the parameters with the learned law, the nonlinear model, **training curves** (train vs held-out loss per Adam step), law cross-validation, the end-to-end ESR distribution and heatmaps, and held-out waveforms. Heavy steps are cached in `figs/cache_*.npy`; delete them after re-fitting.
- `listen.py` / `figs/audio/`: recording vs model at settings held out of the control law (4 pairs of WAVs, including a self-oscillating one).

Run order: `extract.py → fit_linear.py 4 → fit_osc.py full 0.1 18 → law.py cv final → eval_audio.py → export_law.py → export_js.py`
(about 15 min on one core, NumPy/SciPy only; `dsp.py` compiles the C++ headers with the system compiler).

## Limits (state these in the paper)
- One unit, 4 notes, one waveform, one input level: the drive (input level re the tanh knee) is only identified around that level.
- No dynamics in this subset: cutoff modulation speed and the ADS2 → VCF path are not validated (the thesis' dynamic subset, not in this folder, could do that).
- Between-take drift of ~1–2 % at the top of the range (each note was a separate take) limits cutoff accuracy to about ±15 cents.
- In self-oscillation, K and gain partly trade off against each other; the law smooths this, but individual K values above 4 are not physically precise.
- CV scale: MIDI × 10/127 V (the thesis' estimate). The measured converter slope is 0.89 oct/V at that scale, so keyboard tracking in the plugin uses the measured slope.
