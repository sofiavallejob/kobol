# Kobol VCO neural model

Pipeline (run in order, dataset unzipped to ./kobol):
1. f0.py / extract.py: re-estimate f0 from audio (ACF + FFT refine), extract 256 phase-aligned harmonics per recording -> harm.npz
2. train3.py: `python train3.py full "[]" 192 0.05` (STEPS env, default 12000). `val` mode holds out 66 recordings.
3. export.py: weights (float16) + measured harmonics -> kobol_vco_models_web.json
4. engine.js: forward pass in JS (pitch(cvf,cvw), shape(cvf,cvw), recording(w,n))

Models (inputs: [(cvf-6.27)/3.63, (cvw-4.5)/4.5]):
- Pitch unit: linear(2->1) + MLP 2-64-64-64-1 (SiLU), output log2(f0 Hz). Held-out: ~1.8 cents RMS.
- Shape unit: MLP 2-192-192-192-512 (SiLU), output Re/Im of harmonics 1..256, aligned so H1 is a sine. Held-out: ~14 dB median SNR (neighbour average baseline 10 dB).
- ref.py: loads kobol_vco_models.pt in PyTorch.

f0b.csv: corrected f0 per recording (w = waveform id 0-18, n = file index 21-43, cvf = 2.64 + 0.33*(n-21)).

## v2: harmonic plus noise with residual correction
- engine2.js: smooth nets + stored residuals at the 19x23 grid, bilinear blend (exact at measured settings, pitch and harmonics).
- resval.py: held-out test. Smooth only 13.6 dB SNR, smooth + residual 15.3 dB, neighbour average 10.1 dB.
- jit.py -> jit.csv: pitch jitter (cents rms) and amplitude wobble per file. Audio range: ~0.4-0.9 cents, <0.2 %. Low-end values unreliable (few cycles).
- stoch.py / noise.py: phase-tracked periodic reconstruction, residual PSD -> noise floor (~ -68 dBFS, ~53 dB below signal) -> noise_ir.npy (FIR).
- clips.py: 1 s MP3 excerpts of every recording for A/B (in the web bundle).

## v3: physical pitch + grey-box waveshaper (current)
- pitchphys2.py: exponential converter (1.501 oct/V) + waveform detune + per-CV-step offsets (shared across waveforms). Held-out 1.43 cents rms (neural pitch unit 1.77). Reset-time term not supported by the data.
- greybox.py: saw core -> triangle fold + 2 comparators -> 1-pole lowpass -> recording-chain highpass (learned 11.3 Hz). MLP 2-64-64-64-8. Closed-form harmonics, alias-free.
- evalfw.py / eval2.py -> eval_all.csv. Held-out medians:
  grey-box + residual: SNR 21.7 dB, LSD 3.7 dB, spectral convergence -11.7 dB
  grey-box:            18.7 / 4.0 / -11.3
  v2 net + residual:   15.3 / 6.3 / -9.8
  neighbour average:   10.1 / 5.3 / -8.0
  ceiling (same file halves) SC -19.3, neighbouring recording -7.5
- Seeds (grey-box held-out SNR): 18.8, 18.4, 17.4.
- kobol_vco_pipeline.ipynb: runs the whole pipeline.
