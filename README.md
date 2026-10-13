# Back to the Kobol

vco-greybox/    Learned grey-box VCO (the machine-learning part)
  f0refine.py, extract.py      re-estimated f0, 256 harmonics per recording
  pitchphys2.py, pitch_cv.py   physical pitch law + CV offsets, cross-validation, thesis PU baseline
  exp.py                       grey-box waveshaper (GB) and black-box baseline (BB), all variants, CV harness
  analyze.py, evalraw.py, seeds.py   metrics, bootstrap CIs, raw-WAV evaluation, seed variability
  trainfinal.py                trains the final model on all data and exports the web bundle
  final_full.pt                trained final grey-box model (PyTorch state dict)
  pitch_phys_full.npy          fitted pitch-law parameters (5 + 23 CV offsets)
  engine4.js                   the same model in JavaScript (verified equal to PyTorch)
  kobol_vco_web_bundle*.json   weights + measured harmonics for the browser (with / without audio clips)
  kobol_vco_pipeline.ipynb     notebook running the whole pipeline
  RESULTS.md, *.csv, figs/     all numbers and figures
  (older v1/v2 scripts are kept for history: train3.py, stoch.py, jit.py, noise.py, clips.py)

wdf-modules/    White-box models: LFO (CA3080), voltage-processor lag, CA3080 VCAs, MM5837 noise
  wdf.py, wdf.js, validate.py, validate2.py, sp/ (ngspice netlists), validation json, figs

prototype/      voice-prototype.html (self-contained, open in a browser) + its JS sources
                vcf.js (generated): VCF + SSM2050 ADS1/ADS2; backup-before-vcf/ = page before the VCF was added

vcf-model/      VCF (SSM2040) neural grey-box, learned from Kobol_VCO_static. See vcf-model/README.md for why the
                thesis models failed and the results (held-out waveform ESR about -26 dB vs about -0.2 dB in the thesis)
