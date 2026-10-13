# Back to the Kobol

A collection of models, experiments, and browser-based prototypes for reconstructing and analyzing the Kobol synthesizer. The project includes machine-learning-based grey-box models, white-box circuit models, validation tools, and an interactive voice prototype.

## Project Structure

### `vco-greybox/` — VCO Grey-Box Modeling

Machine-learning-based VCO modeling, pitch refinement, evaluation, and browser deployment.

| File / Directory | Description |
|---|---|
| `f0refine.py`, `extract.py` | Fundamental-frequency re-estimation and extraction of 256 harmonics per recording. |
| `pitchphys2.py`, `pitch_cv.py` | Physical pitch law, cross-validation offsets, and thesis baseline. |
| `exp.py` | Grey-box waveshaper (GB), black-box baseline (BB), model variants, and cross-validation harness. |
| `analyze.py`, `evalraw.py`, `seeds.py` | Evaluation metrics, bootstrap confidence intervals, raw-WAV evaluation, and seed variability. |
| `trainfinal.py` | Trains the final model on the complete dataset and exports the browser bundle. |
| `final_full.pt` | Trained grey-box model weights (PyTorch state dictionary). |
| `pitch_phys_full.npy` | Fitted pitch-law parameters, including 23 cross-validation offsets. |
| `engine4.js` | JavaScript implementation of the model, verified against PyTorch. |
| `kobol_vco_web_bundle*.json` | Browser-ready model weights and measured harmonics, with optional audio clips. |
| `kobol_vco_pipeline.ipynb` | Notebook covering the complete modeling pipeline. |
| `RESULTS.md`, `*.csv`, `figs/` | Experimental results, evaluation data, and figures. |

Historical scripts from earlier versions are retained for reference: `train3.py`, `stoch.py`, `jit.py`, `noise.py`, and `clips.py`.

### `wdf-modules/` — White-Box Circuit Models

White-box models of individual synthesizer components:

- LFO based on the CA3080.
- Voltage-processor lag.
- CA3080-based VCAs.
- MM5837 noise generator.

Includes Python and JavaScript implementations (`wdf.py`, `wdf.js`), validation scripts, `sp/` with ngspice netlists, validation results in JSON, and figures.

### `prototype/` — Interactive Voice Prototype

A self-contained browser-based synthesizer prototype.

- `voice-prototype.html` — Main prototype; open directly in a browser.
- JavaScript source files for the prototype.
- `vcf.js` — Generated VCF implementation, including the SSM2050 ADS1/ADS2 envelope generators.
- `backup-before-vcf/` — Snapshot of the prototype before the VCF was added.

### `vcf-model/` — VCF Grey-Box Modeling

Neural grey-box modeling of the SSM2040 voltage-controlled filter, trained using `Kobol_VCO_static`.

See [`vcf-model/README.md`](vcf-model/README.md) for the modeling methodology, an analysis of why the thesis models failed, and the evaluation results. The final model achieves a held-out waveform error-to-signal ratio (ESR) of approximately **−26 dB**, compared with approximately **−0.2 dB** for the thesis models.

## Reproducibility and Evaluation

The project includes training and evaluation scripts, cross-validation experiments, raw audio comparisons, circuit simulations, and browser-ready model exports. Refer to the README files within individual directories for component-specific details and results.
