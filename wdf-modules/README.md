# WDF models of the Kobol control board (LFO, noise, voltage processor) and the CA3080 VCAs

Topology from the original RSF schematics; component values in `wdf.py` (P) are ASSUMED, era-typical placeholders
until the parts list for this board is available. Replace them in one place (P in wdf.py / WDF.P in wdf.js).

- Noise: Gaussian junction noise -> WDF colouring network (series coupling C, shunt C pre-warped at its corner, load R).
- Voltage processor: WDF RC lag with log-taper pot; gain stages as ideal op-amps.
- LFO: virtual-ground integrator (ideal current source into a WDF capacitor) + Schmitt comparator with symmetry.
- VCA: CA3080 as tanh differential pair (input attenuator 100k/220R) driving a WDF R||C output network.
- ADS envelope: behavioural (SSM2050 not documented at component level).

Validation against ngspice 42 (validate.py, sp/*.cir, wdf_validation.json, wdf_validation.pdf), fs = 48 kHz:
- noise network: max deviation 0.7 dB between 20 Hz and 16 kHz (bilinear warping near Nyquist, after pre-warping)
- lag: max step-response error below 0.7 mV on a 5 V step, three pot settings
- LFO: frequency within 0.4 % of ngspice (behavioural Schmitt with finite gain) and within 0.4 % of the ideal-circuit formula
- VCA: DC transfer error below 14 uV; THD 0.37 % at 5 V peak input (tanh input stage)

## Update: MM5837 noise source and CA3080 LFO (hypotheses from secondary sources)
- MM5837: bit-exact 17-stage LFSR (taps 14, 17), clock assumed 100 kHz, integrated per sample. Period 131071 clocks (maximal).
  White flat within 0.9 dB (100 Hz-10 kHz); optional pink filter (-3.04 dB/oct measured).
- OTALFO: CA3080 integrator (square -> 100k/220R -> tanh pair -> WDF capacitor) + Schmitt; Iabc exponential, 0.01-100 Hz,
  rate CV 1 V/oct, symmetry = OTA input offset. Sub-sample threshold correction. Within 0.4 % of the ideal circuit and
  0.7 % of ngspice from 0.04 to 100 Hz (validate2.py, validation2.json).
- paperfigs.py: system diagram, grey-box VCO architecture, 6-panel white-box validation figure (figs/).
- NOTE: published R31-R41 values belong to the main expander board's parts list (designators restart per board); not used.
