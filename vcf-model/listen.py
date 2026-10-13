# A/B audio: recording vs model at a setting held out of the control law (used by the notebook)
import numpy as np
from law import targets, Hybrid
from fit_osc import SRC, FS
from source import synth
from dsp import vcf
from load import load, SR

def ab(note, r, c, dur=1.5):
    """returns (recording, model, sample rate), both 1.5 s of the steady part, peak-normalised together"""
    O = np.load('oscfit.npy', allow_pickle=True).item(); T = targets()
    te = (T[:, 0] == r) & (T[:, 1] == c)
    p = Hybrid(T[~te, 0], T[~te, 1], T[~te, 2:])(np.array([r]), np.array([c]))[0]
    x = synth(note, dur + 0.3, FS, SRC[note])
    y = vcf(x, FS, 1000 * 2 ** p[0], p[1], O['drive'], 10 ** (p[2] / 20), q=O['q'])[int(0.3 * FS)::2]   # 96k -> 48k
    rec = load(note, r, c)[:len(y)]
    g = 0.9 / max(np.abs(rec).max(), np.abs(y).max())
    return rec * g, y * g, SR
