# Stage 2: nonlinear SSM2040 on the self-oscillating files (r = 111, 127).
# Global: drive (input level re cell tanh knee), q (feedback OTA knee re cells).
# Per file: fc, K, gain (Nelder-Mead on the C++ filter driven by the reconstructed VCO saw).
import numpy as np, sys, itertools
from scipy.optimize import minimize
from source import synth, source, F
from dsp import vcf
from feats import feats
from load import NOTES, STEPS
FS = 96000; DUR = 0.8; SEG = slice(int(0.4 * FS), None)
SRC = {n: source(n) for n in NOTES}; X = {n: synth(n, DUR, FS, SRC[n]) for n in NOTES}

def sim_feats(n, fc, K, drive, gain, q):
    y = vcf(X[n], FS, fc, K, drive, gain, q=q)[SEG]
    return feats(y, FS, SRC[n][0], kmax=6)

def loss_file(n, r, c, p, drive, q, detail=False):
    fc, K, g = np.exp(p[0]), p[1], np.exp(p[2])
    d = F[(n, r, c)]
    k, H, fo, ao, _ = sim_feats(n, fc, K, drive, g, q)
    cents = 1200 * np.log2(max(fo, 1) / d['osc_f'])
    adb = 20 * np.log10(ao + 1e-12) - 20 * np.log10(d['osc_a'])
    hm = np.abs(d['H'][:len(H)]) > 10 ** (-80 / 20)
    hdb = (20 * np.log10(np.abs(H) + 1e-12) - 20 * np.log10(np.abs(d['H'][:len(H)]) + 1e-12))[hm]
    L = (cents / 10) ** 2 + adb ** 2 + (np.mean(hdb ** 2) if hm.any() else 0)
    return (L, cents, adb, hdb) if detail else L

def fit_file(n, r, c, drive, q, p0=None):
    d = F[(n, r, c)]
    p0 = p0 if p0 is not None else [np.log(d['osc_f']), 5.0 if r == 111 else 6.0, 0.0]
    best = minimize(lambda p: loss_file(n, r, c, p, drive, q), p0, method='Nelder-Mead',
                    options=dict(xatol=1e-3, fatol=1e-3, maxfev=150, initial_simplex=np.array(p0) + np.array([[0, 0, 0], [0.03, 0, 0], [0, 1.0, 0], [0, 0, 0.5]])))
    return best.x, best.fun

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'grid'
    if mode == 'grid':
        files = [('B3', r, c) for r in (111, 127) for c in STEPS]
        for drive, q in itertools.product([0.1, 0.3, 1.0], [3, 10, 30]):
            Ls = [fit_file(*f, drive, q)[1] for f in files]
            print(f'drive {drive:4} q {q:3}: median loss {np.median(Ls):7.2f} mean {np.mean(Ls):7.2f}', flush=True)
    if mode == 'refine':
        files = [('B3', r, c) for r in (111, 127) for c in STEPS] + [('B5', 127, c) for c in STEPS[::2]]
        for drive, q in itertools.product([0.05, 0.1, 0.15], [12, 18, 25]):
            Ls = [fit_file(*f, drive, q)[1] for f in files]
            print(f'drive {drive:4} q {q:3}: median loss {np.median(Ls):7.3f} mean {np.mean(Ls):7.3f}', flush=True)
    if mode == 'full':
        drive, q = float(sys.argv[2]), float(sys.argv[3])
        out = {}
        for n in NOTES:
            for r in (111, 127):
                for c in STEPS:
                    p, L = fit_file(n, r, c, drive, q)
                    _, cents, adb, hdb = loss_file(n, r, c, p, drive, q, detail=True)
                    out[(n, r, c)] = dict(fc=np.exp(p[0]), K=p[1], g=np.exp(p[2]), loss=L, cents=cents, adb=adb,
                                          hrms=np.sqrt(np.mean(hdb ** 2)) if len(hdb) else np.nan)
                    o = out[(n, r, c)]
                    print(f'{n} r{r} c{c:3d}: fc {o["fc"]:8.1f} K {o["K"]:6.2f} g {20*np.log10(o["g"]):6.2f} dB | osc err {cents:+6.2f} c {adb:+5.2f} dB | input harm rms err {o["hrms"]:.2f} dB', flush=True)
        np.save('oscfit.npy', dict(drive=drive, q=q, files=out), allow_pickle=True)
