# Per-file analysis: f0 of the VCO, harmonic amplitudes (least squares on the steady segment),
# self-oscillation peak (strongest non-harmonic component) and noise floor -> features.npz
import numpy as np
from load import *

KMAX = 200

def f0_refine(x, guess):
    f, A = spectrum(x, 1 << 20)
    m = (f > guess * 0.97) & (f < guess * 1.03)
    i = np.argmax(A * m)
    a, b, c = np.log(A[i - 1:i + 2] + 1e-20)
    return f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])

def harmonics(x, f0, kmax):
    t = np.arange(len(x)) / SR
    k = np.arange(1, kmax + 1); k = k[k * f0 < 0.45 * SR]
    w = np.hanning(len(x))
    E = np.exp(-2j * np.pi * np.outer(t, k * f0))
    H = 2 * (w @ (x[:, None] * E)) / w.sum()     # windowed DFT at exact harmonic frequencies
    return k, H

if __name__ == '__main__':
    GUESS = {'B2': 61.7, 'B3': 246.8, 'B4': 494.0, 'B5': 988.2}
    out = {}
    for n in NOTES:
        f0n = f0_refine(load(n, 15, 79), GUESS[n])
        for r in STEPS:
            for c in STEPS:
                x = load(n, r, c)
                f0 = f0_refine(x, f0n)
                if abs(1200 * np.log2(f0 / f0n)) > 20: f0 = f0n   # input swamped by self-oscillation
                k, H = harmonics(x, f0, KMAX)
                # residual after removing harmonics -> self-oscillation peak + floor
                t = np.arange(len(x)) / SR
                res = x - np.real(np.exp(2j * np.pi * np.outer(t, k * f0)) @ H)
                f, A = spectrum(res)
                j = np.argmax(A * (f > 20))
                out[(n, r, c)] = dict(f0=f0, k=k, H=H, osc_f=f[j], osc_a=A[j],
                                      floor=np.median(A[f > 20]), rms=np.sqrt(np.mean(x ** 2)),
                                      res_rms=np.sqrt(np.mean(res ** 2)))
    np.save('features.npy', out, allow_pickle=True)
    print('f0 per note', {n: round(out[(n, 15, 79)]['f0'], 3) for n in NOTES})
    print('note  r  | osc_f (Hz) by cutoff step / osc amp dB (harmonic rms dB)')
    for n in NOTES:
        for r in STEPS:
            print(n, r, ' '.join('%7.0f/%4.0f' % (out[(n, r, c)]['osc_f'], 20 * np.log10(out[(n, r, c)]['res_rms'] + 1e-9)) for c in STEPS))
