# Reconstruct the VCO source (amplitude + phase per harmonic) seen at the filter input, per note.
# Amplitudes: S_n(k) from the joint linear fit. Phases: measured harmonic phases of the r=15 files,
# minus the fitted filter phase, minus the best time shift (free-running VCO), circular mean over files.
import numpy as np
from load import NOTES, STEPS, SR
F = np.load('features.npy', allow_pickle=True).item()
L = np.load('linfit_4.npz'); S = L['S']; q = L['q']; settings = [tuple(s) for s in L['settings']]

def Hlin(f, fc, K, g):
    return np.exp(g) / ((1 + 1j * f / fc) ** 4 + K)

def source(n):
    ni = NOTES.index(n); f0 = F[(n, 15, 79)]['f0']
    kmax = min(int(20000 // f0), len(F[(n, 15, 79)]["H"]), S.shape[1]); k = np.arange(1, kmax + 1)
    amp = np.exp(S[ni, :kmax])
    acc = np.zeros(kmax, complex)
    for c in [63, 79, 95, 111]:
        d = F[(n, 15, c)]; i = settings.index((15, c))
        H = d['H'][:kmax] / Hlin(k * d['f0'], *np.exp(q[i, :1]), q[i, 1], q[i, 2])
        ph = np.angle(H)
        # remove time shift: align fundamental to phase 0 -> subtract k * ph1
        ph = np.angle(np.exp(1j * (ph - k * ph[0])))
        w = np.abs(d['H'][:kmax]) > 1e-4
        acc += w * np.exp(1j * ph)
    phase = np.angle(acc); phase[np.abs(acc) == 0] = 0
    return f0, k, amp, phase

def synth(n, dur, fs, src=None):
    f0, k, amp, phase = src or source(n)
    t = np.arange(int(dur * fs)) / fs
    x = np.zeros_like(t)
    for kk, a, p in zip(k, amp, phase):
        x += a * np.cos(2 * np.pi * kk * f0 * t + p)
    return x

if __name__ == '__main__':
    for n in NOTES:
        f0, k, amp, ph = source(n)
        # compare with ideal saw: amplitude 1/k, phase pattern
        print(n, 'f0 %.3f' % f0, 'amp dB rel 1/k (k=1..8):', np.round(20 * np.log10(amp[:8] * k[:8] / amp[0]), 1),
              ' phase-(saw) k=2..6:', np.round(np.angle(np.exp(1j * (ph[1:6] - ph[0] * k[1:6]))), 2))
