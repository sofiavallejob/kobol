import numpy as np

from load import spectrum
def feats(x, fs, f0, kmax=200):
    """harmonic amplitudes at k*f0 + strongest non-harmonic peak (freq, amp) of a steady segment"""
    k, H = harmonics_fs(x, fs, f0, kmax)
    t = np.arange(len(x)) / fs
    res = x - np.real(np.exp(2j * np.pi * np.outer(t, k * f0)) @ H)
    n = len(res); w = np.blackman(n); R = np.fft.rfft(res * w, 4 * n); f = np.fft.rfftfreq(4 * n, 1 / fs)
    A = np.abs(R) / w.sum() * 2; m = (f > 20) & (f < 20000)
    j = np.argmax(A * m)
    if 0 < j < len(A) - 1:   # parabolic refine
        a, b, c = np.log(A[j - 1:j + 2] + 1e-20); fj = f[j] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])
    else: fj = f[j]
    return k, H, fj, A[j], np.sqrt(np.mean(res ** 2))
def harmonics_fs(x, fs, f0, kmax):
    t = np.arange(len(x)) / fs
    k = np.arange(1, kmax + 1); k = k[k * f0 < min(20000, 0.45 * fs)]
    w = np.hanning(len(x)); E = np.exp(-2j * np.pi * np.outer(t, k * f0))
    return k, 2 * (w @ (x[:, None] * E)) / w.sum()
