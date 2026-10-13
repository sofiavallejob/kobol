# Kobol VCF static dataset: loading + spectral analysis helpers
import numpy as np, scipy.io.wavfile as wavfile, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Kobol_VCO_static')
NOTES = ['B2', 'B3', 'B4', 'B5']
STEPS = [15, 31, 47, 63, 79, 95, 111, 127]
SR = 48000
SEG = (1.0, 2.5)  # steady-state window (gate is on ~0.51 s .. ~2.8 s)

def load(note, r, c, seg=SEG):
    sr, x = wavfile.read(f'{ROOT}/{note}/{note}_r{r:03d}_c{c:03d}.wav')
    x = x.astype(np.float64) / 32768.0
    return x[int(seg[0] * sr):int(seg[1] * sr)] if seg else x

def spectrum(x, nfft=None):
    n = len(x); win = np.blackman(n)
    X = np.fft.rfft(x * win, nfft or 4 * n)
    f = np.fft.rfftfreq(nfft or 4 * n, 1 / SR)
    return f, np.abs(X) / win.sum() * 2

def peaks(x, k=6, fmin=20):
    f, A = spectrum(x)
    from scipy.signal import find_peaks
    i, _ = find_peaks(A, distance=40)
    i = i[f[i] > fmin]; i = i[np.argsort(A[i])[::-1][:k]]
    return [(round(f[j], 1), round(20 * np.log10(A[j] + 1e-12), 1)) for j in i]
