# Stage 1: linearised SSM2040 fit on the non-oscillating files (r <= 95).
# |A(n,r,c,k)| = S_n(k) * g(r,c) * |1 / ((1 + j f/fc)^N + K)|   at f = k * f0_n
# S_n(k): source (saw) spectrum per note, free per harmonic, shared by the 64 settings of that note.
import numpy as np
from scipy.optimize import least_squares
from load import NOTES, STEPS
F = np.load('features.npy', allow_pickle=True).item()
THR = 10 ** (-80 / 20)
RLIN = [r for r in STEPS if r <= 95]
N_POLES = int(__import__('sys').argv[1]) if len(__import__('sys').argv) > 1 else 4

rows = []  # (note idx, setting idx, harmonic k, f, log|A|)
settings = [(r, c) for r in RLIN for c in STEPS]
for ni, n in enumerate(NOTES):
    for si, (r, c) in enumerate(settings):
        d = F[(n, r, c)]
        for k, h in zip(d['k'], d['H']):
            if abs(h) > THR and k * d['f0'] < 20000:
                rows.append((ni, si, k, k * d['f0'], np.log(abs(h))))
rows = np.array(rows); ni, si, kk, ff, la = rows.T; ni = ni.astype(int); si = si.astype(int); kk = kk.astype(int)
KM = int(kk.max()); NS = len(settings)
sidx = ni * KM + (kk - 1)

def unpack(p):
    S = p[:4 * KM]; q = p[4 * KM:].reshape(NS, 3)   # log fc, K, log g
    return S, q

def model(p):
    S, q = unpack(p)
    fc = np.exp(q[si, 0]); K = q[si, 1]; g = q[si, 2]
    s = 1j * ff / fc
    return S[sidx] + g - np.log(np.abs((1 + s) ** N_POLES + K))

def resid(p):
    S, q = unpack(p)
    # weak priors: g of the r=15 settings ~ 0 (fixes the S/g gauge), smooth S in k
    reg = [0.3 * q[[i for i, (r, c) in enumerate(settings) if r == 15], 2]]
    return np.concatenate([model(p) - la] + reg)

# init: S from saw 1/k, fc from step law, K small
p0 = np.zeros(4 * KM + 3 * NS)
p0[:4 * KM] = np.tile(np.log(0.3 / np.arange(1, KM + 1)), 4)
for i, (r, c) in enumerate(settings):
    p0[4 * KM + 3 * i:4 * KM + 3 * i + 3] = [np.log(68 * 2.17 ** ((c - 15) / 16)), 0.5 * r / 127 * 4, 0]
lb = np.full_like(p0, -np.inf); ub = np.full_like(p0, np.inf)
for i in range(NS):
    lb[4 * KM + 3 * i] = np.log(5); ub[4 * KM + 3 * i] = np.log(40000)
    lb[4 * KM + 3 * i + 1] = -0.9; ub[4 * KM + 3 * i + 1] = 3.999
sol = least_squares(resid, p0, bounds=(lb, ub), x_scale='jac', max_nfev=300)
e = (model(sol.x) - la) * 20 / np.log(10)
print(f'poles={N_POLES}  points={len(la)}  rms err {np.sqrt(np.mean(e**2)):.2f} dB  median |err| {np.median(abs(e)):.2f} dB')
S, q = unpack(sol.x)
print(' r   c    fc(Hz)     K     g(dB)   rms err dB')
for i, (r, c) in enumerate(settings):
    m = si == i
    print(f'{r:3d} {c:3d} {np.exp(q[i,0]):9.1f} {q[i,1]:6.3f} {q[i,2]*8.686:7.2f}   {np.sqrt(np.mean(e[m]**2)):.2f}')
np.savez(f'linfit_{N_POLES}.npz', S=S.reshape(4, KM), q=q, settings=np.array(settings))
