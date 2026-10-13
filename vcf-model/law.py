# Stage 3: control law  (cutoff CV, resonance CV) -> (log2 fc, K, gain dB).
# Neural law: MLP 2-H-H-3 (tanh), numpy + Adam, trained on the 64 per-setting fits.
# Baselines: physical law (exponential converter with soft top compression, linear resonance current
# with a dead zone) and bilinear interpolation of the grid. Cross-validated by holding out interior
# rows (resonance), columns (cutoff) and random points.
import numpy as np, sys
from scipy.optimize import least_squares
from load import NOTES, STEPS

V = lambda m: np.asarray(m) * 10.0 / 127.0        # MCV24 MIDI value -> volts (thesis, Table 1)

def targets():
    L = np.load('linfit_4.npz'); q = L['q']; settings = [tuple(s) for s in L['settings']]
    O = np.load('oscfit.npy', allow_pickle=True).item()
    rows = []
    for r in STEPS:
        for c in STEPS:
            if r <= 95:
                i = settings.index((r, c)); fc, K, gdb = np.exp(q[i, 0]), q[i, 1], q[i, 2] * 20 / np.log(10)
            else:
                fs = [O['files'][(n, r, c)] for n in NOTES]
                fc = np.exp(np.mean([np.log(f['fc']) for f in fs])); K = np.mean([f['K'] for f in fs])
                gdb = np.mean([20 * np.log10(f['g']) for f in fs])
            rows.append((r, c, np.log2(fc / 1000), K, gdb))
    return np.array(rows)

SC = np.array([12.0, 1.0, 0.5])                   # output scaling: 1/12 oct, K units, 2 dB  ~ equal weight
def norm_in(r, c): return np.stack([(V(c) - 5) / 3, (V(r) - 5) / 3], -1)

class MLP:
    def __init__(self, H=24, seed=0):
        g = np.random.default_rng(seed)
        self.W = [g.normal(0, 1 / np.sqrt(2), (2, H)), g.normal(0, 1 / np.sqrt(H), (H, H)), g.normal(0, 0.1 / np.sqrt(H), (H, 3))]
        self.b = [np.zeros(H), np.zeros(H), np.zeros(3)]
    def forward(self, x):
        self.a = [x]; h = x
        for i in range(2):
            h = np.tanh(h @ self.W[i] + self.b[i]); self.a.append(h)
        return h @ self.W[2] + self.b[2]
    def fit(self, x, y, steps=6000, lr=3e-3, wd=1e-4, val=None, every=25):
        # val = (x_val, y_val) optional held-out set; self.hist = [(step, train MSE, val MSE)] in scaled units
        P = self.W + self.b; m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]
        ys = y * SC; self.hist = []
        for t in range(1, steps + 1):
            out = self.forward(x); e = (out - ys) / len(x)
            if t == 1 or t % every == 0:
                tr = np.mean((out - ys) ** 2)
                va = np.mean((self.forward(val[0]) - val[1] * SC) ** 2) if val is not None else np.nan
                self.hist.append((t, tr, va)); self.forward(x)   # restore activations for backprop
            gW = [None] * 3; gb = [None] * 3
            d = e
            gW[2] = self.a[2].T @ d; gb[2] = d.sum(0)
            for i in (1, 0):
                d = (d @ self.W[i + 1].T) * (1 - self.a[i + 1] ** 2)
                gW[i] = self.a[i].T @ d; gb[i] = d.sum(0)
            G = [gw + wd * w for gw, w in zip(gW, self.W)] + gb
            for j, (p, gr) in enumerate(zip(P, G)):
                m[j] = 0.9 * m[j] + 0.1 * gr; v[j] = 0.999 * v[j] + 0.001 * gr ** 2
                p -= lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
        return self
    def __call__(self, x): return self.forward(x) / SC

class Ensemble:
    def __init__(self, x, y, n=5, **kw): self.m = [MLP(seed=s).fit(x, y, **kw) for s in range(n)]
    def __call__(self, x): return np.mean([m(x) for m in self.m], 0)

# physical law: log2 fc = a + b Vc - softplus(s (Vc - Vk)) / s  (expo converter, compression near the top)
#               K = kr * softplus(t (Vr - V0)) / t                (linear Iabc above a dead zone)
#               gain dB = g0 + g1 Vc + g2 Vc^2                    (small pass-band trims)
def phys(p, r, c):
    vc, vr = V(c), V(r); sp = lambda z: np.logaddexp(0, z)
    return np.stack([p[0] + p[1] * vc - sp(p[2] * (vc - p[3])) * p[4] / p[2],
                     p[5] * sp(p[6] * (vr - p[7])) / p[6] + p[8] * vr * vc / 10,
                     p[9] + p[10] * vc + p[11] * vc ** 2], -1)
def fit_phys(r, c, y):
    p0 = [-4.9, 0.89, 3.0, 9.2, 1.0, 0.75, 4.0, 2.4, 0.0, 0.0, 0.0, 0.0]
    return least_squares(lambda p: ((phys(p, r, c) - y) * SC).ravel(), p0).x

class Hybrid:
    """physical law + neural residual (MLP ensemble on what the physical law misses)"""
    def __init__(self, r, c, y, pp=None, n=5):
        self.pp = fit_phys(r, c, y) if pp is None else pp
        self.net = Ensemble(norm_in(r, c), y - phys(self.pp, r, c), n=n, steps=4000, wd=3e-3)
    def __call__(self, r, c): return phys(self.pp, r, c) + self.net(norm_in(r, c))

def bilinear(rtr, ctr, ytr, r, c):
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    pts = np.stack([V(ctr), V(rtr)], -1); q = np.stack([V(c), V(r)], -1)
    out = LinearNDInterpolator(pts, ytr)(q)
    nn = NearestNDInterpolator(pts, ytr)(q); out[np.isnan(out)] = nn[np.isnan(out)]
    return out

def cv_eval(T):
    r, c, y = T[:, 0], T[:, 1], T[:, 2:]
    interior = [s for s in STEPS[1:-1]]
    schemes = {
        'row': [(T[:, 0] == s) for s in interior],
        'col': [(T[:, 1] == s) for s in interior],
        'rand': [np.isin(np.arange(len(T)) % 8, [k]) & np.isin(T[:, 0], interior) & np.isin(T[:, 1], interior) for k in range(8)],
    }
    res = {}
    for name, folds in schemes.items():
        err = {'neural': [], 'physical': [], 'interp': [], 'hybrid': []}
        for te in folds:
            tr = ~te
            x = norm_in(r, c)
            net = Ensemble(x[tr], y[tr])
            pp = fit_phys(r[tr], c[tr], y[tr])
            err['neural'].append(net(x[te]) - y[te])
            err['physical'].append(phys(pp, r[te], c[te]) - y[te])
            err['interp'].append(bilinear(r[tr], c[tr], y[tr], r[te], c[te]) - y[te])
            hy = Hybrid(r[tr], c[tr], y[tr], pp)
            err['hybrid'].append(hy(r[te], c[te]) - y[te])
        res[name] = {k: np.concatenate(v) for k, v in err.items()}
    return res

if __name__ == '__main__':
    T = targets()
    print(' r   c   fc(Hz)     K    gain dB')
    for row in T: print(f'{int(row[0]):3d} {int(row[1]):3d} {1000*2**row[2]:8.1f} {row[3]:6.2f} {row[4]:7.2f}')
    if 'cv' in sys.argv:
        res = cv_eval(T)
        print('\nheld-out RMS error:   fc (cents) |   K   | gain (dB)')
        for s, d in res.items():
            for k, e in d.items():
                rm = np.sqrt(np.mean(e ** 2, 0))
                print(f'  {s:4s} {k:9s} {1200*rm[0]:9.1f} | {rm[1]:5.3f} | {rm[2]:5.2f}')
    if 'final' in sys.argv:
        hy = Hybrid(T[:, 0], T[:, 1], T[:, 2:]); net = hy.net
        e = hy(T[:, 0], T[:, 1]) - T[:, 2:]
        print('train RMS: fc %.1f cents, K %.3f, gain %.2f dB' % (1200 * np.sqrt(np.mean(e[:, 0] ** 2)), np.sqrt(np.mean(e[:, 1] ** 2)), np.sqrt(np.mean(e[:, 2] ** 2))))
        np.savez('law_mlp.npz', **{f'm{i}_W{j}': m.W[j] for i, m in enumerate(net.m) for j in range(3)},
                 **{f'm{i}_b{j}': m.b[j] for i, m in enumerate(net.m) for j in range(3)}, SC=SC)
        np.save('law_phys.npy', hy.pp)
