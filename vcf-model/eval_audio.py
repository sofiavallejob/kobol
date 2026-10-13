# End-to-end evaluation on held-out settings: control law (trained without the held-out setting)
# -> C++ SSM2040 driven by the reconstructed VCO -> compare with the recording.
# Metrics per file:
#   harm_dB : RMS dB error over harmonics above -80 dBFS (magnitude only)
#   ESR     : error-to-signal ratio of the periodic waveform (harmonic domain, best time alignment);
#             same kind of metric as the thesis' validation loss (~0.95 for PedalNet)
#   osc     : self-oscillation frequency error (cents) and level error (dB), r >= 111
import numpy as np, sys
from law import targets, Hybrid, fit_phys, phys, bilinear
from fit_osc import sim_feats, SRC, X, FS, SEG
from source import F
from dsp import vcf
from feats import harmonics_fs, feats
from load import NOTES, STEPS
OSC = np.load('oscfit.npy', allow_pickle=True).item(); DRIVE, Q = OSC['drive'], OSC['q']

def render(n, fc, K, g):
    return vcf(X[n], FS, fc, K, DRIVE, 10 ** (g / 20), q=Q)[SEG]

def metrics(n, r, c, y):
    d = F[(n, r, c)]; f0 = SRC[n][0]
    k, H = harmonics_fs(y, FS, f0, 400)
    Hr = d['H'][:len(k)]; k = k[:len(Hr)]; H = H[:len(Hr)]
    m = np.abs(Hr) > 10 ** (-80 / 20)
    out = dict(harm_dB=np.sqrt(np.mean((20 * np.log10(np.abs(H[m]) + 1e-12) - 20 * np.log10(np.abs(Hr[m]))) ** 2)))
    th = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    E = np.abs(H[None, :] * np.exp(1j * np.outer(th, k)) - Hr[None, :]) ** 2
    out['ESR'] = E.sum(1).min() / (np.abs(Hr) ** 2).sum()
    if r >= 111:
        _, _, fo, ao, _ = feats(y, FS, f0, kmax=6)
        out['cents'] = 1200 * np.log2(fo / d['osc_f']); out['osc_dB'] = 20 * np.log10(ao / d['osc_a'])
    return out

def run(scheme):
    T = targets(); r, c, y = T[:, 0], T[:, 1], T[:, 2:]
    interior = STEPS[1:-1]
    if scheme == 'rand':
        folds = [(np.arange(len(T)) % 8 == k) & np.isin(r, interior) & np.isin(c, interior) for k in range(8)]
    elif scheme == 'col':
        folds = [c == s for s in interior]
    elif scheme == 'row':
        folds = [r == s for s in interior]
    else:   # 'train': fitted on everything (what ships in the plugin)
        folds = [np.ones(len(T), bool)]
    res = []
    for te in folds:
        tr = ~te if scheme != 'train' else te
        laws = {'hybrid': Hybrid(r[tr], c[tr], y[tr])}
        if scheme != 'train':
            laws['interp'] = lambda rr, cc, tr=tr: bilinear(r[tr], c[tr], y[tr], rr, cc)
        for i in np.where(te)[0]:
            for name, law in laws.items():
                p = law(r[i:i + 1], c[i:i + 1])[0]
                for n in NOTES:
                    yy = render(n, 1000 * 2 ** p[0], p[1], p[2])
                    res.append(dict(method=name, note=n, r=int(r[i]), c=int(c[i]), **metrics(n, int(r[i]), int(c[i]), yy)))
    return res

def summary(res, scheme):
    for m in sorted(set(x['method'] for x in res)):
        R = [x for x in res if x['method'] == m]
        lin = [x for x in R if x['r'] <= 95]; osc = [x for x in R if x['r'] >= 111]
        s = f'{scheme:5s} {m:7s} n={len(R):3d} | r<=95: harm {np.median([x["harm_dB"] for x in lin]):5.2f} dB, ESR median {np.median([x["ESR"] for x in lin]):.4f}'
        if osc:
            s += (f' | r>=111: osc |cents| median {np.median([abs(x["cents"]) for x in osc]):6.1f}, '
                  f'|level| {np.median([abs(x["osc_dB"]) for x in osc]):4.2f} dB')
        print(s, flush=True)

if __name__ == '__main__':
    allres = {}
    for scheme in sys.argv[1:] or ['train', 'rand', 'col', 'row']:
        res = run(scheme); allres[scheme] = res; summary(res, scheme)
    np.save('eval_audio.npy', allres, allow_pickle=True)
