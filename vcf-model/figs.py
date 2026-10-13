# All VCF figures: dataset, fitting diagnostics, training curves, cross-validation, evaluation.
# PNG -> vcf-model/figs/ (browse), PDF -> ../paper-latex/figs/ (paper). Style matches wdf-modules/paperfigs.py.
#   python figs.py            (all figures; heavy parts are cached in figs/cache_*.npy)
#   python figs.py data law   (only some)
import os, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from load import NOTES, STEPS, load, spectrum, SR

plt.rcParams.update({'font.size': 8.5, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .25, 'grid.linewidth': .5, 'legend.frameon': False,
                     'legend.fontsize': 7})
C_ML = '#2F6FB3'; C_WDF = '#D9822B'; C_BEH = '#7A7F85'; C_REC = '#111111'; C_RED = '#C0392B'; C_GRN = '#3E8E8A'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figs'); PAPER = os.path.join(HERE, '..', 'paper-latex', 'figs')
os.makedirs(OUT, exist_ok=True); os.makedirs(PAPER, exist_ok=True)
V = lambda m: np.asarray(m) * 10.0 / 127.0
ramp = LinearSegmentedColormap.from_list('r', ['#9DC3E6', C_ML, '#0B2545'])
F = np.load(os.path.join(HERE, 'features.npy'), allow_pickle=True).item()
db = lambda x: 20 * np.log10(np.abs(x) + 1e-12)

def save(fig, name, paper=True):
    fig.savefig(f'{OUT}/{name}.png', dpi=200, bbox_inches='tight')
    if paper: fig.savefig(f'{PAPER}/{name}.pdf', bbox_inches='tight')
    plt.close(fig); print('  wrote', name)

def cached(name, fn):
    p = f'{OUT}/cache_{name}.npy'
    if os.path.exists(p): return np.load(p, allow_pickle=True).item()
    r = fn(); np.save(p, r, allow_pickle=True); return r

def lin_rows(N):
    """the harmonic data used by fit_linear.py, and its residuals for the N-pole fit"""
    L = np.load(f'{HERE}/linfit_{N}.npz'); S, q = L['S'], L['q']; settings = [tuple(s) for s in L['settings']]
    out = []
    for ni, n in enumerate(NOTES):
        for si, (r, c) in enumerate(settings):
            d = F[(n, r, c)]
            for k, h in zip(d['k'], d['H']):
                if abs(h) > 10 ** (-80 / 20) and k * d['f0'] < 20000 and k <= S.shape[1]:
                    f = k * d['f0']; fc, K, g = np.exp(q[si, 0]), q[si, 1], q[si, 2]
                    m = S[ni, k - 1] + g - np.log(np.abs((1 + 1j * f / fc) ** N + K))
                    out.append((ni, r, c, k, f, np.log(abs(h)), m))
    return np.array(out)

# ------------------------------------------------------------------ 1. dataset
def fig_data():
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.9]))
    for i, c in enumerate(STEPS):
        d = F[('B3', 15, c)]; ax[0].plot(d['k'] * d['f0'], db(d['H']), '.-', ms=2.2, lw=.6, color=ramp(i / 7), label=f'{c}')
    ax[0].set(xscale='log', xlim=(200, 20000), ylim=(-95, -5), xlabel='Frequency (Hz)', ylabel='Harmonic level (dBFS)',
              title='(a) cutoff sweep (B3, res. 15)')
    ax[0].legend(ncol=2, title='cutoff', fontsize=6, title_fontsize=6.5, loc='lower left', handlelength=1)
    for i, r in enumerate([15, 63, 95, 127]):
        f, A = spectrum(load('B3', r, 63)); m = (f > 100) & (f < 6000)
        L = db(A[m]) - 70 * i
        ax[1].plot(f[m], L, lw=.45, color=ramp(i / 3) if r < 127 else C_RED)
        ax[1].text(6300, np.median(L) + 12, f'r {r}' + (' osc.' if r == 127 else ''), fontsize=6.5, color=ramp(i / 3) if r < 127 else C_RED)
    ax[1].set(xscale='log', xlim=(100, 11000), ylim=(-360, 5), xlabel='Frequency (Hz)', ylabel='Level (dB, 70 dB offsets)',
              title='(b) resonance up (B3, cutoff 63)')
    ax[1].set_yticklabels([])
    M = np.array([[np.mean([db(F[(n, r, c)]['res_rms']) for n in NOTES]) for c in STEPS] for r in STEPS])
    im = ax[2].imshow(M, origin='lower', cmap='magma', vmin=-70, vmax=0, aspect='auto')
    ax[2].set(xticks=range(8), yticks=range(8), xlabel='Cutoff (MIDI)', ylabel='Resonance (MIDI)', title='(c) non-harm. level (dB)')
    ax[2].set_xticklabels(STEPS, fontsize=6, rotation=90); ax[2].set_yticklabels(STEPS, fontsize=6); ax[2].grid(False)
    fig.colorbar(im, ax=ax[2], fraction=.05, pad=.03)
    fig.tight_layout(); save(fig, 'fig_vcf_data')

def fig_source():
    from source import source
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.2))
    for i, n in enumerate(NOTES):
        f0, k, amp, ph = source(n); kk = k[:40]
        ax[0].plot(kk, db(amp[:40] * kk / amp[0]), '.-', ms=2.5, lw=.7, color=ramp(i / 3), label=n)
        ax[1].plot(kk[:16], np.angle(np.exp(1j * (ph[:16] - ph[0] * kk[:16]))), '.-', ms=3, lw=.7, color=ramp(i / 3))
    ax[0].set(xlabel='Harmonic k', ylabel='dB re ideal saw (1/k)', title='(a) reconstructed VCO amplitude, per note'); ax[0].legend()
    ax[1].set(xlabel='Harmonic k', ylabel='Phase (rad)', title='(b) reconstructed phase (fundamental aligned)')
    fig.tight_layout(); save(fig, 'fig_vcf_source', paper=False)

# ------------------------------------------------------------------ 2. linear fit
def fig_linfit():
    R = lin_rows(4); L = np.load(f'{HERE}/linfit_4.npz'); S = L['S']
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw=dict(width_ratios=[1.2, 1.2, 0.75]))
    for j, (sel, lab, col) in enumerate([(lambda r, c: r == 15, 'cutoff', 2), (lambda r, c: c == 63, 'resonance', 1)]):
        ks = sorted(set((int(r), int(c)) for r, c in R[:, 1:3] if sel(r, c)))
        for i, (r, c) in enumerate(ks):
            m = (R[:, 0] == 0) & (R[:, 1] == r) & (R[:, 2] == c)          # B2: most harmonics
            src = S[0, R[m, 3].astype(int) - 1]
            col_ = ramp(i / max(1, len(ks) - 1))
            ax[j].plot(R[m, 4], (R[m, 5] - src) * 8.686, '.', ms=2.2, color=col_)
            o = np.argsort(R[m, 4]); ax[j].plot(R[m, 4][o], (R[m, 6] - src)[o] * 8.686, '-', lw=.8, color=col_,
                                                label=f'{c if j == 0 else r}')
        ax[j].set(xscale='log', xlim=(50, 20000), ylim=(-80, 15), xlabel='Frequency (Hz)', ylabel='|H| (dB)')
        ax[j].legend(ncol=2, fontsize=6, title=lab, title_fontsize=6.5, loc='lower left', handlelength=1)
    ax[0].set_title('(a) cutoff sweep, res. 15 (B2)'); ax[1].set_title('(b) resonance sweep, cutoff 63')
    ax[0].text(.98, .97, 'dots: data / source\nlines: 4-pole fit', transform=ax[0].transAxes, ha='right', va='top', fontsize=6.5)
    errs = []
    for N in (2, 3, 4, 6):
        RN = lin_rows(N); e = (RN[:, 6] - RN[:, 5]) * 8.686; errs.append((np.sqrt(np.mean(e ** 2)), np.median(np.abs(e))))
    errs = np.array(errs); x = np.arange(4)
    ax[2].bar(x - .18, errs[:, 0], .36, color=[C_BEH, C_BEH, C_ML, C_BEH], label='RMS')
    ax[2].bar(x + .18, errs[:, 1], .36, color=[C_BEH, C_BEH, C_ML, C_BEH], alpha=.45, label='median |e|')
    for i, v in enumerate(errs[:, 0]): ax[2].text(i - .18, v + .06, f'{v:.2f}', ha='center', fontsize=6.5)
    ax[2].set(xticks=x, xticklabels=['2', '3', '4', '6'], xlabel='Poles N', ylabel='Error (dB)', title='(c) model order'); ax[2].legend()
    fig.tight_layout(); save(fig, 'fig_vcf_linfit')

# ------------------------------------------------------------------ 3. parameters and control law
def law_curves():
    from law import targets, Hybrid, fit_phys, phys
    T = targets(); hy = Hybrid(T[:, 0], T[:, 1], T[:, 2:]); pp = fit_phys(T[:, 0], T[:, 1], T[:, 2:])
    mid = np.linspace(0, 127, 200); out = dict(T=T, mid=mid, hy={}, ph={})
    for r in STEPS:
        out['hy'][('c', r)] = hy(np.full_like(mid, r), mid); out['ph'][('c', r)] = phys(pp, np.full_like(mid, r), mid)
    for c in STEPS:
        out['hy'][('r', c)] = hy(mid, np.full_like(mid, c)); out['ph'][('r', c)] = phys(pp, mid, np.full_like(mid, c))
    return out

def fig_params():
    D = cached('law_curves', law_curves); T, mid = D['T'], D['mid']
    O = np.load(f'{HERE}/oscfit.npy', allow_pickle=True).item()['files']
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for i, r in enumerate(STEPS):
        col = ramp(i / 7) if r <= 95 else (C_RED if r == 127 else C_WDF); m = T[:, 0] == r
        ax[0].plot(V(T[m, 1]), 1000 * 2 ** T[m, 2], 'o', ms=2.6, color=col, label=f'r {r}')
        ax[0].plot(V(mid), 1000 * 2 ** D['hy'][('c', r)][:, 0], '-', lw=.7, color=col)
        ax[2].plot(V(T[m, 1]), T[m, 4], 'o', ms=2.6, color=col); ax[2].plot(V(mid), D['hy'][('c', r)][:, 2], '-', lw=.7, color=col)
    oscf = [np.mean([F[(n, 127, c)]['osc_f'] for n in NOTES]) for c in STEPS]
    ax[0].plot(V(STEPS), oscf, 'x', ms=4, color=C_REC, label='osc. freq. r127')
    ax[0].plot(V(mid), 1000 * 2 ** D['ph'][('c', 15)][:, 0], ':', lw=.9, color=C_REC, label='physical law')
    ax[0].set(yscale='log', xlabel='Cutoff CV (V)', ylabel='$f_c$ (Hz)', title='(a) cutoff'); ax[0].legend(ncol=2, fontsize=5.6, handlelength=1, loc='lower right')
    for i, c in enumerate(STEPS):
        m = T[:, 1] == c; col = ramp(i / 7)
        ax[1].plot(V(T[m, 0]), T[m, 3], 'o', ms=2.6, color=col); ax[1].plot(V(mid), D['hy'][('r', c)][:, 1], '-', lw=.7, color=col)
    for (n, r, c), o in O.items(): ax[1].plot(V(r) + (NOTES.index(n) - 1.5) * .05, o['K'], '.', ms=1.5, color=C_BEH)
    ax[1].axhline(4, color=C_RED, lw=.7, ls='--'); ax[1].text(.3, 4.25, 'K = 4: self-oscillation', fontsize=6.5, color=C_RED)
    ax[1].set(xlabel='Resonance CV (V)', ylabel='Feedback K', title='(b) resonance (lines: one per cutoff)')
    ax[2].set(xlabel='Cutoff CV (V)', ylabel='Gain (dB)', title='(c) pass-band gain')
    fig.tight_layout(); save(fig, 'fig_vcf_params')

# ------------------------------------------------------------------ 4. nonlinear model
def osc_experiment():
    from dsp import vcf
    fs = 96000; n = int(0.6 * fs); x = np.zeros(n); x[:50] = 1e-3
    Ks = np.linspace(4.3, 10, 12); res = {}
    for name, drive, q in [('cells only', 1.0, 1e-9), ('cells + feedback OTA', 0.1, 18.0)]:
        ratio = []
        for K in Ks:
            y = vcf(x, fs, 700.0, K, drive, 1.0, q=q)[int(.3 * fs):]
            zc = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
            ratio.append((len(zc) - 1) / ((zc[-1] - zc[0]) / fs) / 700 if len(zc) > 3 else np.nan)
        res[name] = np.array(ratio)
    res['K'] = Ks; return res

def osc_spectra():
    from fit_osc import X, FS, SEG
    from dsp import vcf
    O = np.load(f'{HERE}/oscfit.npy', allow_pickle=True).item(); out = {}
    for key in [('B3', 127, 63), ('B5', 111, 95)]:
        o = O['files'][key]; y = vcf(X[key[0]], FS, o['fc'], o['K'], O['drive'], o['g'], q=O['q'])[SEG]
        f, A = spectrum(y[::2])                    # 96k -> 48k (content is band-limited)
        fr, Ar = spectrum(load(*key))
        out[key] = (f, A, fr, Ar)
    return out

def fig_nonlinear():
    E = cached('osc_experiment', osc_experiment); S = cached('osc_spectra', osc_spectra)
    O = np.load(f'{HERE}/oscfit.npy', allow_pickle=True).item()['files']
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.2), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15, .8]))
    rec = [F[(n, r, c)]['osc_f'] / np.exp(np.mean([np.log(O[(m, r, c)]['fc']) for m in NOTES])) for n in NOTES for r in (111, 127) for c in STEPS]
    ax[0].plot(E['K'], E['cells only'], 'o-', ms=3, color=C_BEH, label='cell tanh only')
    ax[0].plot(E['K'], E['cells + feedback OTA'], 'o-', ms=3, color=C_ML, label='+ CA3080 feedback')
    ax[0].axhline(1, color=C_REC, lw=.6, ls=':')
    ax[0].set(xlabel='Feedback K', ylabel='$f_{osc}/f_c$', ylim=(.6, 1.05), title='(a) oscillation frequency'); ax[0].legend(loc='lower left')
    for j, key in enumerate(S):
        from scipy.signal import find_peaks
        f, A, fr, Ar = S[key]
        for (ff, AA, mk, col, lab, ms) in [(fr, Ar, 'o', C_REC, 'recording', 3.2), (f, A, 'x', C_ML, 'model', 3.5)]:
            m = (ff > 40) & (ff < 20000); i, _ = find_peaks(db(AA[m]), distance=60, prominence=10)
            i = i[np.argsort(AA[m][i])[::-1][:45]]
            ax[1 + j].vlines(ff[m][i], -110, db(AA[m][i]), color=col, lw=.4, alpha=.5)
            ax[1 + j].plot(ff[m][i], db(AA[m][i]), mk, ms=ms, mfc='none' if mk == 'o' else col, color=col, mew=.8, label=lab)
        ax[1 + j].set(xscale='log', xlim=(50, 20000), ylim=(-110, 5), xlabel='Frequency (Hz)', title=f'({"bc"[j]}) {key[0]} r{key[1]} c{key[2]}, peaks')
        if j == 0: ax[1 + j].set_ylabel('Level (dB)'); ax[1 + j].legend(loc='lower left', fontsize=6)
    cents = [o['cents'] for o in O.values()]; adb = [o['adb'] for o in O.values()]
    ax[3].hist(np.clip(np.abs(cents), 0, 5), bins=20, color=C_ML, alpha=.8)
    ax[3].set(xlabel='|osc. error| (cents)', ylabel='files', title='(d) 64 osc. files')
    ax[3].text(.95, .9, f'median {np.median(np.abs(cents)):.2f} c\nlevel {np.median(np.abs(adb)):.2f} dB', transform=ax[3].transAxes, ha='right', va='top', fontsize=6.5)
    fig.tight_layout(); save(fig, 'fig_vcf_nonlinear')

# ------------------------------------------------------------------ 5. training curves
def training_runs():
    from law import targets, MLP, Hybrid, fit_phys, phys, norm_in
    T = targets(); r, c, y = T[:, 0], T[:, 1], T[:, 2:]
    out = {}
    for name, te in [('col63', c == 63), ('rand', np.isin(np.arange(64), [10, 21, 29, 36, 45, 50, 19, 42]))]:
        tr = ~te; x = norm_in(r, c); pp = fit_phys(r[tr], c[tr], y[tr])
        pure = [MLP(seed=s).fit(x[tr], y[tr], val=(x[te], y[te])).hist for s in range(5)]
        hyb = [MLP(seed=s).fit(x[tr], y[tr] - phys(pp, r[tr], c[tr]), steps=4000, wd=3e-3,
                               val=(x[te], y[te] - phys(pp, r[te], c[te]))).hist for s in range(5)]
        phys_va = np.mean(((phys(pp, r[te], c[te]) - y[te]) * np.array([12, 1, .5])) ** 2)
        out[name] = dict(pure=pure, hyb=hyb, phys_va=phys_va)
    return out

def fig_training():
    D = cached('training', training_runs)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
    for j, (k, title) in enumerate([('col63', '(a) whole cutoff column 63 held out'), ('rand', '(b) 8 random interior settings held out')]):
        for s, (hp, hh) in enumerate(zip(D[k]['pure'], D[k]['hyb'])):
            hp, hh = np.array(hp), np.array(hh)
            ax[j].plot(hp[:, 0], hp[:, 1], color=C_BEH, lw=.6, alpha=.7, label='pure MLP, train' if s == 0 else None)
            ax[j].plot(hp[:, 0], hp[:, 2], color=C_BEH, lw=.9, ls='--', alpha=.9, label='pure MLP, held-out' if s == 0 else None)
            ax[j].plot(hh[:, 0], hh[:, 1], color=C_ML, lw=.6, alpha=.7, label='hybrid residual, train' if s == 0 else None)
            ax[j].plot(hh[:, 0], hh[:, 2], color=C_ML, lw=.9, ls='--', alpha=.9, label='hybrid, held-out' if s == 0 else None)
        ax[j].axhline(D[k]['phys_va'], color=C_WDF, lw=.9, ls=':', label='physical law alone, held-out')
        ax[j].set(yscale='log', xscale='log', xlabel='Adam step', title=title)
    ax[0].set_ylabel('MSE (scaled: 1/12 oct, K, 2 dB)'); ax[1].legend(fontsize=6, loc='upper right')
    fig.tight_layout(); save(fig, 'fig_vcf_training')

# ------------------------------------------------------------------ 6. law cross-validation
def fig_lawcv():
    def run():
        from law import targets, cv_eval
        return cv_eval(targets())
    R = cached('law_cv', run)
    meths = ['neural', 'physical', 'interp', 'hybrid']; labels = ['pure MLP', 'physical', 'interpolation', 'hybrid (ours)']
    cols = [C_BEH, C_WDF, C_GRN, C_ML]; schemes = ['rand', 'col', 'row']
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    for q, (scale, unit) in enumerate([(1200, 'cents'), (1, ''), (1, 'dB')]):
        for i, m in enumerate(meths):
            v = [np.sqrt(np.mean(R[s][m][:, q] ** 2)) * scale for s in schemes]
            ax[q].bar(np.arange(3) + (i - 1.5) * .2, v, .2, color=cols[i], label=labels[i])
        ax[q].set(xticks=range(3), xticklabels=['random', 'columns', 'rows'],
                  title=['(a) cutoff $f_c$', '(b) feedback K', '(c) gain'][q], ylabel=f'held-out RMS {unit}'.strip())
    ax[0].set_yscale('log'); ax[0].set_yticks([10, 30, 100, 300]); ax[0].set_yticklabels(['10', '30', '100', '300']); ax[0].legend(fontsize=6)
    fig.tight_layout(); save(fig, 'fig_vcf_lawcv')

# ------------------------------------------------------------------ 7. end-to-end evaluation
def fig_eval():
    A = np.load(f'{HERE}/eval_audio.npy', allow_pickle=True).item()
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    for s, ls in [('rand', '-'), ('col', '--'), ('row', ':')]:
        for m, col in [('hybrid', C_ML), ('interp', C_GRN)]:
            e = np.sort([10 * np.log10(x['ESR']) for x in A[s] if x['method'] == m and x['r'] <= 95])
            ax[0].plot(e, np.linspace(0, 1, len(e)), ls=ls, color=col, lw=1, label=f'{"ours" if m == "hybrid" else "interp."}, {dict(rand="random", col="columns", row="rows")[s]}')
    ax[0].axvline(10 * np.log10(0.95), color=C_RED, lw=1); ax[0].text(-1.5, .55, 'thesis\nPedalNet', color=C_RED, fontsize=6.5, ha='right')
    ax[0].set(xlabel='Waveform ESR (dB)', ylabel='Fraction of held-out files', xlim=(-45, 5), title='(a) ESR, non-oscillating')
    ax[0].legend(fontsize=5.6, loc='lower right', handlelength=1.6)
    for j, s in enumerate(['rand', 'col']):
        M = np.full((8, 8), np.nan)
        for i, r in enumerate(STEPS):
            for k, c in enumerate(STEPS):
                v = [x['ESR'] if x['r'] <= 95 else 10 ** (-abs(x['cents']) / 100) for x in A[s] if x['method'] == 'hybrid' and x['r'] == r and x['c'] == c]
                if v: M[i, k] = 10 * np.log10(np.median(v)) if r <= 95 else np.nan
        im = ax[1 + j].imshow(M, origin='lower', cmap='viridis_r', vmin=-45, vmax=0, aspect='auto')
        for i, r in enumerate(STEPS):
            for k, c in enumerate(STEPS):
                cs = [abs(x['cents']) for x in A[s] if x['method'] == 'hybrid' and x['r'] == r and x['c'] == c and r >= 111]
                if cs: ax[1 + j].text(k, i, f'{np.median(cs):.0f}', ha='center', va='center', fontsize=5.5, color=C_RED)
        ax[1 + j].set(xticks=range(8), yticks=range(8), xlabel='Cutoff (MIDI)', title=f'({"bc"[j]}) {"random settings" if s == "rand" else "whole columns"} held out')
        ax[1 + j].set_xticklabels(STEPS, fontsize=5.5, rotation=90); ax[1 + j].set_yticklabels(STEPS, fontsize=5.5); ax[1 + j].grid(False)
    ax[1].set_ylabel('Resonance (MIDI)'); cb = fig.colorbar(im, ax=ax[2], fraction=.05, pad=.03); cb.set_label('median ESR (dB)', fontsize=7)
    fig.tight_layout(); save(fig, 'fig_vcf_eval')

def cycles_data():
    from law import targets, Hybrid
    from fit_osc import X, FS, SEG, SRC
    from dsp import vcf
    from feats import harmonics_fs
    O = np.load(f'{HERE}/oscfit.npy', allow_pickle=True).item(); T = targets()
    picks = [('B2', 31, 47), ('B3', 63, 79), ('B4', 79, 63), ('B2', 95, 31), ('B3', 47, 111), ('B5', 79, 95)]
    out = {}
    for n, r, c in picks:
        te = (T[:, 0] == r) & (T[:, 1] == c)
        hy = Hybrid(T[~te, 0], T[~te, 1], T[~te, 2:]); p = hy(np.array([r]), np.array([c]))[0]
        y = vcf(X[n], FS, 1000 * 2 ** p[0], p[1], O['drive'], 10 ** (p[2] / 20), q=O['q'])[SEG]
        k, H = harmonics_fs(y, FS, SRC[n][0], 400); Hr = F[(n, r, c)]['H'][:len(k)]; k, H = k[:len(Hr)], H[:len(Hr)]
        th = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
        E = (np.abs(H[None] * np.exp(1j * np.outer(th, k)) - Hr[None]) ** 2).sum(1); t0 = th[np.argmin(E)]
        ph = np.linspace(0, 2, 800)
        wr = np.real(np.exp(2j * np.pi * np.outer(ph, k)) @ Hr); wm = np.real(np.exp(2j * np.pi * np.outer(ph, k)) @ (H * np.exp(1j * k * t0)))
        out[(n, r, c)] = (ph, wr, wm, E.min() / (np.abs(Hr) ** 2).sum())
    return out

def fig_cycles():
    D = cached('cycles', cycles_data)
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 3.0), sharex=True)
    for a, ((n, r, c), (ph, wr, wm, esr)) in zip(ax.ravel(), D.items()):
        a.plot(ph, wr, color=C_REC, lw=1.1, label='recording'); a.plot(ph, wm, color=C_ML, lw=.9, ls='--', label='model (held out)')
        a.set_title(f'{n}, res {r}, cutoff {c}   ESR {10*np.log10(esr):.0f} dB', fontsize=7.5); a.tick_params(labelsize=6.5)
    ax[0, 0].legend(fontsize=6, loc='lower left')
    for a in ax[1]: a.set_xlabel('Periods')
    fig.tight_layout(); save(fig, 'fig_vcf_cycles')

def fig_system():
    """system diagram of the full voice (replaces wdf-modules/paperfigs.py's version, which had no VCF)"""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    def box(ax, x, y, w, h, txt, col, sub=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=col, ec='none', alpha=.92))
        ax.text(x + w / 2, y + h / 2 + (0.13 if sub else 0), txt, ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        if sub: ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha='center', va='center', color='white', fontsize=6.6)
    def arr(ax, a, b, c='#444', ls='-', rad=0.0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9, color=c, lw=1.1, ls=ls, shrinkA=2, shrinkB=2, connectionstyle=f'arc3,rad={rad}'))
    fig, ax = plt.subplots(figsize=(7.2, 3.2)); ax.set_xlim(0, 16.4); ax.set_ylim(-0.2, 6.9); ax.axis('off'); ax.grid(False)
    box(ax, 0.2, 4.3, 2.4, 1.0, 'LFO', C_WDF, 'CA3080 integrator'); box(ax, 0.2, 2.9, 2.4, 1.0, 'Voltage proc.', C_WDF, 'WDF lag')
    box(ax, 0.2, 1.6, 2.4, 0.8, 'Keyboard CV', C_BEH); box(ax, 0.2, 0.0, 2.4, 1.0, 'Noise', C_WDF, 'MM5837 + WDF')
    box(ax, 3.5, 4.3, 2.4, 1.0, 'VCO 1', C_ML, 'grey-box, learned'); box(ax, 3.5, 2.9, 2.4, 1.0, 'VCO 2', C_ML, 'same model')
    box(ax, 6.8, 4.3, 1.8, 1.0, 'VCA 1', C_WDF, 'CA3080'); box(ax, 6.8, 2.9, 1.8, 1.0, 'VCA 2', C_WDF, 'CA3080')
    box(ax, 9.6, 3.6, 2.3, 1.0, 'VCF', C_ML, 'SSM2040, learned'); box(ax, 12.8, 3.6, 2.2, 1.0, 'Output VCA', C_WDF, 'CA3080')
    box(ax, 9.6, 1.7, 2.3, 0.9, 'ADS2', C_BEH, 'SSM2050'); box(ax, 12.8, 1.7, 2.2, 0.9, 'ADS1', C_BEH, 'SSM2050')
    arr(ax, (1.4, 2.4), (1.4, 2.9)); arr(ax, (2.6, 3.4), (3.5, 4.6)); arr(ax, (2.6, 3.4), (3.5, 3.4))
    arr(ax, (4.7, 4.3), (4.7, 3.9), C_ML, '--'); ax.text(4.8, 4.06, 'sync', fontsize=7, color=C_ML)
    arr(ax, (5.9, 4.8), (6.8, 4.8)); arr(ax, (5.9, 3.4), (6.8, 3.4)); arr(ax, (8.6, 4.8), (9.6, 4.3)); arr(ax, (8.6, 3.4), (9.6, 3.9))
    arr(ax, (2.6, 0.5), (9.9, 3.6), '#444', '-', 0.22); arr(ax, (11.9, 4.1), (12.8, 4.1))
    arr(ax, (10.75, 2.6), (10.75, 3.6), '#444', ':'); arr(ax, (13.9, 2.6), (13.9, 3.6), '#444', ':')
    arr(ax, (2.6, 4.95), (3.5, 4.95), '#444', ':')
    arr(ax, (1.4, 5.3), (7.7, 5.3), '#444', ':', -0.25); ax.text(4.6, 6.5, 'LFO modulation (pitch, VCAs)', fontsize=7, color='#444', ha='center')
    arr(ax, (15.0, 4.1), (15.7, 4.1)); ax.text(15.75, 4.1, 'out', va='center', fontsize=8)
    for i, (c, l) in enumerate([(C_ML, 'learned from recordings (VCO: 437, VCF: 256)'), (C_WDF, 'white-box: schematic + ngspice'), (C_BEH, 'circuit-behavioural')]):
        ax.add_patch(FancyBboxPatch((9.6, 0.95 - 0.5 * i), 0.35, 0.3, boxstyle='round,pad=0.01', fc=c, ec='none'))
        ax.text(10.1, 1.1 - 0.5 * i, l, va='center', fontsize=7.5)
    plt.tight_layout(); save(fig, 'fig_system')

ALL = dict(data=fig_data, source=fig_source, linfit=fig_linfit, params=fig_params, nonlinear=fig_nonlinear,
           training=fig_training, lawcv=fig_lawcv, eval=fig_eval, cycles=fig_cycles, system=fig_system)
if __name__ == '__main__':
    for k in (sys.argv[1:] or ALL):
        print(k); ALL[k]()
