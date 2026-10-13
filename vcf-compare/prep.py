# Build data.js for the comparison page: recordings (MP3, base64), recording spectra/waveforms
# (from the original WAVs), reconstructed VCO source per note, leave-one-out filter parameters.
import sys, os, json, base64, subprocess, tempfile, numpy as np, scipy.io.wavfile as wavfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'vcf-model'))
os.chdir(os.path.join(HERE, '..', 'vcf-model'))
from load import NOTES, STEPS, SR, ROOT
from source import source
F = np.load('features.npy', allow_pickle=True).item(); O = np.load('oscfit.npy', allow_pickle=True).item()
LOO = json.load(open(os.path.join(HERE, 'loo_params.json')))
NFFT = 32768; NB = 300; FMIN, FMAX = 30.0, 20000.0
edges = np.geomspace(FMIN, FMAX, NB + 1)

def spec_bins(x):                      # same as the page's JS: Blackman, N=32768, peak-hold per log bin
    w = np.blackman(NFFT); X = np.abs(np.fft.rfft(x[:NFFT] * w)) / w.sum() * 2
    f = np.fft.rfftfreq(NFFT, 1 / SR); out = np.full(NB, -140.0)
    idx = np.searchsorted(edges, f) - 1
    for b in range(NB):
        m = idx == b
        if m.any(): out[b] = 20 * np.log10(X[m].max() + 1e-9)
    return np.round(np.maximum(out, -140)).astype(int).tolist()

data = dict(sr=SR, nfft=NFFT, fmin=FMIN, fmax=FMAX, nb=NB, drive=O['drive'], q=O['q'], steps=STEPS, notes={}, rec={}, params=LOO)
for n in NOTES:
    f0, k, amp, ph = source(n)
    data['notes'][n] = dict(f0=f0, amp=np.round(amp, 7).tolist(), ph=np.round(ph, 4).tolist())
tmp = tempfile.mkdtemp()
for n in NOTES:
    for r in STEPS:
        for c in STEPS:
            sr, x = wavfile.read(f'{ROOT}/{n}/{n}_r{r:03d}_c{c:03d}.wav'); x = x.astype(np.float64) / 32768
            seg = x[int(1.0 * SR):int(2.2 * SR)].copy(); fade = int(0.01 * SR)
            seg[:fade] *= np.linspace(0, 1, fade); seg[-fade:] *= np.linspace(1, 0, fade)
            wavfile.write(f'{tmp}/a.wav', SR, (seg * 32767).astype(np.int16))
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', f'{tmp}/a.wav', '-codec:a', 'libmp3lame', '-b:a', '96k', f'{tmp}/a.mp3'], check=True)
            per = SR / F[(n, 15, 79)]['f0']; L = int(round(2 * per))
            s0 = x[int(1.2 * SR):]
            data['rec'][f'{n}_{r}_{c}'] = dict(
                mp3=base64.b64encode(open(f'{tmp}/a.mp3', 'rb').read()).decode(),
                spec=spec_bins(s0),
                wave=np.round(s0[:L + int(per) + 2] * 1000).astype(int).tolist(),      # 3 periods, page aligns and shows 2
                osc=float(F[(n, r, c)]['osc_f']) if r >= 111 else None)
    print(n, flush=True)
js = 'const DATA=' + json.dumps(data, separators=(',', ':')) + ';\n'
open(os.path.join(HERE, 'data.js'), 'w').write(js); print('data.js', round(len(js) / 1e6, 2), 'MB')
