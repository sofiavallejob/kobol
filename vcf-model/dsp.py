import ctypes, os, subprocess, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, 'libkobol.dylib')
src = [os.path.join(HERE, 'capi.cpp')] + [os.path.join(HERE, '..', 'dsp', f) for f in os.listdir(os.path.join(HERE, '..', 'dsp')) if f.endswith('.h')]
if not os.path.exists(LIB) or max(os.path.getmtime(s) for s in src) > os.path.getmtime(LIB):
    subprocess.check_call(['c++', '-O3', '-std=c++17', '-shared', '-fPIC', '-o', LIB, src[0]])
_l = ctypes.CDLL(LIB)
P = ctypes.POINTER(ctypes.c_double); D = ctypes.c_double
_l.vcf_render.argtypes = [P, P, ctypes.c_int, D, D, D, D, D, ctypes.c_int, P, D]
def ptr(a): return a.ctypes.data_as(P)
def vcf(x, fs, fc, K, drive, gain=1.0, iters=2, q=1.0):
    x = np.ascontiguousarray(x, np.float64); y = np.empty_like(x)
    fa = None
    if np.ndim(fc): fa = np.ascontiguousarray(fc, np.float64)
    _l.vcf_render(ptr(x), ptr(y), len(x), fs, 0.0 if fa is not None else fc, K, drive, gain, iters, ptr(fa) if fa is not None else None, q)
    return y
_l.vcf_law.argtypes = [D, D, P]
def law_cpp(cutoffV, resV):
    o = np.zeros(3); _l.vcf_law(cutoffV, resV, ptr(o)); return o
_l.env_vca_render.argtypes = [P, P, P, P, ctypes.c_int, D, D, D, D]
def env_vca(x, gate, fs, a, d, s):
    x = np.ascontiguousarray(x, np.float64); g = np.ascontiguousarray(gate, np.float64)
    e = np.empty_like(x); y = np.empty_like(x)
    _l.env_vca_render(ptr(x), ptr(g), ptr(e), ptr(y), len(x), fs, a, d, s); return e, y
_l.voice_render.argtypes = [P, ctypes.c_int, D, ctypes.c_int] + [D] * 9; _l.voice_render.restype = D
def voice(dur, fs, note, cutoffV, resV, envAmt=0.0, a1=0.0, d1=0.5, s1=1.0, a2=0.0, d2=0.4, s2=0.3):
    y = np.zeros(int(dur * fs)); t = _l.voice_render(ptr(y), len(y), fs, note, cutoffV, resV, envAmt, a1, d1, s1, a2, d2, s2)
    return y, t
