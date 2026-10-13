import numpy as np, subprocess, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from wdf import *
from scipy.signal import welch
FS=48000; res={}
def spice(net,name):
    open(f'sp/{name}.cir','w').write(net); subprocess.run(['ngspice','-b',f'sp/{name}.cir'],capture_output=True,timeout=280); return np.loadtxt(f'sp/{name}.dat')
# ---- MM5837: sequence period, spectrum, pink slope
m=MM5837(FS); reg0=m.reg; n=0
while True:
    m.clock(); n+=1
    if m.reg==reg0: break
res['mm5837_period_clocks']=n; res['mm5837_period_s']=n/P['mm5837']['fclk']
m=MM5837(FS); w=np.array([m.tick() for _ in range(FS*4)])
pk=Pink(); pnk=np.array([pk.tick(x) for x in w])
f,Pw=welch(w,FS,nperseg=8192); f,Pp=welch(pnk,FS,nperseg=8192)
sel=(f>100)&(f<10000); slope=np.polyfit(np.log2(f[sel]),10*np.log10(Pp[sel]),1)[0]; res['pink_slope_dB_per_oct']=float(slope)
res['white_flatness_dB_100Hz_10kHz']=float(np.ptp(10*np.log10(np.convolve(Pw[sel],np.ones(15)/15,'same')[7:-7])))
np.savez('noise_spec.npz',f=f,Pw=Pw,Pp=Pp)
# ---- OTA LFO: frequency vs knob, WDF vs ngspice
pl=P['otalfo']; pts=[0.15,0.4,0.65,0.9,1.0]; lf=[]
for x in pts:
    o=OTALFO(FS); o.set(x); ftar=pl['fmin']*(pl['fmax']/pl['fmin'])**x
    T=min(3.2/ftar,85); fsx=FS if ftar>1 else 2000
    o=OTALFO(fsx); o.set(x); tri=np.array([o.tick()[0] for _ in range(int(T*fsx))])
    zc=np.where((tri[:-1]<0)&(tri[1:]>=0))[0]; fw=fsx/np.mean(np.diff(zc)) if len(zc)>2 else np.nan
    I=o.iabc; st=min(1/ftar/2000,1e-3)
    d=spice(f"""otalfo
Bc qq 0 V = {pl['Vsat']}*tanh(200*({pl['ratio']}*v(q)-v(tri)))
Rd qq q 100
Cd q 0 1n
Bo 0 tri I = {I}*tanh(v(q)*{pl['Rdiv']/(pl['Rin']+pl['Rdiv'])}/(2*{pl['Vt']}))
C1 tri 0 {pl['C']}
.ic v(tri)=0 v(q)={pl['Vsat']} v(qq)={pl['Vsat']}
.tran {st} {T} uic
.control
run
wrdata sp/otalfo.dat v(tri)
.endc
.end""",'otalfo')
    ts,vs=d[:,0],d[:,1]; z=ts[np.where((vs[:-1]<0)&(vs[1:]>=0))[0]]; fsp=1/np.mean(np.diff(z)) if len(z)>2 else np.nan
    lf.append(dict(knob=x,target_Hz=ftar,wdf_Hz=float(fw),ngspice_Hz=float(fsp),err_pct=float(100*(fw-fsp)/fsp)))
res['otalfo']=lf
# waveforms with symmetry offsets
wv={}
for vo in (0.0,0.012):
    o=OTALFO(FS); o.set(0.62); o.voff=vo; out=np.array([o.tick() for _ in range(FS)]); wv[vo]=out
np.savez('lfo_wave.npz',w0=wv[0.0],w1=wv[0.012])
json.dump(res,open('validation2.json','w'),indent=1,default=float); print(json.dumps(res,indent=1,default=float))
