import numpy as np, subprocess, os, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from wdf import *
plt.rcParams.update({'font.size':8,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False})
FS=48000; os.makedirs('sp',exist_ok=True); res={}
def spice(net,name,cols):
    open(f'sp/{name}.cir','w').write(net)
    subprocess.run(['ngspice','-b',f'sp/{name}.cir'],capture_output=True,timeout=120)
    d=np.loadtxt(f'sp/{name}.dat'); return d
# 1) noise colouring network: frequency response, WDF impulse response vs ngspice AC
p=P['noise']; n=NoiseColour(FS); N=1<<16
h=np.array([n.tick(1.0 if i==0 else 0.0) for i in range(N)]); H=np.fft.rfft(h); f=np.fft.rfftfreq(N,1/FS)
d=spice(f"""noise colour
V1 in 0 AC 1
R48 in a {p['R48']}
C8 a b {p['C8']}
C9 b 0 {p['C9']}
R50 b 0 {p['R50']}
.control
ac dec 60 1 24000
wrdata sp/noise.dat vdb(b)
.endc
.end""",'noise',2)
fs_,dbs=d[:,0],d[:,1]; dbw=np.interp(fs_,f,20*np.log10(np.abs(H)+1e-12))
m=(fs_>20)&(fs_<16000); res['noise_maxdev_dB_20Hz_16kHz']=float(np.max(np.abs(dbw[m]-dbs[m])))
fig,ax=plt.subplots(1,4,figsize=(7.2,1.9))
ax[0].semilogx(fs_,dbs,'k',lw=2,label='ngspice'); ax[0].semilogx(fs_,dbw,'C1--',lw=1,label='WDF'); ax[0].set_ylim(-12,3); ax[0].set_title('Noise colouring'); ax[0].set_xlabel('Hz'); ax[0].set_ylabel('dB'); ax[0].legend(frameon=False,fontsize=6)
# 2) lag: step response at three pot settings
T=1.2; t=np.arange(int(T*FS))/FS; out={}
for x in (0.3,0.6,0.9):
    L=Lag(FS); L.set(x); y=np.array([L.tick(5.0 if tt>0.01 else 0.0) for tt in t]); R=L.R
    d=spice(f"""lag
V1 in 0 PWL(0 0 0.01 0 0.0100001 5)
R1 in o {R}
C1 o 0 {P['lag']['C']}
.tran 20u {T}
.control
run
wrdata sp/lag{x}.dat v(o)
.endc
.end""",f'lag{x}',2)
    ys=np.interp(t,d[:,0],d[:,1]); out[x]=float(np.max(np.abs(y-ys)))
    ax[1].plot(t,ys,'k',lw=2); ax[1].plot(t,y,'C1--',lw=1)
res['lag_max_abs_err_V']=out; ax[1].set_title('Voltage processor lag'); ax[1].set_xlabel('s')
# 3) LFO: integrator + Schmitt, frequency vs ngspice
pl=P['lfo']; lf={}
for x in (0.2,0.5):
    o=LFO(FS); o.set(x); R=o.R; T=4.0 if x<0.4 else 8.0
    tri=np.array([o.tick()[0] for _ in range(int(T*FS))])
    zc=np.where((tri[:-1]<0)&(tri[1:]>=0))[0]; fw=FS/np.mean(np.diff(zc[1:]))
    th=pl['ratio']*pl['Vsat']
    d=spice(f"""lfo
* ideal integrator (high-gain VCVS) and Schmitt comparator as a hysteretic switch pair
Bq q 0 V = {pl['Vsat']}*(2*v(st)-1)
Rr q m {R}
C1 m tri {pl['C']}
E1 tri 0 0 m 1e6
* state flip-flop: st=1 pushes the triangle down; flip when tri crosses -th / +th
Bs st 0 V = 1
.control
.endc
.end""",'dummy',2) if False else None
    # analytic reference for the ideal circuit: f = 1/(4 R C th/Vsat)
    fa=1/(4*R*pl['C']*th/pl['Vsat']); lf[x]=dict(wdf_Hz=fw,ideal_circuit_Hz=fa,err_pct=100*(fw-fa)/fa)
    if x==0.5:
        tt=np.arange(len(tri))/FS; ax[2].plot(tt[:int(2.5/fa*FS)],tri[:int(2.5/fa*FS)],'C1',lw=1)
res['lfo']=lf; ax[2].set_title('LFO triangle'); ax[2].set_xlabel('s')
# 3b) LFO against ngspice transient with a behavioural Schmitt trigger
x=0.5; o=LFO(FS); o.set(x); R=o.R; th=pl['ratio']*pl['Vsat']
d=spice(f"""lfo spice
Bcmp qq 0 V = {pl['Vsat']}*tanh(200*(v(tri)+{pl['ratio']}*v(q)))
Rd qq q 100
Cd q 0 1n
Rr q m {R}
C1 m tri {pl['C']}
E1 tri 0 0 m 1e6
.ic v(tri)=0 v(q)={pl['Vsat']} v(qq)={pl['Vsat']}
.tran 20u 6 uic
.control
run
wrdata sp/lfo.dat v(tri)
.endc
.end""",'lfo',2)
ts,vs=d[:,0],d[:,1]; z=ts[np.where((vs[:-1]<0)&(vs[1:]>=0))[0]]; fsp=1/np.mean(np.diff(z[1:])) if len(z)>3 else float('nan')
tri=np.array([o.tick()[0] for _ in range(int(6*FS))]); zc=np.where((tri[:-1]<0)&(tri[1:]>=0))[0]; fw=FS/np.mean(np.diff(zc[1:]))
res['lfo_vs_ngspice']=dict(ngspice_Hz=fsp,wdf_Hz=fw,err_pct=100*(fw-fsp)/fsp)
# 4) OTA VCA: transfer curve and gain vs control
pv=P['vca']; v=OTAVCA(FS)
vin=np.linspace(-10,10,401); yw=[]
for x in vin:
    for _ in range(40): y=v.tick(x,8.0)
    yw.append(y)
d=spice(f"""ota
Vin in 0 0
Rin in d {pv['Rin']}
Rdiv d 0 {pv['Rdiv']}
Bo 0 o I = {min(max((8.0-0.6)/pv['Rc'],0),pv['Iabc_max'])}*(tanh(v(d)/(2*{pv['Vt']}))+{pv['feed']})
Rl o 0 {pv['Rload']}
Cl o 0 {pv['Cload']}
.dc Vin -10 10 0.05
.control
run
wrdata sp/ota.dat v(o)
.endc
.end""",'ota',2)
ys=np.interp(vin,d[:,0],d[:,1]); res['vca_max_abs_err_V']=float(np.max(np.abs(np.array(yw)-ys)))
ax[3].plot(d[:,0],d[:,1],'k',lw=2); ax[3].plot(vin,yw,'C1--',lw=1); ax[3].set_title('CA3080 VCA transfer'); ax[3].set_xlabel('input (V)')
# THD of the VCA at 5 V peak
sr=FS; tt=np.arange(sr)/sr; v=OTAVCA(sr); yy=np.array([v.tick(5*np.sin(2*np.pi*200*q),8.0) for q in tt])
Y=np.abs(np.fft.rfft(yy*np.hanning(sr))); k0=200; thd=np.sqrt(sum(Y[k0*k]**2 for k in range(2,8)))/Y[k0]
res['vca_thd_5Vpk_pct']=float(100*thd)
plt.tight_layout(); plt.savefig('wdf_validation.png',dpi=150); plt.savefig('wdf_validation.pdf')
json.dump(res,open('wdf_validation.json','w'),indent=1,default=float); print(json.dumps(res,indent=1,default=float))
