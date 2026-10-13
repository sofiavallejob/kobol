import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from wdf import *
plt.rcParams.update({'font.size':8.5,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'grid.linewidth':.5})
C_ML='#2F6FB3'; C_WDF='#D9822B'; C_BEH='#7A7F85'; C_SP='#111111'
out='/home/claude/figs'
# ---------------- Fig: system diagram ----------------
def box(ax,x,y,w,h,txt,col,sub=None):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.08',fc=col,ec='none',alpha=.92))
    ax.text(x+w/2,y+h/2+(0.13 if sub else 0),txt,ha='center',va='center',color='white',fontsize=9,fontweight='bold')
    if sub: ax.text(x+w/2,y+h/2-0.2,sub,ha='center',va='center',color='white',fontsize=6.8)
def arr(ax,a,b,c='#444',ls='-',rad=0.0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,color=c,lw=1.1,ls=ls,shrinkA=2,shrinkB=2,connectionstyle=f'arc3,rad={rad}'))
fig,ax=plt.subplots(figsize=(7.2,3.2)); ax.set_xlim(0,14.4); ax.set_ylim(-0.2,6.9); ax.axis('off'); ax.grid(False)
box(ax,0.2,4.3,2.5,1.0,'LFO',C_WDF,'CA3080 integrator')
box(ax,0.2,2.9,2.5,1.0,'Voltage proc.',C_WDF,'WDF lag')
box(ax,0.2,1.6,2.5,0.8,'Keyboard CV',C_BEH)
box(ax,0.2,0.0,2.5,1.0,'Noise',C_WDF,'MM5837 + WDF')
box(ax,3.8,4.3,2.6,1.0,'VCO 1',C_ML,'grey-box, learned')
box(ax,3.8,2.9,2.6,1.0,'VCO 2',C_ML,'same model')
box(ax,7.4,4.3,2.0,1.0,'VCA 1',C_WDF,'CA3080')
box(ax,7.4,2.9,2.0,1.0,'VCA 2',C_WDF,'CA3080')
box(ax,10.4,3.6,2.1,1.0,'Output VCA',C_WDF,'CA3080')
box(ax,7.4,1.5,2.0,0.9,'ADS',C_BEH,'behavioural')
arr(ax,(1.45,2.4),(1.45,2.9)); arr(ax,(2.7,3.4),(3.8,4.6)); arr(ax,(2.7,3.4),(3.8,3.4))
arr(ax,(5.1,4.3),(5.1,3.9),C_ML,'--'); ax.text(5.2,4.06,'sync',fontsize=7,color=C_ML)
arr(ax,(6.4,4.8),(7.4,4.8)); arr(ax,(6.4,3.4),(7.4,3.4)); arr(ax,(9.4,4.8),(10.4,4.3)); arr(ax,(9.4,3.4),(10.4,3.9))
arr(ax,(2.7,0.5),(10.9,3.6),'#444','-',0.18)
arr(ax,(2.7,4.95),(3.8,4.95),'#444',':')
arr(ax,(1.45,5.3),(8.4,5.3),'#444',':',-0.25); ax.text(5.1,6.5,'LFO modulation (pitch, VCAs)',fontsize=7,color='#444',ha='center')
arr(ax,(8.4,2.4),(8.4,2.9),'#444',':'); arr(ax,(12.5,4.1),(13.3,4.1)); ax.text(13.4,4.1,'out',va='center',fontsize=8)
for i,(c,l) in enumerate([(C_ML,'learned from 437 recordings'),(C_WDF,'white-box: schematic + ngspice'),(C_BEH,'behavioural')]):
    ax.add_patch(FancyBboxPatch((10.4,2.2-0.55*i),0.35,0.3,boxstyle='round,pad=0.01',fc=c,ec='none')); ax.text(10.9,2.35-0.55*i,l,va='center',fontsize=7.5)
plt.tight_layout(); plt.savefig(f'{out}/fig_system.png',dpi=200); plt.savefig(f'{out}/fig_system.pdf'); plt.close()
# ---------------- Fig: grey-box VCO architecture ----------------
fig,ax=plt.subplots(figsize=(7.2,2.7)); ax.set_xlim(0,14.4); ax.set_ylim(0,5.2); ax.axis('off'); ax.grid(False)
box(ax,0.1,3.3,1.7,1.2,'CVs',C_BEH,'v_f, v_w')
box(ax,2.4,3.3,2.0,1.2,'MLP',C_ML,'2-64-64-64-8')
box(ax,5.0,3.3,2.3,1.2,'Stage params',C_ML,'gains, thresholds, f_c')
C_CF='#3E8E8A'; box(ax,7.9,3.6,1.6,0.8,'Saw core',C_CF); box(ax,7.9,2.6,1.6,0.8,'Fold',C_CF); box(ax,7.9,1.6,1.6,0.8,'Comp A/B',C_CF)
box(ax,10.1,2.4,1.8,1.4,'Σ → H(f)',C_CF,'LP, LP2, chain HP')
box(ax,12.3,2.4,2.0,1.4,'Harmonics',C_ML,'256, band-limited')
box(ax,2.4,0.4,4.9,1.2,'Physical pitch law',C_ML,'exp. converter, detune, 23 CV offsets')
arr(ax,(1.8,3.9),(2.4,3.9)); arr(ax,(4.4,3.9),(5.0,3.9)); arr(ax,(7.3,3.9),(7.9,4.0)); arr(ax,(7.3,3.7),(7.9,3.0)); arr(ax,(7.3,3.5),(7.9,2.0))
arr(ax,(9.5,4.0),(10.1,3.4)); arr(ax,(9.5,3.0),(10.1,3.1)); arr(ax,(9.5,2.0),(10.1,2.8)); arr(ax,(11.8,3.1),(12.3,3.1))
arr(ax,(0.95,3.3),(2.4,1.0)); arr(ax,(7.3,1.0),(13.25,2.4))
ax.text(8.6,0.75,'f drives the phase accumulator and H(f)',fontsize=7.5,color='#333')
for i,(c,l) in enumerate([(C_ML,'learned'),(C_CF,'closed-form circuit stages')]):
    ax.add_patch(FancyBboxPatch((0.15+3.0*i,4.75),0.3,0.28,boxstyle='round,pad=0.01',fc=c,ec='none')); ax.text(0.6+3.0*i,4.89,l,va='center',fontsize=7.5)
plt.tight_layout(); plt.savefig(f'{out}/fig_vco_arch.png',dpi=200); plt.savefig(f'{out}/fig_vco_arch.pdf'); plt.close()
import sys
if 'nowdf' in sys.argv: raise SystemExit
# ---------------- Fig: white-box validation ----------------
v1=json.load(open('wdf_validation.json')); v2=json.load(open('validation2.json'))
fig,axs=plt.subplots(2,3,figsize=(7.2,4.4)); ax=axs.ravel()
d=np.loadtxt('sp/noise.dat'); FS=48000
n=NoiseColour(FS); N=1<<16; h=np.array([n.tick(1.0 if i==0 else 0.0) for i in range(N)]); H=np.fft.rfft(h); f=np.fft.rfftfreq(N,1/FS)
ax[0].semilogx(d[:,0],d[:,1],color=C_SP,lw=2.2,label='ngspice'); ax[0].semilogx(f[1:],20*np.log10(np.abs(H[1:])),color=C_WDF,lw=1.2,ls='--',label='WDF')
ax[0].set_xlim(2,24000); ax[0].set_ylim(-12,2); ax[0].set_title('(a) Noise output network'); ax[0].set_xlabel('frequency (Hz)'); ax[0].set_ylabel('dB'); ax[0].legend(frameon=False,fontsize=7)
z=np.load('noise_spec.npz'); sel=z['f']>20
ax[1].semilogx(z['f'][sel],10*np.log10(z['Pw'][sel]/z['Pw'][sel][50:200].mean()),color=C_BEH,lw=.9,label='MM5837 (white)')
ax[1].semilogx(z['f'][sel],10*np.log10(z['Pp'][sel]/z['Pp'][sel][50:200].mean()),color=C_WDF,lw=.9,label='pink')
ff=np.array([50,20000]); ax[1].semilogx(ff,-3*np.log2(ff/ z['f'][sel][120]),'k:',lw=1,label='−3 dB/oct')
ax[1].set_ylim(-30,8); ax[1].set_title('(b) Noise spectra'); ax[1].set_xlabel('frequency (Hz)'); ax[1].legend(frameon=False,fontsize=7)
T=1.2; t=np.arange(int(T*FS))/FS
for x,c in zip((0.3,0.6,0.9),('#E3A15C','#D9822B','#9A4F0E')):
    dd=np.loadtxt(f'sp/lag{x}.dat'); L=Lag(FS); L.set(x); y=np.array([L.tick(5.0 if tt>0.01 else 0.0) for tt in t])
    ax[2].plot(dd[:,0],dd[:,1],color=C_SP,lw=2.2); ax[2].plot(t,y,color=c,lw=1.2,ls='--')
ax[2].set_title('(c) Voltage processor lag'); ax[2].set_xlabel('time (s)'); ax[2].set_ylabel('V')
L=v2['otalfo']; kn=np.linspace(0,1,50); pl=P['otalfo']
ax[3].semilogy(kn,pl['fmin']*(pl['fmax']/pl['fmin'])**kn,color='#BBB',lw=3,label='0.01–100 Hz target')
ax[3].semilogy([r['knob'] for r in L],[r['ngspice_Hz'] for r in L],'o',color=C_SP,ms=6,mfc='none',label='ngspice')
ax[3].semilogy([r['knob'] for r in L],[r['wdf_Hz'] for r in L],'x',color=C_WDF,ms=6,label='WDF')
ax[3].set_title('(d) LFO rate'); ax[3].set_xlabel('rate control'); ax[3].set_ylabel('Hz'); ax[3].legend(frameon=False,fontsize=7)
w=np.load('lfo_wave.npz'); tt=np.arange(len(w['w0']))/FS; k=int(0.6*FS)
ax[4].plot(tt[:k],w['w0'][:k,0],color=C_WDF,lw=1.3,label='triangle'); ax[4].plot(tt[:k],w['w1'][:k,0],color=C_ML,lw=1.1,label='triangle, symmetry trim')
ax[4].plot(tt[:k],0.5*w['w0'][:k,1]-1.6,color=C_BEH,lw=1,label='square (scaled)'); ax[4].set_ylim(-2.3,1.3)
ax[4].set_title('(e) LFO waveforms'); ax[4].set_xlabel('time (s)'); ax[4].legend(frameon=False,fontsize=6.5,loc='upper right')
d=np.loadtxt('sp/ota.dat'); pv=P['vca']; vin=np.linspace(-10,10,201); vv=OTAVCA(FS); yw=[]
for x in vin:
    for _ in range(40): y=vv.tick(x,8.0)
    yw.append(y)
ax[5].plot(d[:,0],d[:,1],color=C_SP,lw=2.2,label='ngspice'); ax[5].plot(vin,yw,color=C_WDF,lw=1.2,ls='--',label='WDF')
ax[5].set_title('(f) CA3080 VCA transfer'); ax[5].set_xlabel('input (V)'); ax[5].set_ylabel('output (V)'); ax[5].legend(frameon=False,fontsize=7)
ins=ax[5].inset_axes([0.6,0.1,0.36,0.32]); lv=np.array([1,2,5,8,10]); thd=[]
for a in lv:
    vv=OTAVCA(FS); yy=np.array([vv.tick(a*np.sin(2*np.pi*200*q/FS),8.0) for q in range(FS)]); Y=np.abs(np.fft.rfft(yy*np.hanning(FS)))
    thd.append(100*np.sqrt(sum(Y[200*k]**2 for k in range(2,8)))/Y[200])
ins.plot(lv,thd,'o-',color=C_WDF,ms=3,lw=1); ins.set_xlabel('V peak',fontsize=6); ins.set_ylabel('THD %',fontsize=6); ins.tick_params(labelsize=6); ins.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{out}/fig_wdf.png',dpi=200); plt.savefig(f'{out}/fig_wdf.pdf'); plt.close()
print('thd',np.round(thd,3))
