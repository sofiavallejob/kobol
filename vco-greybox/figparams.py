import numpy as np, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('exp.py').read().split("\nif __name__=='__main__'")[0])
plt.rcParams.update({'font.size':8,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False})
m=GB(VARIANTS['final']); m.load_state_dict(torch.load('final_full.pt'))
cw=np.linspace(0,9,300); fig,ax=plt.subplots(1,3,figsize=(7.2,2.3),sharex=True)
for cvf,ls in [(4.62,':'),(6.27,'-'),(8.25,'--')]:
    x=torch.tensor(np.stack([np.full_like(cw,(cvf-6.27)/3.63),(cw-4.5)/4.5],1),dtype=torch.float32)
    with torch.no_grad(): o=m.net(x).numpy()
    At,As,A1,t1,A2,t2,pt,lfc=o.T; d1=(1-np.tanh(t1))/2; d2=(1-np.tanh(t2))/2
    # effective rms contribution of each stage (gain x stage rms): tri 1/sqrt3, saw 1/sqrt3, comparator 2*sqrt(d(1-d))
    ax[0].plot(cw,np.abs(At)/np.sqrt(3),'C0'+ls,lw=1); ax[0].plot(cw,np.abs(As)/np.sqrt(3),'C1'+ls,lw=1)
    ax[0].plot(cw,np.abs(A1)*2*np.sqrt(d1*(1-d1)),'C2'+ls,lw=1); ax[0].plot(cw,np.abs(A2)*2*np.sqrt(d2*(1-d2)),'C3'+ls,lw=1)
    ax[1].plot(cw,d1,'C2'+ls,lw=1); ax[1].plot(cw,np.where(np.abs(A2)*2*np.sqrt(d2*(1-d2))>0.02,d2,np.nan),'C3'+ls,lw=1)
    ax[2].plot(cw,np.exp(lfc)/1000,'k'+ls,lw=1,label=f'{cvf:.2f} V')
ax[0].set_title('Stage contribution (rms)'); ax[1].set_title('Comparator duty (active stages)'); ax[2].set_title('Output lowpass (kHz)')
ax[0].legend(['triangle','saw','comp. A','comp. B'],fontsize=7,frameon=False); ax[2].legend(fontsize=7,frameon=False,title='freq. CV',title_fontsize=7)
for a in ax: a.set_xlabel('Waveform CV (V)')
plt.tight_layout(); plt.savefig('figs/fig_waveshaper_params.pdf'); plt.savefig('figs/fig_waveshaper_params.png',dpi=150); print('ok')
