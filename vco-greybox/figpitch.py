import numpy as np, torch, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('evalfw.py').read().split("if __name__")[0])
plt.rcParams.update({'font.size':9,'font.family':'serif','axes.spines.top':False,'axes.spines.right':False})
def cyc(c,N=512,f=None):
    ph=np.arange(N)/N; kk=np.arange(1,K+1); y=np.zeros(N)
    for q in range(K):
        if c[q]!=0: y+=np.abs(c[q])*np.cos(2*np.pi*kk[q]*ph+np.angle(c[q]))
    return y
# Fig 3: pitch deviation and CV-step offsets
P=np.load('pitch_phys_full.npy'); IA,IB=-0.7602117,1.5008941
fig,ax=plt.subplots(1,2,figsize=(7.2,2.4))
for w,col in [(0,'C0'),(9,'C1'),(18,'C3')]:
    m=(W==w)&(Nn>=23)&(Nn<=41); ax[0].plot(CVF[m],1200*(np.log2(F0[m])-(IA+IB*CVF[m])),'o-',color=col,ms=3,lw=.8,label=['triangle','saw + square +','pulse +'][[0,9,18].index(w)])
ax[0].set_xlabel('Frequency CV (V)'); ax[0].set_ylabel('cents from 1.5 oct/V line'); ax[0].legend(fontsize=7,frameon=False)
ax[1].bar(np.arange(23)+21,1200*P[5:],color='0.4'); ax[1].set_ylim(-20,20); ax[1].set_xlabel('CV step (file index)'); ax[1].set_ylabel('offset (cents)')
ax[1].set_title('Per-step CV offsets (limits clipped)',fontsize=8)
plt.tight_layout(); plt.savefig('figs/fig_pitch.pdf'); plt.savefig('figs/fig_pitch.png',dpi=150)
print('ok')
