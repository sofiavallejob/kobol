import numpy as np, torch, json, base64, sys
exec(open('exp.py').read().split("\nif __name__=='__main__'")[0])
c,glob,m=run('final',np.arange(len(W)),int(sys.argv[1]))
torch.save(m.state_dict(),'final_full.pt'); print(glob,'train SNR med %.2f'%np.median([hsnr(c[j],C[j],Mk[j]) for j in range(len(W))]))
sd=m.state_dict(); ks=sorted({int(q.split('.')[1]) for q in sd if q.startswith('net.')}); ls=[]; flat=[]
for i in ks:
    Wm=sd[f'net.{i}.weight'].numpy(); b=sd[f'net.{i}.bias'].numpy(); ls.append({'o':Wm.shape[0],'i':Wm.shape[1]}); flat+=[Wm.ravel(),b]
flat=np.concatenate(flat).astype(np.float32)
import os; M=json.load(open('kobol_vco_web_bundle.json' if os.path.exists('kobol_vco_web_bundle.json') else 'model3.json'))
M['gb']={'layers':ls,'w':base64.b64encode(flat.tobytes()).decode(),'fh':glob['fh'],'fc2':glob['fc2'],'slew':[x*1e-6 for x in glob['slew_us']],
         'tri':[[float(x.real),float(x.imag)] for x in TRI0.numpy()]}
json.dump(M,open('model4.json','w'),separators=(',',':'))
x=torch.tensor([[(5.5-6.27)/3.63,(6.3-4.5)/4.5]],dtype=torch.float32)
with torch.no_grad(): cc=m(x,torch.tensor([300.0])).numpy()[0]
print('ref300',[(round(float(cc[q].real),5),round(float(cc[q].imag),5)) for q in [0,1,2,40]])
