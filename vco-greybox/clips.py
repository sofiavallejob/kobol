import numpy as np, soundfile as sf, subprocess, pandas as pd, base64, json
exec(open('load.py').read())
e=pd.read_csv('f0b.csv'); SR=44100; out={}
for _,r in e.iterrows():
    w,n=int(r.w),int(r.n); x=D[(w,n)][:,0]; m=np.abs(x).max(); i=np.argmax(np.abs(x)>0.05*m)+3000
    seg=x[i:i+int(1.0*SR)]
    sf.write('clips/t.wav',seg,SR)
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i','clips/t.wav','-ac','1','-b:a','64k','-cutoff','19500','clips/t.mp3'],check=True)
    out[f'{w}_{n}']=base64.b64encode(open('clips/t.mp3','rb').read()).decode()
s=json.dumps(out,separators=(',',':')); open('clips.json','w').write(s); print(len(s)/1e6,'MB')
