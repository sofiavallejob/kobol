import soundfile as sf, numpy as np, glob
D={}
for p in glob.glob('kobol/audios/*.wav'):
    a,b=map(int,p.split('/')[-1][:-4].split('_'))
    x,sr=sf.read(p); D[(a,b)]=x
