# Leave-one-out control-law parameters: for every measured (r, c), the hybrid law trained WITHOUT it.
import numpy as np, json
from law import targets, Hybrid
T = targets(); out = {}
for i, (r, c) in enumerate(T[:, :2].astype(int)):
    te = np.arange(len(T)) == i
    p = Hybrid(T[~te, 0], T[~te, 1], T[~te, 2:])(np.array([r]), np.array([c]))[0]
    out[f'{r}_{c}'] = [float(1000 * 2 ** p[0]), float(max(0, p[1])), float(p[2])]
    print(r, c, np.round(out[f'{r}_{c}'], 2), flush=True)
json.dump(out, open('../vcf-compare/loo_params.json', 'w'))
