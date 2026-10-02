"""Same-sample check for panel B: c_in and Dc both computed on completed, linked hand-offs only.
Same within-(match, position group) permutation as perm.py."""
import sys, json, numpy as np
sys.path.insert(0, './src')
from perm import load, one_permutation
import perm
OUT = './out'
H = lambda g: g[(g.terminal == 'pass') & g.dc.notna()]
perm.STATS = {'c_in_hand': lambda g: H(g).c_in.mean(), 'dc': lambda g: H(g).dc.mean(),
              'c_out': lambda g: H(g).c_out.mean()}
lev = load(); rng = np.random.default_rng(0)
obs = {k: np.array([f(g) for _, g in lev.groupby('player_id')]) for k, f in perm.STATS.items()}
oc = np.corrcoef(obs['c_in_hand'], obs['dc'])[0, 1]
nc, nv = [], {k: [] for k in perm.STATS}
for _ in range(500):
    pv = one_permutation(lev, rng)
    nc.append(np.corrcoef(pv['c_in_hand'], pv['dc'])[0, 1])
    for k in perm.STATS: nv[k].append(np.var(pv[k], ddof=1))
nc = np.array(nc)
res = dict(obs=oc, null_mean=nc.mean(), lo=np.percentile(nc, 2.5), hi=np.percentile(nc, 97.5),
           pctile=np.mean(nc < oc), n_hand=int(sum(len(H(g)) for _, g in lev.groupby('player_id'))))
for k in perm.STATS:
    vo = np.var(obs[k], ddof=1); vn = np.mean(nv[k])
    res[k+'_excess_sd'] = float(np.sqrt(max(0, vo - vn))); res[k+'_pctile'] = float(np.mean(np.array(nv[k]) < vo))
res = {k: float(v) for k, v in res.items()}
print(json.dumps(res, indent=1)); json.dump(res, open(f'{OUT}/perm_hand.json', 'w'), indent=1)
