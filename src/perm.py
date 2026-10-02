"""Permutation inference for the hand-off measures.

Null: within a (match, position group) cell, reassign spells among the teammates who
actually appeared in that cell, preserving each player's count. This keeps the opponent,
the match, the score state and the positional role fixed, and destroys only the link
between a particular spell and a particular player. It therefore tests whether situations
are allocated to identities within a role, not whether players differ in ability.

Reported quantities
  excess SD = sqrt( max(0, Var_obs - mean(Var_null)) ),
              averaging the null VARIANCES, not squaring a mean SD.
  The interval given for the null is a 95% permutation interval for the null statistic,
  not a confidence interval for the excess.
"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, './src')
from roles import pos_group, TEAM

OUT = './out'
MIN_N = 250

def load():
    d = pd.read_csv(f'{OUT}/handoff_scored.csv', low_memory=False)
    d['pg'] = d.position.map(pos_group)
    lev = d[(d.team == TEAM) & d.resolved & d.pg.notna() & (d.pg != 'GK')].copy()
    keep = lev.groupby('player_id').size().loc[lambda s: s >= MIN_N].index
    return lev[lev.player_id.isin(keep)].copy()

STATS = {
    'c_in':      lambda g: g.c_in.mean(),
    'loss':      lambda g: g.lost.mean(),
    'retention_above_exp': lambda g: g.c_in.mean() - g.lost.mean(),
    'dc':        lambda g: g.loc[g.terminal == 'pass', 'dc'].mean(),
    'prog':      lambda g: g.loc[g.terminal == 'pass', 'prog'].mean(),
    'dv':        lambda g: g.loc[g.terminal == 'pass', 'dv'].mean(),
}

def observed(lev):
    return {k: np.array([f(g) for _, g in lev.groupby('player_id')]) for k, f in STATS.items()}

def one_permutation(lev, rng):
    """Shuffle player labels inside each (match, position group) cell."""
    lab = lev.player_id.values.copy()
    for _, idx in lev.groupby(['match_id', 'pg']).indices.items():
        if len(idx) > 1:
            lab[idx] = rng.permutation(lab[idx])
    perm = lev.assign(_p=lab)
    out = {}
    for k, f in STATS.items():
        v = np.array([f(g) for _, g in perm.groupby('_p')])
        out[k] = v
    return out

def run(reps=500, seed=0):
    lev = load(); rng = np.random.default_rng(seed)
    obs = observed(lev)
    n_players = len(lev.player_id.unique())
    null_var = {k: [] for k in STATS}; null_sd = {k: [] for k in STATS}
    null_corr = []
    for _ in range(reps):
        pv = one_permutation(lev, rng)
        for k in STATS:
            v = pv[k][~np.isnan(pv[k])]
            null_var[k].append(np.var(v, ddof=1)); null_sd[k].append(np.std(v, ddof=1))
        a, b = pv['c_in'], pv['dc']
        m = ~(np.isnan(a) | np.isnan(b))
        null_corr.append(np.corrcoef(a[m], b[m])[0, 1])
    res = {}
    for k in STATS:
        o = obs[k][~np.isnan(obs[k])]
        vo = np.var(o, ddof=1); vn = np.mean(null_var[k])
        res[k] = dict(obs_sd=float(np.sqrt(vo)), null_sd_mean=float(np.mean(null_sd[k])),
                      null_sd_lo=float(np.percentile(null_sd[k], 2.5)),
                      null_sd_hi=float(np.percentile(null_sd[k], 97.5)),
                      excess_sd=float(np.sqrt(max(0.0, vo - vn))),
                      pctile=float(np.mean(np.array(null_sd[k]) < np.sqrt(vo))))
    a, b = obs['c_in'], obs['dc']; m = ~(np.isnan(a) | np.isnan(b))
    oc = float(np.corrcoef(a[m], b[m])[0, 1])
    res['_corr_c_in_dc'] = dict(obs=oc, null_mean=float(np.mean(null_corr)),
                                lo=float(np.percentile(null_corr, 2.5)),
                                hi=float(np.percentile(null_corr, 97.5)),
                                pctile=float(np.mean(np.array(null_corr) < oc)))
    res['_n_players'] = n_players; res['_n_spells'] = int(len(lev)); res['_reps'] = reps
    return res, lev

if __name__ == '__main__':
    res, lev = run()
    print(f"players={res['_n_players']}  spells={res['_n_spells']}  permutations={res['_reps']}")
    print(f"\n{'quantity':22s}{'obs SD':>9}{'null SD':>9}{'95% perm interval':>22}{'excess SD':>11}{'pctile':>8}")
    for k in STATS:
        r = res[k]
        print(f"{k:22s}{r['obs_sd']:9.4f}{r['null_sd_mean']:9.4f}"
              f"   [{r['null_sd_lo']:.4f},{r['null_sd_hi']:.4f}]{r['excess_sd']:11.4f}{r['pctile']:8.3f}")
    c = res['_corr_c_in_dc']
    print(f"\ncorr(c_in, dc): observed {c['obs']:.4f}; null mean {c['null_mean']:.4f}, "
          f"95% permutation interval [{c['lo']:.4f},{c['hi']:.4f}]; observed at pctile {c['pctile']:.3f}")
    json.dump(res, open(f'{OUT}/perm_results.json', 'w'), indent=1)
