"""Score the in- and out-situation of every on-ball spell with one yardstick, then
describe each player's role by three quantities that are never collapsed into one score:

  (1) what he was given      distribution of c_in
  (2) how he handled it      loss rate, and loss rate relative to what c_in implies
  (3) what he handed on      among completed passes, Dc = c_in - c_out  and  Dv = v_out - v_in

The yardstick is fitted on the opponents' receptions only, so the benchmark is not the
team being scouted. Dc and Dv are reported only for completed passes and are always
accompanied by the loss rate, because a player who sheds difficulty by losing the ball
must not score well.
"""
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression

OUT = './out'
TEAM = 'Bayer Leverkusen'

BASE = ['x','y','under_pressure','pass_len','pass_ang','pass_prog','pass_under_pressure',
        'minute','height_high','height_ground','is_open_play','is_throw','is_corner','is_fk','is_gk']
F360 = ['sar','n_opp5','n_opp10','opp_between','free_tm_ahead']
FEATS = BASE + F360

def frame_for(df, side):
    """Build the feature matrix for the `in` or the `out` situation of each spell."""
    pre = 'in_' if side == 'in' else 'out_'
    ppre = '' if side == 'in' else 'outpass_'
    f = pd.DataFrame(index=df.index)
    f['x'] = df[pre+'x']; f['y'] = df[pre+'y']
    f['under_pressure'] = df[pre+'under_pressure'].astype(float)
    for c in ('sar','n_opp5','n_opp10','opp_between','free_tm_ahead'): f[c] = df[pre+c]
    f['pass_len'] = df[ppre+'pass_len']; f['pass_ang'] = df[ppre+'pass_ang']
    f['pass_prog'] = df[ppre+'pass_prog']
    f['pass_under_pressure'] = df[ppre+'pass_under_pressure'].astype(float)
    h = df[ppre+'pass_height']
    f['height_high'] = (h == 'High Pass').astype(float); f['height_ground'] = (h == 'Ground Pass').astype(float)
    p = df[pre+'play_pattern']
    f['is_open_play'] = (p == 'Regular Play').astype(float); f['is_throw'] = (p == 'From Throw In').astype(float)
    f['is_corner'] = (p == 'From Corner').astype(float); f['is_fk'] = (p == 'From Free Kick').astype(float)
    f['is_gk'] = (p == 'From Goal Kick').astype(float)
    f['minute'] = df['minute']
    return f[FEATS].astype(float)

def fit_yardstick(df):
    """Fit P(loss | situation) on the OPPONENTS' spells only; no player identity, no position."""
    opp = df[(df.team != TEAM) & df.resolved]
    X = frame_for(opp, 'in').values; y = opp.lost.astype(int).values
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, opp.match_id.values):
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
                                           min_samples_leaf=80, l2_regularization=1.0, random_state=0)
        m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof); br = brier_score_loss(y, oof)
    base = brier_score_loss(y, np.full_like(oof, y.mean()))
    print(f'yardstick (opponents only): n={len(y):,} loss={y.mean():.3f} '
          f'OOF AUC={auc:.4f} Brier={br:.4f} (base {base:.4f})')
    q = pd.qcut(oof, 10, labels=False, duplicates='drop')
    print(pd.DataFrame({'p': oof, 'o': y, 'q': q}).groupby('q')
          .agg(n=('o','size'), pred=('p','mean'), obs=('o','mean')).round(3).to_string())
    full = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
                                          min_samples_leaf=80, l2_regularization=1.0, random_state=0)
    full.fit(X, y)
    iso = IsotonicRegression(out_of_bounds='clip').fit(oof, y)   # recalibrate on the OOF predictions
    cal = iso.predict(oof)
    print(f'after isotonic recalibration: Brier={brier_score_loss(y, cal):.4f}')
    q = pd.qcut(cal, 10, labels=False, duplicates='drop')
    print(pd.DataFrame({'p': cal, 'o': y, 'q': q}).groupby('q')
          .agg(n=('o','size'), pred=('p','mean'), obs=('o','mean')).round(3).to_string())
    return full, iso, oof, opp.index, dict(auc=float(auc), brier=float(br), base=float(base), n=int(len(y)))

def score(df, model, iso, oof, oof_idx):
    df = df.copy()
    raw = model.predict_proba(frame_for(df, 'in').values)[:, 1]
    raw[df.index.get_indexer(oof_idx)] = oof            # opponents keep their out-of-fold value
    df['c_in'] = iso.predict(raw)
    # c_out is taken from the spell it hands into, so the two are identical by construction
    key = df.set_index(['match_id', 'ev_index']).c_in
    m = df.out_exists.fillna(False).astype(bool)
    idx = pd.MultiIndex.from_arrays([df.loc[m, 'match_id'], df.loc[m, 'next_ev_index']])
    df['c_out'] = np.nan
    df.loc[m, 'c_out'] = key.reindex(idx).values
    df['dc'] = df.c_in - df.c_out                       # >0 : the next man got an easier ball
    df['dv'] = df.out_xt_at - df.in_xt_at               # >0 : the ball is closer to scoring
    df['prog'] = df.out_x - df.in_x
    return df

def consistency(df):
    m = df.out_exists.fillna(False).astype(bool)
    miss = df.loc[m, 'c_out'].isna().sum()
    print(f'hand-offs linked: {m.sum():,}; c_out resolved for {m.sum()-miss:,} (c_out is the next '
          f'spell\'s c_in by construction)')

def profile(df, min_n=250):
    lev = df[(df.team == TEAM) & df.resolved]
    opp = df[(df.team != TEAM) & df.resolved]
    short = lambda s: 'Grimaldo' if 'García' in s else ('Hincapié' if 'Reyna' in s else s.split()[-1])
    rows = []
    for pid, g in lev.groupby('player_id'):
        if len(g) < min_n: continue
        pos = g.position.mode().iloc[0] if len(g.position.mode()) else None
        pg = pos_group(pos)
        ref = opp[opp.position.map(pos_group) == pg]
        h = g[g.terminal == 'pass'].dropna(subset=['c_out'])
        refh = ref[ref.terminal == 'pass'].dropna(subset=['c_out'])
        rows.append(dict(
            player=short(g.player.iloc[0]), pos=pg, n=len(g),
            c_in=g.c_in.mean(), c_in_ref=ref.c_in.mean() if len(ref) else np.nan,
            loss=g.lost.mean(), loss_exp=g.c_in.mean(),
            pass_share=(g.terminal == 'pass').mean(),
            dc=h.dc.mean(), dc_ref=refh.dc.mean() if len(refh) else np.nan,
            dv=h.dv.mean(), dv_ref=refh.dv.mean() if len(refh) else np.nan,
            prog=h.prog.mean(), c_out=h.c_out.mean(), n_hand=len(h)))
    p = pd.DataFrame(rows)
    p['skill'] = p.loss_exp - p.loss                    # >0 : loses it less often than the situation implies
    p['rel_c_in'] = p.c_in - p.c_in_ref                 # >0 : handed harder balls than the position norm
    p['rel_dc'] = p.dc - p.dc_ref
    p['rel_dv'] = p.dv - p.dv_ref
    return p.sort_values('rel_c_in', ascending=False)

def pos_group(p):
    if not isinstance(p, str): return None
    if 'Goalkeeper' in p: return 'GK'
    if 'Wing Back' in p: return 'WB'
    if 'Back' in p: return 'CB' if 'Center' in p else 'FB'
    if 'Attacking Midfield' in p or 'Wing' in p: return 'AM/W'
    if 'Midfield' in p: return 'CM'
    if 'Forward' in p or 'Striker' in p: return 'FW'
    return None

if __name__ == '__main__':
    df = pd.read_csv(f'{OUT}/handoff.csv', low_memory=False)
    model, iso, oof, oof_idx, meta = fit_yardstick(df)
    df = score(df, model, iso, oof, oof_idx)
    consistency(df)
    df.to_csv(f'{OUT}/handoff_scored.csv', index=False)
    json.dump(meta, open(f'{OUT}/yardstick_meta.json', 'w'))
    p = profile(df); p.to_csv(f'{OUT}/roles.csv', index=False)
    pd.set_option('display.width', 250)
    cols = ['player','pos','n','c_in','c_in_ref','rel_c_in','loss','skill','n_hand',
            'dc','dc_ref','rel_dc','dv','dv_ref','prog']
    print('\n=== 角色表 Leverkusen 2023/24 ===')
    print(p[cols].round(3).to_string(index=False))
