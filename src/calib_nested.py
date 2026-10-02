"""Fully nested calibration check: for each outer fold of matches, the GBM and the isotonic map
are both built from the training matches only (isotonic fitted on inner out-of-fold scores);
the held-out matches are touched only for evaluation."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, './src')
from roles import frame_for, TEAM
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
OUT = './out'
P = dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=80, l2_regularization=1.0, random_state=0)
df = pd.read_csv(f'{OUT}/handoff.csv', low_memory=False)
opp = df[(df.team != TEAM) & df.resolved]
X = frame_for(opp, 'in').values; y = opp.lost.astype(int).values; g = opp.match_id.values
raw = np.full(len(y), np.nan); cal = np.full(len(y), np.nan)
for tr, te in GroupKFold(5).split(X, y, g):
    inner = np.full(len(tr), np.nan)
    for a, b in GroupKFold(4).split(X[tr], y[tr], g[tr]):
        inner[b] = HGB(**P).fit(X[tr][a], y[tr][a]).predict_proba(X[tr][b])[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip').fit(inner, y[tr])
    raw[te] = HGB(**P).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    cal[te] = iso.predict(raw[te])
q = pd.qcut(cal, 10, labels=False, duplicates='drop')
t = pd.DataFrame({'p': cal, 'o': y, 'q': q}).groupby('q').agg(n=('o', 'size'), pred=('p', 'mean'), obs=('o', 'mean'))
r = dict(auc_raw=float(roc_auc_score(y, raw)), max_decile_gap=float((t.pred - t.obs).abs().max()),
         mean_pred=float(cal.mean()), mean_obs=float(y.mean()))
print(t.round(3).to_string()); print(json.dumps(r, indent=1)); json.dump(r, open(f'{OUT}/calib_nested.json', 'w'), indent=1)
