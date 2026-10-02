"""Two panels matching what the permutation null actually supports.

A  Allocation and handling: what each player was handed, and how often he lost it,
   against the rate his situations imply. Both quantities survive the null.
B  The coupling: c_in against Dc = c_in - c_out, with permutation replicates behind.
   The null reproduces the observed pattern, so Dc cannot be read as relief ability.
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, './src')
import perm
from perm import load, one_permutation
H = lambda g: g[(g.terminal == 'pass') & g.dc.notna()]
perm.STATS = {'c_in_hand': lambda g: H(g).c_in.mean(), 'dc': lambda g: H(g).dc.mean()}

OUT = './out'
BLUE, ORANGE, INK, MUTED, GRIDC, SURF = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e6e5df', '#fcfcfb'
SHORT = lambda s: 'Grimaldo' if 'García' in s else ('Hincapié' if 'Reyna' in s else s.split()[-1])

def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#c9c8c2')
    ax.grid(True, color=GRIDC, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)

lev = load()
obs = []
for pid, g in lev.groupby('player_id'):
    h = g[g.terminal == 'pass'].dropna(subset=['c_out'])
    obs.append(dict(player=SHORT(g.player.iloc[0]), pos=g.pg.iloc[0], n=len(g), n_hand=len(h),
                    c_in=g.c_in.mean(), loss=g.lost.mean(), dc=h.dc.mean(), c_in_hand=h.c_in.mean(),
                    movable=np.mean([len(s) for (m, p), s in g.groupby(['match_id', 'pg'])
                                     if lev[(lev.match_id == m) & (lev.pg == p)].player_id.nunique() > 1])))
o = pd.DataFrame(obs)
o['ret_above'] = o.c_in - o.loss

rng = np.random.default_rng(3)
null_pts = []
for _ in range(60):
    pv = one_permutation(lev, rng)
    null_pts.append(np.column_stack([pv['c_in_hand'], pv['dc']]))
null_pts = np.vstack(null_pts)

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), dpi=220)
fig.patch.set_facecolor(SURF)

# ---------------- A: allocation and handling ----------------
ax = axes[0]; style(ax)
lo, hi = o.c_in.min() - .03, o.c_in.max() + .03
ax.plot([lo, hi], [lo, hi], color='#a8a79f', lw=1.1, ls=(0, (4, 3)), zorder=1)
ax.scatter(o.c_in, o.loss, s=30 + o.n / 16, c=np.where(o.ret_above >= 0, BLUE, ORANGE),
           edgecolor='#4a4943', linewidth=0.7, zorder=3)
OFF = {'Boniface': (9, -3), 'Schick': (-36, 4), 'Frimpong': (9, -9), 'Adli': (9, -2),
       'Wirtz': (9, -3), 'Hofmann': (9, 4), 'Grimaldo': (9, -10), 'Tella': (-30, 5)}
# crowded defensive/midfield cluster: labels placed in the empty lower-right, with leader lines
CALL = {'Kossonou': (0.215, 0.135), 'Stanišić': (0.225, 0.110), 'Xhaka': (0.235, 0.085),
        'Palacios': (0.240, 0.060), 'Andrich': (0.225, 0.035), 'Hincapié': (0.205, 0.012),
        'Tapsoba': (0.150, 0.010), 'Tah': (0.095, 0.045)}
for _, r in o.iterrows():
    if r.player in CALL:
        tx, ty = CALL[r.player]
        ax.annotate(r.player, (r.c_in, r.loss), xytext=(tx, ty), textcoords='data', fontsize=7.8,
                    color=INK, zorder=4, va='center',
                    arrowprops=dict(arrowstyle='-', color='#a8a79f', lw=0.6, shrinkA=1, shrinkB=4))
    else:
        ax.annotate(r.player, (r.c_in, r.loss), xytext=OFF.get(r.player, (9, -3)),
                    textcoords='offset points', fontsize=7.8, color=INK, zorder=4)
ax.set_xlabel('difficulty he was handed,  $\\bar{c}_{\\mathrm{in}}$', fontsize=9, color=INK)
ax.set_ylabel('share of spells he lost', fontsize=9, color=INK)
ax.set_title('A  What he was handed, and how he handled it', loc='left', fontsize=10.5, color=INK, pad=42)
ax.text(0, 1.015, 'Dashed line: loss rate the yardstick implies. Area: spells. Orange: lost more\n'
                  'often than implied. Between-player dispersion in reception difficulty and in\n'
                  'retention residuals both exceeds the permutation benchmark.',
        transform=ax.transAxes, fontsize=7.8, color=MUTED, linespacing=1.5, va='bottom')
ax.set_xlim(lo, hi); ax.set_ylim(-0.005, o.loss.max() + .07)

# ---------------- B: the coupling ----------------
ax = axes[1]; style(ax)
ax.scatter(null_pts[:, 0], null_pts[:, 1], s=9, color='#c9c8c2', alpha=0.45,
           linewidth=0, zorder=1, label='permutation replicates')
ax.scatter(o.c_in_hand, o.dc, s=30 + o.n_hand / 16, color=BLUE, edgecolor='#4a4943',
           linewidth=0.7, zorder=3, label='observed players')
ax.axhline(0, color='#a8a79f', lw=0.8, ls=(0, (4, 3)))
ax.set_xlabel('difficulty he was handed, completed hand-offs only,  $\\bar{c}_{\\mathrm{in}}$', fontsize=9, color=INK)
ax.set_ylabel('raw difference,  $\\Delta c=\\bar{c}_{\\mathrm{in}}-\\bar{c}_{\\mathrm{out}}$',
              fontsize=9, color=INK)
ax.set_title('B  Why a raw difficulty difference needs caution', loc='left', fontsize=10.5, color=INK, pad=42)
ax.text(0, 1.015, 'Same spells on both axes. Grey: labels shuffled within match and position group.\n'
                  'Observed r = 0.982; null mean 0.991, 95% permutation interval [0.988, 0.994].',
        transform=ax.transAxes, fontsize=7.8, color=MUTED, linespacing=1.5, va='bottom')
ax.legend(frameon=False, fontsize=7.8, loc='upper left', labelcolor=INK)

fig.tight_layout()
fig.savefig(f'{OUT}/fig_final.pdf', bbox_inches='tight')
fig.savefig(f'{OUT}/fig_final.png', bbox_inches='tight')
print(o[['player', 'pos', 'n', 'n_hand', 'c_in', 'c_in_hand', 'loss', 'ret_above', 'dc']].round(4)
      .sort_values('ret_above', ascending=False).to_string(index=False))
