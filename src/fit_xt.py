"""Fit an Expected Threat (xT) grid (Singh 2018 formulation) from StatsBomb open events.

Uses completed passes and carries as moves, shots with StatsBomb xG for the scoring term.
Trained on PL 2015/16 + BL 2015/16 + BL 2023/24 (Leverkusen) events. Writes out/xt_grid.json.
"""
import json, glob, sys, numpy as np

NX, NY = 12, 8
W, H = 120.0, 80.0

def cell(loc):
    x, y = loc[0], loc[1]
    cx = min(NX - 1, max(0, int(x / W * NX)))
    cy = min(NY - 1, max(0, int(y / H * NY)))
    return cx, cy

def main(paths, out):
    shots = np.zeros((NX, NY)); xg_sum = np.zeros((NX, NY))
    moves = np.zeros((NX, NY)); trans = np.zeros((NX, NY, NX, NY))
    n_files = 0
    for p in paths:
        try:
            ev = json.load(open(p))
        except Exception as e:
            print('skip', p, e); continue
        n_files += 1
        for e in ev:
            t = e['type']['name']
            if 'location' not in e: continue
            if t == 'Shot':
                cx, cy = cell(e['location'])
                shots[cx, cy] += 1
                xg_sum[cx, cy] += e['shot'].get('statsbomb_xg', 0.0)
            elif t == 'Pass':
                a = cell(e['location'])
                moves[a] += 1  # all pass attempts count as moves; only completions transfer mass
                if 'outcome' in e['pass'] or 'end_location' not in e['pass']: continue
                b = cell(e['pass']['end_location'])
                trans[a[0], a[1], b[0], b[1]] += 1
            elif t in ('Dispossessed', 'Miscontrol'):
                moves[cell(e['location'])] += 1  # failed move, absorbs mass
            elif t == 'Carry':
                a = cell(e['location']); b = cell(e['carry']['end_location'])
                if a == b: continue
                moves[a] += 1; trans[a[0], a[1], b[0], b[1]] += 1
    total = shots + moves
    with np.errstate(invalid='ignore', divide='ignore'):
        p_shot = np.where(total > 0, shots / total, 0)
        p_move = np.where(total > 0, moves / total, 0)
        g = np.where(shots > 0, xg_sum / shots, 0)
        T = np.where(moves[:, :, None, None] > 0, trans / moves[:, :, None, None], 0)
    xt = np.zeros((NX, NY))
    for it in range(500):
        new = p_shot * g + p_move * np.einsum('abcd,cd->ab', T, xt)
        if np.abs(new - xt).max() < 1e-7: xt = new; break
        xt = new
    print(f'files={n_files} shots={int(shots.sum())} moves={int(moves.sum())} iters={it+1}')
    print('xT by x-column (row y=3):', np.round(xt[:, 3], 3))
    json.dump({'nx': NX, 'ny': NY, 'grid': xt.tolist()}, open(out, 'w'))

if __name__ == '__main__':
    paths = glob.glob('./data/xt_train/*.json')
    paths = [p for p in paths if not p.split('/')[-1].startswith('m_')]
    paths += glob.glob('./data/events/*.json')
    main(paths, './out/xt_grid.json')
