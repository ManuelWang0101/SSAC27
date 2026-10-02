"""Hand-off table: what a player was given, what he did with it, and what the next
teammate then faced.

Unit of analysis: one on-ball spell. It starts when player i completes a reception and
ends when he passes, shoots, wins a foul, or loses the ball.

For every spell we record
  in  : the situation he was handed              (features at his reception)
  do  : what happened during the spell           (terminal type, carry, dribbles)
  out : the situation the next teammate received (features at THAT reception)

`out` exists only for spells that end in a completed pass to a teammate. Spells that end
in a loss, a shot, or a foul are kept with out = NaN and an explicit terminal label, so
the accounting covers every reception rather than only the successful hand-offs.
"""
import json, os, sys, math, numpy as np, pandas as pd

DATA = './data'
OUT = './out'
XT = json.load(open(f'{OUT}/xt_grid.json'))
NX, NY, GRID = XT['nx'], XT['ny'], np.array(XT['grid'])

def xt(loc):
    if not loc: return np.nan
    return GRID[min(NX-1, max(0, int(loc[0]/120*NX))), min(NY-1, max(0, int(loc[1]/80*NY)))]

def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

ONBALL = {'Pass','Carry','Shot','Dribble','Ball Receipt*','Miscontrol','Dispossessed',
          'Clearance','Interception','Ball Recovery','Goal Keeper','Foul Won'}
LOST_PASS = {'Incomplete','Out','Pass Offside','Unknown','Injury Clearance'}

def situation(e, frames):
    """Pre-decision features of a reception (identical definition for `in` and `out`)."""
    loc = e['location']
    f = frames.get(e['id'])
    d = dict(x=loc[0], y=loc[1], xt_at=xt(loc),
             under_pressure=bool(e.get('under_pressure', False)),
             play_pattern=e['play_pattern']['name'],
             sar=np.nan, n_opp5=np.nan, n_opp10=np.nan, opp_between=np.nan,
             free_tm_ahead=np.nan, has_frame=False)
    if f:
        ff = f['freeze_frame']; d['has_frame'] = True
        opp = [p['location'] for p in ff if not p['teammate']]
        tm = [p['location'] for p in ff if p['teammate'] and not p['actor']]
        if opp:
            dd = [dist(loc, o) for o in opp]
            d.update(sar=min(dd), n_opp5=sum(1 for x in dd if x <= 5),
                     n_opp10=sum(1 for x in dd if x <= 10),
                     opp_between=sum(1 for o in opp if o[0] > loc[0]))
        d['free_tm_ahead'] = sum(1 for t in tm if t[0] > loc[0]+2 and
                                 (not opp or min(dist(t, o) for o in opp) > 5))
    return d

def incoming(e, byid, team):
    out = dict(pass_len=np.nan, pass_ang=np.nan, pass_prog=np.nan,
               pass_height=None, pass_type=None, pass_under_pressure=False, passer_id=None)
    for rid in e.get('related_events', []):
        r = byid.get(rid)
        if r and r['type']['name'] == 'Pass' and r['team']['id'] == team:
            p = r['pass']
            out.update(pass_len=p.get('length'), pass_ang=p.get('angle'),
                       pass_height=p.get('height', {}).get('name'),
                       pass_type=p.get('type', {}).get('name'),
                       pass_under_pressure=bool(r.get('under_pressure', False)),
                       passer_id=r['player']['id'] if 'player' in r else None)
            if r.get('location'): out['pass_prog'] = e['location'][0] - r['location'][0]
            break
    return out

def spells_for_match(mid):
    ev = sorted(json.load(open(f'{DATA}/events/{mid}.json')), key=lambda e: e['index'])
    byid = {e['id']: e for e in ev}
    frames = {}
    p = f'{DATA}/three-sixty/{mid}.json'
    if os.path.exists(p): frames = {f['event_uuid']: f for f in json.load(open(p))}

    # index every completed reception so we can look the next one up
    recs = []
    for i, e in enumerate(ev):
        if e['type']['name'] != 'Ball Receipt*' or 'player' not in e: continue
        if 'outcome' in e.get('ball_receipt', {}): continue
        if not e.get('location'): continue
        recs.append((i, e))
    pos_in_list = {e['id']: k for k, (i, e) in enumerate(recs)}

    rows = []
    for k, (i, e) in enumerate(recs):
        team, pid = e['team']['id'], e['player']['id']
        row = dict(match_id=mid, ev_index=e['index'], team=e['team']['name'], team_id=team,
                   player_id=pid, player=e['player']['name'],
                   position=e.get('position', {}).get('name'), minute=e['minute'],
                   period=e['period'], possession=e.get('possession'))
        row.update({f'in_{a}': b for a, b in situation(e, frames).items()})
        row.update(incoming(e, byid, team))

        # --- walk his own on-ball spell ---
        terminal = 'none'; carry_gain = 0.0; carry_dist = 0.0; dribbles = 0
        shot_xg = 0.0; release_loc = None; n_act = 0
        j = i + 1
        while j < len(ev):
            n = ev[j]; t = n['type']['name']
            same = ('player' in n and n['player']['id'] == pid and n['team']['id'] == team)
            if not same:
                if t in ONBALL:
                    terminal = 'lost_other' if n['team']['id'] != team else 'ended_teammate'
                    break
                j += 1; continue
            n_act += 1
            if t == 'Carry':
                carry_gain += max(0.0, xt(n['carry']['end_location']) - xt(n['location']))
                carry_dist += dist(n['location'], n['carry']['end_location'])
            elif t == 'Dribble':
                if n['dribble'].get('outcome', {}).get('name') == 'Complete': dribbles += 1
                else: terminal = 'lost_dribble'; break
            elif t == 'Pass':
                release_loc = n.get('location')
                if n['pass'].get('outcome', {}).get('name') in LOST_PASS:
                    terminal = 'lost_pass'
                else:
                    terminal = 'pass'
                    row['pass_out_id'] = n['id']
                    row['pass_out_end'] = n['pass'].get('end_location')
                    row['recipient_id'] = n['pass'].get('recipient', {}).get('id')
                break
            elif t == 'Shot':
                shot_xg = n['shot'].get('statsbomb_xg', 0.0); terminal = 'shot'
                release_loc = n.get('location'); break
            elif t in ('Miscontrol', 'Dispossessed'):
                terminal = 'lost_' + t.lower(); break
            elif t == 'Foul Won':
                terminal = 'foul_won'; break
            elif t == 'Clearance':
                terminal = 'clearance'; release_loc = n.get('location'); break
            j += 1
        row.update(terminal=terminal, n_actions=n_act, carry_gain=carry_gain,
                   carry_dist=carry_dist, dribbles=dribbles, shot_xg=shot_xg,
                   release_x=release_loc[0] if release_loc else np.nan,
                   release_xt=xt(release_loc) if release_loc else np.nan)

        # --- the next teammate's reception, if the spell ended in a completed pass ---
        row['out_exists'] = False
        if terminal == 'pass':
            for k2 in range(k + 1, min(k + 12, len(recs))):
                i2, e2 = recs[k2]
                if e2['team']['id'] != team: break
                if e2.get('possession') != e.get('possession'): break
                if e2['player']['id'] == pid: continue
                row['out_exists'] = True
                row['next_player_id'] = e2['player']['id']
                row['next_player'] = e2['player']['name']
                row['next_ev_index'] = e2['index']
                row.update({f'out_{a}': b for a, b in situation(e2, frames).items()})
                row.update({f'outpass_{a}': b for a, b in incoming(e2, byid, team).items()})
                break
        rows.append(row)
    return rows

if __name__ == '__main__':
    ms = json.load(open(f'{DATA}/matches_9_281.json'))
    rows = []
    for m in ms: rows += spells_for_match(m['match_id'])
    df = pd.DataFrame(rows)
    df['kept'] = df.terminal.isin(['pass', 'shot', 'foul_won'])
    df['lost'] = df.terminal.str.startswith('lost')
    df['resolved'] = df.kept | df.lost | (df.terminal == 'clearance')
    df.to_csv(f'{OUT}/handoff.csv', index=False)
    print(df.shape)
    print(df.terminal.value_counts().to_string())
    print('\nspells ending in a completed pass:', (df.terminal == 'pass').sum(),
          '| of which the next teammate reception is linked:', df.out_exists.sum())
    lev = df[df.team == 'Bayer Leverkusen']
    print('Leverkusen spells:', len(lev), '| linked:', lev.out_exists.sum())
