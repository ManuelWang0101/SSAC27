"""Download the StatsBomb open data used here into ./data (run from the repo root).
Bayer Leverkusen 2023/24 Bundesliga (competition 9, season 281): matches, events, lineups, 360 frames.
xT grid training also uses 2015/16 PL/Bundesliga events (fit_xt.py); out/xt_grid.json is provided."""
import json, os, urllib.request
BASE = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data'
def get(path, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest): urllib.request.urlretrieve(f'{BASE}/{path}', dest)
get('matches/9/281.json', 'data/matches_9_281.json')
for m in json.load(open('data/matches_9_281.json')):
    mid = m['match_id']
    get(f'events/{mid}.json', f'data/events/{mid}.json')
    get(f'lineups/{mid}.json', f'data/lineups/{mid}.json')
    try: get(f'three-sixty/{mid}.json', f'data/three-sixty/{mid}.json')
    except Exception: pass
print('done')
