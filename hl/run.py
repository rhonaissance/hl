"""Walk-forward experiment on data/day/<COIN>/*.npz.

1. every day x queue scenario: probe = resting orders on a depth grid (no inventory) -> fill-conditional markouts
2. every test day: fit estimators on the previous --train days only (reference beta, markout ridge per scenario),
   run every policy config, record daily PnL
3. results/<stamp>_<COIN>/ meta.json (commit, data sha256, configs) probe.csv daily.csv summary.csv

usage: python -m hl.run --coin BTC [--train 5] [--queue pess,prop] [--configs all|a,b] [--grid JSON] [--jobs 4]
       --grid '{"base": "deep_catch", "d_liq": [8, 15, 25], "liq_on": [1, 2, 4]}'   (Infinity allowed)
Judge on queue=pess; every config ever tried counts toward the DSR trial number.
"""
import argparse
import dataclasses
import glob
import hashlib
import itertools
import json
import os
import subprocess
import time
import multiprocessing as mp

import numpy as np
import pandas as pd

from . import data, sim, stats
from .sim import INF, Cfg, Est, Params

PROBE = [(s, float(d)) for d in (0, 1, 2, 3, 5, 8, 12, 20, 30, 50) for s in (1, -1)]
PRESETS = {
    # classic MM corner: ignore x, quote near touch, inventory skew
    'mm_touch': dict(d0=0.5, skew=1.0, q_max=5),
    'mm_tox': dict(d0=0.5, skew=1.0, q_max=5, tox=3.0),
    'mm_gate': dict(d0=1.0, skew=1.0, q_max=5, tox=3.0, gate=0.0),
    # previous project: fixed deep level on both sides; take-profit 10bp over entry or time stop
    'deep_fixed': dict(d0=15.0, q_max=2, d_exit=10.0, t_hold=900, requote=2.0),
    # deep-catching corner: quote only into liquidation flow, one unit, take-profit over entry or time stop
    'deep_catch': dict(d0=INF, liq_on=2.0, d_liq=15.0, q_max=1, d_exit=10.0, t_hold=900, requote=2.0),
    'deep_catch_gate': dict(d0=INF, liq_on=2.0, d_liq=15.0, q_max=1, d_exit=10.0, t_hold=900, requote=2.0, gate=0.0),
    # interior: both mechanisms + estimator gate
    'hybrid': dict(d0=3.0, skew=1.0, q_max=4, tox=3.0, liq_on=2.0, d_liq=12.0, d_exit=5.0, t_hold=1800, gate=0.0),
}


def day_of(p):
    return os.path.basename(p)[:-4]


def load(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def cat(Fs):
    return {k: np.concatenate([f[k] for f in Fs]) for k in Fs[0]}


def _cfg(a, q):
    return Cfg(queue=q, lat_ms=a.lat, maker_bp=a.maker, taker_bp=a.taker, warm_s=a.warm)


def probe_job(arg):
    p, q, a, want_ref = arg
    D, cfg = load(p), _cfg(a, q)
    F = sim.simulate(D, cfg, probe=PROBE)['fills']
    F.update(sim.markouts(D, F, cfg.horizons, cfg.maker_bp))
    return p, q, F, sim.ref_xy(D, cfg.h_ref) if want_ref else None


def policy_job(arg):
    p, q, name, P, est, a = arg
    D, cfg = load(p), _cfg(a, q)
    r = sim.simulate(D, cfg, P=Params(**P), est=est)
    F = r['fills']
    mk = sim.markouts(D, F, (cfg.h_mark,), cfg.maker_bp)['net%d' % cfg.h_mark]
    ok = np.isfinite(mk)
    row = dict(day=day_of(p), queue=q, config=name, **(r['daily'] or {}))
    row['net_bp'] = float(np.average(mk[ok], weights=F['w'][ok])) if ok.any() else np.nan
    return row


def configs(names, grid):
    C = {n: PRESETS[n] for n in (PRESETS if names == 'all' else names.split(','))}
    if grid:
        g = json.loads(grid)
        b = g.pop('base')
        for vals in itertools.product(*g.values()):
            C['%s[%s]' % (b, ','.join(f'{k}={v}' for k, v in zip(g, vals)))] = {**PRESETS[b], **dict(zip(g, vals))}
    return C


def probe_table(F, queues, files, hs=Cfg().horizons):
    rows = []
    for (p, q), f in F.items():
        df = pd.DataFrame({'d': f['d'], 'w': f['w'], **{f'net{h}': f[f'net{h}'] for h in hs}})
        for d, g in df.groupby('d'):
            r = dict(queue=q, day=day_of(p), depth=d, fills=g.w.sum())
            for h in hs:
                ok = g[f'net{h}'].notna()
                r[f'net{h}'] = np.average(g[f'net{h}'][ok], weights=g.w[ok]) if ok.any() else np.nan
            rows.append(r)
    P = pd.DataFrame(rows)
    g = P.groupby(['queue', 'depth'])
    T = g.agg(days=('day', 'nunique'), fills_day=('fills', 'mean'), **{f'net{h}': (f'net{h}', 'mean') for h in hs})
    T['t60'] = g['net60'].apply(stats.tstat)
    return T.reset_index()


def summarize(daily, n_cfg):
    g = daily.groupby(['queue', 'config'])
    S = g.agg(days=('day', 'nunique'), pnl_day=('pnl', 'mean'), fills_day=('maker_fills', 'mean'),
              maker_usd_day=('maker_usd', 'mean'), taker_usd_day=('taker_usd', 'mean'), fees_day=('fees', 'mean'),
              net_bp=('net_bp', 'mean'), q_max=('q_max_units', 'max'))
    S['t'] = g['pnl'].apply(stats.tstat)
    S['sr_ann'] = g['pnl'].apply(lambda x: x.mean() / x.std(ddof=1) * np.sqrt(365) if x.std(ddof=1) > 0 else np.nan)
    S['dsr'] = g['pnl'].apply(lambda x: stats.dsr(x, n_cfg))
    return S.reset_index().sort_values(['queue', 't'], ascending=[True, False])


def _git(*c):
    try:
        return subprocess.run(['git', *c], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except OSError:
        return ''


def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--coin', required=True)
    ap.add_argument('--days', default='data/day')
    ap.add_argument('--dates', default='', help='YYYYMMDD:YYYYMMDD (default: all built days)')
    ap.add_argument('--train', type=int, default=5)
    ap.add_argument('--queue', default='pess,prop')
    ap.add_argument('--configs', default='all')
    ap.add_argument('--grid', default='')
    ap.add_argument('--lat', type=int, default=200)
    ap.add_argument('--maker', type=float, default=1.5)
    ap.add_argument('--taker', type=float, default=4.5)
    ap.add_argument('--warm', type=float, default=900)
    ap.add_argument('--jobs', type=int, default=os.cpu_count())
    ap.add_argument('--out', default='results')
    a = ap.parse_args(argv)

    files = sorted(glob.glob(f'{a.days}/{a.coin}/*.npz'))
    if a.dates:
        keep = set(data.days(a.dates))
        files = [f for f in files if day_of(f) in keep]
    if len(files) <= a.train:
        raise SystemExit(f'{len(files)} days found, need more than --train {a.train}')
    queues, cfgs = a.queue.split(','), configs(a.configs, a.grid)
    out = f'{a.out}/{time.strftime("%Y%m%d-%H%M%S")}_{a.coin}'
    os.makedirs(out)
    meta = dict(commit=_git('rev-parse', 'HEAD'), dirty=bool(_git('status', '--porcelain')), args=vars(a),
                n_configs=len(cfgs), configs=cfgs, probe=PROBE, cfg=dataclasses.asdict(Cfg()),
                data=[(os.path.basename(p), os.path.getsize(p), _sha(p)) for p in files])
    with open(f'{out}/meta.json', 'w') as f:
        json.dump(meta, f, indent=1, default=str)
    if meta['dirty']:
        print('WARNING: uncommitted changes - do not use this run for decisions')

    with mp.get_context("spawn").Pool(a.jobs) as pool:
        pr = pool.map(probe_job, [(p, q, a, q == queues[0]) for p in files for q in queues])
        F = {(p, q): f for p, q, f, _ in pr}
        XY = {p: xy for p, _, _, xy in pr if xy is not None}
        T = probe_table(F, queues, files)
        T.to_csv(f'{out}/probe.csv', index=False)
        print('\nprobe: resting order at depth d (bp from reference), fill-weighted net markout (bp, after maker fee)')
        print(T.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
        jobs = []
        for k in range(a.train, len(files)):
            tr = files[k - a.train:k]
            beta = sim.fit_ref([XY[p] for p in tr])
            for q in queues:
                fm = sim.fit_markout(cat([F[(p, q)] for p in tr]), Cfg().h_mark)
                est = Est(beta, *fm) if fm else Est(beta)
                jobs += [(files[k], q, n, P, est, a) for n, P in cfgs.items()]
        daily = pd.DataFrame(pool.map(policy_job, jobs))
    daily.to_csv(f'{out}/daily.csv', index=False)
    S = summarize(daily, len(cfgs))
    S.to_csv(f'{out}/summary.csv', index=False)
    print(f'\npolicy, walk-forward ({len(files) - a.train} test days, {len(cfgs)} configs = DSR trials)')
    print(S.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    print(f'\n-> {out}')
    return out


if __name__ == '__main__':
    main()
