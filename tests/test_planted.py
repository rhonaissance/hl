"""Planted-signal regression: staircase sell sweeps of 40bp over 5s. 'L' sweeps are flagged as liquidations and
fully revert within 60s; 'I' sweeps are unflagged and permanent. Checks signs end to end (sides, markouts, features,
policy accounting)."""
import numpy as np

from hl import run, sim
from hl.sim import Cfg, Params


def planted(seed, kinds, n=14400, L=20, every=600, steps=10):
    rng = np.random.default_rng(seed)
    t = 1_790_000_000_000 + np.arange(n) * 500
    shock, liq, sweep = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
    k = 1000
    while k < n - 200:
        kind = kinds[rng.integers(len(kinds))]
        shock[k:k + steps] += np.linspace(0, -40e-4, steps + 1)[1:]
        sweep[k:k + steps], liq[k:k + steps] = True, kind == 'L'
        shock[k + steps:k + steps + 120] += np.linspace(-40e-4, 0, 120) if kind == 'L' else -40e-4
        if kind == 'I':
            shock[k + steps + 120:] += -40e-4
        k += every + int(rng.integers(0, 200))
    mid = 50000 * np.exp(np.cumsum(rng.normal(0, 0.5e-4, n)) + shock)
    bb = np.floor(mid - 0.5)
    bpx, apx = bb[:, None] - np.arange(L), bb[:, None] + 1 + np.arange(L)
    rows = []
    for j in range(n):
        for _ in range(rng.poisson(1.0)):
            g = int(rng.choice([-1, 1]))
            rows.append((t[j] + int(rng.integers(0, 500)), bpx[j, 0] if g < 0 else apx[j, 0], rng.exponential(0.05), g, 0))
        if sweep[j]:
            for p in np.arange(bpx[j - 1, 0], bpx[j, 0] - 1, -5.0):
                rows.append((t[j] - 1, p, 0.5, -1, int(liq[j])))
    tr = np.array(sorted(rows, key=lambda r: r[0]))
    return dict(t=t, nb=np.full(n, L, np.int16), na=np.full(n, L, np.int16), bpx=bpx, bsz=np.ones((n, L)),
                apx=apx, asz=np.ones((n, L)), tt=tr[:, 0].astype(np.int64), tpx=tr[:, 1], tsz=tr[:, 2],
                tside=tr[:, 3].astype(np.int8), tliq=tr[:, 4].astype(np.int8))


CFG = Cfg(warm_s=300)


def test_probe_separates_forced_from_informed():
    D = planted(1, 'LI')
    F = sim.simulate(D, CFG, probe=run.PROBE)['fills']
    F.update(sim.markouts(D, F, CFG.horizons, CFG.maker_bp))
    deep, liq = (F['d'] >= 8) & (F['s'] == 1), np.expm1(F['x0']) > 2
    assert np.nanmean(F['net60'][deep & liq]) > 5 and np.nanmean(F['net60'][deep & ~liq]) < -5
    w, mu, sd, yb = sim.fit_markout(F, 60)
    est = sim.Est((0, 0), w, mu, sd, yb)
    assert est.pred(15, (3.0, 0.0, 5, 0, 5)) > 0 > est.pred(15, (0.0, 3.0, 5, 0, 5))


def test_deep_catch_only_trades_forced_flow():
    for kinds, sign in (('L', 1), ('I', 0)):
        d = sim.simulate(planted(2, kinds), CFG, P=Params(**run.PRESETS['deep_catch']))['daily']
        assert (d['pnl'] > 0 and d['maker_fills'] >= 10) if sign else d['maker_fills'] == 0
    assert sim.simulate(planted(2, 'I'), CFG, P=Params(**run.PRESETS['deep_fixed']))['daily']['pnl'] < 0
