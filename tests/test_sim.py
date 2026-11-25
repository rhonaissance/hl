import math

import numpy as np
import pytest

from hl import sim
from hl.sim import INF, Cfg, Params

B = [(99.9, 5), (99.8, 5), (99.7, 5), (99.6, 5), (99.5, 5)]
A = [(100.1, 5), (100.2, 5), (100.3, 5), (100.4, 5), (100.5, 5)]
CFG = dict(warm_s=0, lat_ms=200, maker_bp=1.5, taker_bp=4.5, slip_bp=0.0, probe_usd=100.0)  # 1 coin at 100


def day(snaps, trades, L=5):
    n = len(snaps)
    BP, BS, AP, AS = (np.full((n, L), np.nan) for _ in range(4))
    for k, (_, bids, asks) in enumerate(snaps):
        for c, (p, s) in enumerate(bids):
            BP[k, c], BS[k, c] = p, s
        for c, (p, s) in enumerate(asks):
            AP[k, c], AS[k, c] = p, s
    tr = np.array(trades, float).reshape(-1, 5)
    return dict(t=np.array([s[0] for s in snaps], np.int64), nb=np.isfinite(BP).sum(1), na=np.isfinite(AP).sum(1),
                bpx=BP, bsz=BS, apx=AP, asz=AS, tt=tr[:, 0].astype(np.int64), tpx=tr[:, 1], tsz=tr[:, 2],
                tside=tr[:, 3].astype(np.int8), tliq=tr[:, 4].astype(np.int8))


def probe(D, pr, queue='pess'):
    return sim.simulate(D, Cfg(queue=queue, **CFG), probe=pr)['fills']


def test_queue_pess_join_behind_visible():
    D = day([(0, B, A), (1000, B, A)], [(300, 99.9, 3, -1, 0), (400, 99.9, 3, -1, 0)])
    F = probe(D, [(1, 10.0)])  # bid 10bp under mid 100.0 -> 99.9, 5 ahead
    assert list(F['t']) == [400] and F['w'][0] == pytest.approx(1.0) and F['px'][0] == pytest.approx(99.9)


def test_latency_and_partial_price_through():
    D = day([(0, B, A), (1000, B, A)],
            [(100, 99.0, 10, -1, 0), (300, 99.0, 0.25, -1, 0), (350, 98.9, 2, -1, 0)])
    F = probe(D, [(1, 10.0)])  # not live before 200ms; through-prints fill up to their size
    assert list(F['t']) == [300, 350] and F['w'].sum() == pytest.approx(1.0) and F['w'][0] == pytest.approx(0.25)


def test_beyond_visible_depth():
    D = day([(0, B, A), (1000, B, A)], [(300, 99.0, 10, -1, 0)])
    assert len(probe(D, [(1, 100.0)], 'pess')['t']) == 0          # 99.0 beyond 5 visible levels: no touch fills
    assert probe(D, [(1, 100.0)], 'prop')['w'].sum() == pytest.approx(1.0)  # ahead = deepest visible size 5


def test_prop_cancels_vs_pess():
    B8, B4 = [(99.9, 8)] + B[1:], [(99.9, 4)] + B[1:]
    D = day([(0, B, A), (500, B8, A), (600, B4, A), (1000, B4, A)], [(700, 99.9, 3, -1, 0)])
    assert len(probe(D, [(1, 10.0)], 'pess')['t']) == 0                     # ahead min(5, 4) = 4 > 3
    assert probe(D, [(1, 10.0)], 'prop')['w'].sum() == pytest.approx(0.5)  # ahead 5 - 4*5/8 = 2.5


def test_alo_reject_when_crossing_at_activation():
    Bx, Ax = [(99.8, 5)] + B[2:], [(99.9, 5)] + A
    D = day([(0, B, A), (100, Bx, Ax[:5]), (1000, Bx, Ax[:5])], [(200, 99.0, 10, -1, 0)])
    F = probe(D, [(1, 10.0)])
    assert 200 not in list(F['t'])


def test_policy_accounting():
    D = day([(0, B, A), (1000, B, A)], [(300, 99.8, 10, -1, 0)])
    P = Params(d0=10.0, q_max=1, size_usd=100.0)
    r = sim.simulate(D, Cfg(**CFG), P=P)
    d = r['daily']
    assert d['maker_fills'] == 1 and d['taker_n'] == 1
    assert d['pnl'] == pytest.approx(-99.9 * 1.5e-4 - 99.9 * 4.5e-4)  # buy 99.9 maker, flush at bid 99.9 taker


def test_tox_cancel_and_liq_catch():
    # non-liq sell flood prints at 99.9 (above our 99.7 bid): tox rule cancels before the 99.0 through-print
    flood = [(250 + 10 * k, 99.9, 50, -1, 0) for k in range(5)]
    D = day([(0, B, A), (240, B, A), (1000, B, A)], flood + [(900, 99.0, 10, -1, 0)])
    base = dict(d0=30.0, q_max=1, size_usd=100.0)
    assert sim.simulate(D, Cfg(**CFG), P=Params(**base))['daily']['maker_fills'] == 1
    assert sim.simulate(D, Cfg(**CFG), P=Params(**base, tox=3.0))['daily']['maker_fills'] == 0
    # same flow flagged as liquidations: no regular quotes, catch order at d_liq=50bp (99.5) gets the sweep
    Dl = day([(0, B, A), (300, B, A), (1000, B, A)],
             [(250 + 10 * k, 99.9, 50, -1, 1) for k in range(5)] + [(900, 98.0, 10, -1, 0)])
    F = sim.simulate(Dl, Cfg(**CFG), P=Params(d0=INF, q_max=1, size_usd=100.0, liq_on=2.0, d_liq=50.0))['fills']
    assert len(F['t']) == 1 and F['px'][0] == pytest.approx(99.5) and F['d'][0] == 50.0
