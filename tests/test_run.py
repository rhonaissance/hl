import json
import os

import numpy as np
import pandas as pd

from hl import run


def synth(path, seed, t0, n=6000, L=20):
    rng = np.random.default_rng(seed)
    t = t0 + np.arange(n) * 500
    mid = 50000 + np.cumsum(rng.normal(0, 4, n))
    bb = np.floor(mid - 0.5)
    ba = bb + 1 + (rng.random(n) < 0.3)
    bpx, apx = bb[:, None] - np.arange(L), ba[:, None] + np.arange(L)
    rows = []
    for k in range(n):
        for _ in range(rng.poisson(1.5)):
            g = int(rng.choice([-1, 1]))
            lv = min(int(rng.geometric(0.6)) - 1, L - 1)
            rows.append((t[k] + rng.integers(0, 500), bpx[k, lv] if g < 0 else apx[k, lv],
                         rng.exponential(0.5), g, int(rng.random() < 0.1)))
    tr = np.array(sorted(rows, key=lambda r: r[0]))
    np.savez_compressed(path, t=t, nb=np.full(n, L, np.int16), na=np.full(n, L, np.int16), bpx=bpx,
                        bsz=rng.exponential(1, (n, L)), apx=apx, asz=rng.exponential(1, (n, L)),
                        tt=tr[:, 0].astype(np.int64), tpx=tr[:, 1], tsz=tr[:, 2], tside=tr[:, 3].astype(np.int8),
                        tliq=tr[:, 4].astype(np.int8))


def test_walk_forward_smoke(tmp_path):
    os.makedirs(tmp_path / 'day/SYN')
    for k in range(3):
        synth(tmp_path / f'day/SYN/2026100{k + 1}.npz', k, 1_790_000_000_000 + k * 86_400_000)
    out = run.main(['--coin', 'SYN', '--days', str(tmp_path / 'day'), '--train', '2', '--warm', '60',
                    '--jobs', '2', '--out', str(tmp_path / 'res'),
                    '--grid', '{"base": "deep_catch", "d_liq": [8, 20]}'])
    S = pd.read_csv(f'{out}/summary.csv')
    assert len(S) == 2 * (len(run.PRESETS) + 2) and S['days'].eq(1).all()
    P = pd.read_csv(f'{out}/probe.csv')
    assert set(P['queue']) == {'pess', 'prop'} and (P['fills_day'] > 0).any()
    m = json.load(open(f'{out}/meta.json'))
    assert m['n_configs'] == len(run.PRESETS) + 2 and len(m['data']) == 3
