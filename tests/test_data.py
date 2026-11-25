import json

import lz4.frame
import numpy as np

from hl import data


def _w(p, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    with lz4.frame.open(str(p), 'wt') as f:
        f.write('\n'.join(lines) + '\n')


def lv(px, sz, n=1):
    return {'px': str(px), 'sz': str(sz), 'n': n}


def test_build_formats(tmp_path):
    raw = tmp_path / 'raw'
    snap = lambda t, b, a: {'time': '2026-10-01T00:00:00.1', 'ver_num': 1,
                            'raw': {'channel': 'l2Book', 'data': {'coin': 'BTC', 'time': t,
                                                                 'levels': [[lv(*x) for x in b], [lv(*x) for x in a]]}}}
    _w(raw / 'l2/BTC/20261001/0/l2Book/BTC.lz4', [
        json.dumps(snap(1000, [(100, 1), (99, 2)], [(101, 3)])),
        repr(snap(500, [(100, 1)], [(101, 1)])),          # python-repr row, out of order
        json.dumps(snap(1500, [(101, 1)], [(101, 1)])),   # crossed -> dropped
    ])
    fill = lambda **k: {'coin': 'BTC', 'px': '100.0', 'sz': '0.5', 'side': 'A', 'time': 1200, 'tid': 7,
                        'crossed': True, 'oid': 1, **k}
    _w(raw / 'fills/20261001/0.lz4', [
        json.dumps({'local_time': 'x', 'block_time': 'y', 'block_number': 1, 'events': [
            ['0xa', fill()], ['0xb', fill(side='B', crossed=False, liquidation={'liquidatedUser': '0xb'})],
            ['0xc', fill(coin='ETH', tid=8)]]}),
        json.dumps(['0xd', fill(tid=9, time=900, side='B', px='101.0')]),   # older [user, fill] format
    ])
    p = data.build('BTC', '20261001', str(raw), str(tmp_path / 'day'))
    z = np.load(p)
    assert list(z['t']) == [500, 1000] and list(z['nb']) == [1, 2] and z['bpx'][1, 1] == 99.0 and z['asz'][1, 0] == 3
    assert list(z['tt']) == [900, 1200] and list(z['tside']) == [1, -1] and list(z['tliq']) == [0, 1]
    assert list(z['tpx']) == [101.0, 100.0]


def test_days():
    assert data.days('20261030:20261102') == ['20261030', '20261031', '20261101', '20261102']
