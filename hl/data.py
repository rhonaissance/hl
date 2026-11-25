"""HL S3 archive -> one npz per coin-day.

raw (scripts/fetch.sh):
  data/raw/l2/<COIN>/<YYYYMMDD>/<H>/l2Book/<COIN>.lz4   <- s3://hyperliquid-archive/market_data/<d>/<H>/l2Book/<COIN>.lz4
  data/raw/fills/<YYYYMMDD>/<H>[.lz4]                    <- s3://hl-mainnet-node-data/node_fills_by_block/hourly/<d>/
out:
  data/day/<COIN>/<YYYYMMDD>.npz
    t int64 ms | nb, na levels present | bpx bsz apx asz [n, L] best first, NaN padded
    tt int64 ms | tpx tsz | tside +1 buy aggressor / -1 sell | tliq 1 if any fill of the match has 'liquidation'

usage: python -m hl.data --coin BTC --dates 20261001:20261021 [--raw data/raw] [--out data/day]
"""
import argparse
import ast
import datetime as dt
import glob
import json
import os
import re

import numpy as np

try:
    from orjson import loads as _jl
except ImportError:
    _jl = json.loads

L = 20


def _open(p):
    if p.endswith('.lz4'):
        import lz4.frame
        return lz4.frame.open(p, 'rt')
    return open(p)


def _obj(s):
    try:
        return _jl(s)
    except ValueError:
        return ast.literal_eval(s)  # some archive rows are python-repr dicts


def l2_day(files):
    T, buf, n = [], np.full((1 << 16, 4 * L), np.nan), 0
    for p in files:
        with _open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                o = _obj(line)
                r = o.get('raw', o)
                r = _obj(r) if isinstance(r, str) else r
                d = r.get('data', r)
                if n == len(buf):
                    buf = np.concatenate([buf, np.full_like(buf, np.nan)])
                row = buf[n]
                for c, side in ((0, d['levels'][0]), (2 * L, d['levels'][1])):
                    for k, v in enumerate(side[:L]):
                        row[c + k], row[c + L + k] = float(v['px']), float(v['sz'])
                T.append(int(d['time']))
                n += 1
    a, t = buf[:n], np.array(T, np.int64)
    o = np.argsort(t, kind='stable')
    a, t = a[o], t[o]
    keep = np.r_[t[1:] != t[:-1], True]  # duplicate ms -> last
    a, t = a[keep], t[keep]
    bpx, bsz, apx, asz = a[:, :L], a[:, L:2 * L], a[:, 2 * L:3 * L], a[:, 3 * L:]
    ok = np.isfinite(bpx[:, 0]) & np.isfinite(apx[:, 0]) & (bpx[:, 0] < apx[:, 0])
    return dict(t=t[ok], nb=np.isfinite(bpx[ok]).sum(1).astype(np.int16), na=np.isfinite(apx[ok]).sum(1).astype(np.int16),
                bpx=bpx[ok], bsz=bsz[ok], apx=apx[ok], asz=asz[ok]), int((~ok).sum())


def fills_day(files, coin):
    keys, taker, liq = ('"%s"' % coin, "'%s'" % coin), {}, set()
    for p in files:
        with _open(p) as f:
            for line in f:
                if keys[0] not in line and keys[1] not in line:
                    continue
                o = _obj(line)
                if isinstance(o, dict) and 'events' in o:            # node_fills_by_block
                    ev = o['events']
                elif isinstance(o, list) and len(o) == 2 and isinstance(o[1], dict):  # node_fills [user, fill]
                    ev = [o]
                else:
                    continue
                for e in ev:
                    x = e[1]
                    if x.get('coin') != coin:
                        continue
                    k = (x['tid'], int(x['time']))
                    if x.get('liquidation'):
                        liq.add(k)
                    if x.get('crossed'):  # one taker fill per match; px = maker price
                        taker[k] = (int(x['time']), float(x['px']), float(x['sz']), 1 if x['side'] == 'B' else -1)
    r = list(taker.items())
    tt = np.array([v[0] for _, v in r], np.int64)
    o = np.argsort(tt, kind='stable')
    col = lambda i, ty: np.array([v[i] for _, v in r], ty)[o]
    return dict(tt=tt[o], tpx=col(1, float), tsz=col(2, float), tside=col(3, np.int8),
                tliq=np.array([k in liq for k, _ in r], np.int8)[o])


def _num(p):
    return int(re.match(r'\d+', os.path.basename(p)).group())


def build(coin, day, raw, out):
    l2f = sorted((p for p in glob.glob(f'{raw}/l2/{coin}/{day}/*/l2Book/*') if os.path.basename(p) in (coin, coin + '.lz4')),
                 key=lambda p: int(p.split(os.sep)[-3]))
    ff = sorted(glob.glob(f'{raw}/fills/{day}/*'), key=_num)
    if not l2f or not ff:
        print(f'{coin} {day}: missing raw (l2 {len(l2f)} files, fills {len(ff)} files)')
        return None
    D, bad = l2_day(l2f)
    D.update(fills_day(ff, coin))
    os.makedirs(f'{out}/{coin}', exist_ok=True)
    path = f'{out}/{coin}/{day}.npz'
    np.savez_compressed(path, **D)
    gap = np.diff(D['t']).max() / 1000 if len(D['t']) > 1 else float('nan')
    print(f'{coin} {day}: {len(D["t"])} snaps ({bad} dropped, max gap {gap:.1f}s, {len(l2f)} h), '
          f'{len(D["tt"])} prints ({int(D["tliq"].sum())} liq) -> {path}')
    return path


def days(spec):
    a, b = (spec.split(':') + [spec])[:2]
    d0, d1 = (dt.datetime.strptime(x, '%Y%m%d') for x in (a, b))
    return [(d0 + dt.timedelta(n)).strftime('%Y%m%d') for n in range((d1 - d0).days + 1)]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--coin', required=True)
    ap.add_argument('--dates', required=True, help='YYYYMMDD or YYYYMMDD:YYYYMMDD')
    ap.add_argument('--raw', default='data/raw')
    ap.add_argument('--out', default='data/day')
    a = ap.parse_args()
    for d in days(a.dates):
        build(a.coin, d, a.raw, a.out)
