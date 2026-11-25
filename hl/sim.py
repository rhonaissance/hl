"""Event-driven replay of L2 snapshots + trade prints for the quoting policy

    δ_b(x, q), δ_a(x, q) ∈ [0, ∞]   (∞ = withdraw)
    |q| > q*            -> taker flush
    toxic (non-liq) inflow toward a side -> cancel that side
    liquidation inflow toward a side      -> catch at depth d_liq (deep catching = special case)

Our orders are phantoms: they never change the historical book or prints.
"""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

INF = math.inf
NX = 5  # x = (log1p liq-flow ratio, log1p toxic-flow ratio, 1-min vol bp, s*imb1, -s*mid-dev bp)


@dataclass
class Params:                 # policy θ; INF disables a component
    d0: float = 2.0           # regular quote depth from reference price, bp (INF: no regular quotes)
    k_vol: float = 0.0        # + k_vol * 1-min vol (bp)
    skew: float = 0.0         # + skew * s * q_units: widen the side that adds inventory
    q_max: float = 3.0        # inventory cap in order units: adding orders sized to the room left (off below
    #                           half a unit), excess taker-flushed
    t_hold: float = INF       # s: taker-flush whole inventory if held longer
    d_exit: float = INF       # bp beyond average entry price for the reducing side (take-profit; the closer of
    #                           this and the regular quote is used). Anchored to entry, not to the moving mid.
    tox: float = INF          # withdraw adding side if non-liq aggressive flow toward it > tox x baseline
    liq_on: float = INF       # liquidation flow toward side > liq_on x baseline -> quote that side at d_liq
    d_liq: float = 20.0
    gate: float = -INF        # quote adding side only if predicted fill-conditional net (bp) >= gate
    requote: float = 0.5      # bp move of target before cancel/replace (keeps queue priority)
    size_usd: float = 1000.0  # order unit


@dataclass
class Cfg:                    # market / simulator assumptions
    queue: str = 'pess'       # pess: join behind all visible size, all cancels behind us,
    #                           beyond visible depth -> fills only on price-through
    #                           prop: cancels pro-rata, beyond visible -> ahead = deepest visible size
    lat_ms: int = 200         # placement and cancel latency
    maker_bp: float = 1.5
    taker_bp: float = 4.5
    slip_bp: float = 1.0      # extra cost on taker flush
    probe_usd: float = 1000.0  # probe order size
    freeze: float = 1.0       # never move a resting order away from the market while flow toward its side
    #                           exceeds freeze x baseline (deep orders must not run from the sweep they wait for)
    warm_s: float = 900       # no orders before features warm up
    tau_s: float = 2.0        # s, short flow EWMA
    tau_l: float = 600.0      # s, flow baseline
    tau_v: float = 300.0      # s, vol EWMA
    tau_m: float = 60.0       # s, mid EWMA (mid deviation feature)
    h_ref: float = 5.0        # s, horizon of the reference-price model
    h_mark: int = 60          # s, horizon of the fill-conditional markout model
    horizons: tuple = (1, 10, 60, 300)


class Order:
    __slots__ = ('s', 'px', 'sz', 'sz0', 'd', 'key', 'live', 'cancel_at', 'init', 'ahead', 'pv', 'tr')

    def __init__(self, s, px, sz, d, key, live):
        self.s, self.px, self.sz, self.sz0, self.d, self.key, self.live = s, px, sz, sz, d, key, live
        self.cancel_at, self.init, self.ahead, self.pv, self.tr = INF, False, INF, None, 0.0


def infer_tick(bpx, apx, n=5000):
    p = np.concatenate([bpx[:n].ravel(), apx[:n].ravel()])
    p = np.unique(p[np.isfinite(p)])
    d = np.diff(p)
    d = d[d > 0]
    return 10.0 ** math.floor(math.log10(d.min()) + 1e-6) if len(d) else 1e-8


def tick_at(mid, tick_obs):  # HL: 5 significant figures, integer prices always allowed
    return max(min(1.0, 10.0 ** (math.floor(math.log10(mid)) - 4)), tick_obs)


def rpx(x, s, tick):  # round bid down / ask up to tick
    k = math.floor(x / tick + 1e-9) if s > 0 else math.ceil(x / tick - 1e-9)
    return round(k * tick, 10)


def level(px_, sz_, px, s, eps, full):
    """Visible size at px on side s: 0 if better than best or an empty price inside the visible range,
    INF if deeper than the visible levels of a full-depth snapshot."""
    if (px - px_[0]) * s > eps:
        return 0.0
    for p, q in zip(px_, sz_):
        if abs(p - px) <= eps:
            return q
        if (px - p) * s > eps:
            return 0.0
    return INF if full else 0.0


def simulate(D, cfg, P=None, probe=None, est=None):
    """D: day dict from hl.data. Give P (policy) or probe = [(side, depth_bp), ...] (independent resting
    orders, no inventory). Returns {'fills': dict of arrays, 'daily': dict | None}."""
    ts = D['t']
    ns, nt = len(ts), len(D['tt'])
    if ns == 0:
        return {'fills': {c: np.zeros(0) for c in ['t', 's', 'd', 'px', 'w'] + ['x%d' % c for c in range(NX)]},
                'daily': None}
    BP, BS, AP, AS, NB, NA = D['bpx'], D['bsz'], D['apx'], D['asz'], D['nb'], D['na']
    Lmax = BP.shape[1]
    tm = np.concatenate([D['tt'], ts])
    kd = np.concatenate([np.ones(nt, np.int8), np.zeros(ns, np.int8)])
    ix = np.concatenate([np.arange(nt), np.arange(ns)])
    o = np.lexsort((-kd, tm))  # time order; trades before a snapshot with the same ms
    tm, kd, ix = tm[o].tolist(), kd[o].tolist(), ix[o].tolist()
    tpx, tsz, tsd, tlq = (D[k].tolist() for k in ('tpx', 'tsz', 'tside', 'tliq'))

    pol, prop, lat = P is not None, cfg.queue == 'prop', cfg.lat_ms
    tick0 = infer_tick(BP, AP)
    eps = tick0 / 4
    t_go = int(ts[0]) + int(cfg.warm_s * 1000)
    fs, lq, fl, V, me = [0.0, 0.0], [0.0, 0.0], 0.0, 0.0, 0.0
    m = hs = vol = 0.0
    bp = bs = ap = az = None
    fullb = fulla = False
    xs = [(0.0,) * NX, (0.0,) * NX]
    hist = deque()
    orders, slot, recs = [], {}, []
    q = qpx = cash = fee = mvol = tvol = qmax = qsum = 0.0
    nm = ntk = nq = 0
    t_open = None
    t_prev = tm[0] if tm else 0

    def base():
        return 0.5 * fl / cfg.tau_l + 1e-12

    def xlag(t):  # features of the last decision that could still change the order at t
        tl = t - lat
        while len(hist) > 1 and hist[1][0] <= tl:
            hist.popleft()
        return hist[0][1] if hist and hist[0][0] <= tl else ((0.0,) * NX, (0.0,) * NX)

    def fill(od, f, t):
        nonlocal q, qpx, cash, fee, mvol, nm, t_open
        od.sz -= f
        x = xlag(t)[0 if od.s > 0 else 1]
        if not pol:
            recs.append((t, od.s, od.d, od.px, f / od.sz0) + x)
            return
        prev = q
        q += od.s * f
        qpx = _avg(prev, qpx, q, od.px)
        n = f * od.px
        c = n * cfg.maker_bp / 1e4
        cash -= od.s * n + c
        fee += c
        mvol += n
        nm += 1
        if abs(q) < 1e-12:
            q, t_open = 0.0, None
        elif abs(prev) < 1e-12 or prev * q < 0:
            t_open = t
        recs.append((t, od.s, od.d, od.px, f) + x)

    def taker(qt, t):
        nonlocal q, qpx, cash, fee, tvol, ntk, t_open
        dq = qt - q
        if abs(dq) * m < 1e-9:
            return
        s = 1 if dq > 0 else -1
        rem, cost, last = abs(dq), 0.0, None
        for p_, z_ in (zip(ap, az) if s > 0 else zip(bp, bs)):
            take = min(rem, z_)
            cost, rem, last = cost + take * p_, rem - take, p_
            if rem <= 1e-12:
                break
        if rem > 1e-12:  # deeper than visible: last visible price + 10 bp
            cost += rem * last * (1 + s * 1e-3)
        n = cost * (1 + s * cfg.slip_bp / 1e4)
        c = n * cfg.taker_bp / 1e4
        cash -= s * n + c
        fee += c
        tvol += n
        ntk += 1
        qpx = _avg(q, qpx, qt, cost / abs(dq))
        if abs(qt) < 1e-12:
            q, t_open = 0.0, None
        else:
            if q * qt <= 0:
                t_open = t
            q = qt

    def target(s, d, ref):
        tk = tick_at(m, tick0)
        p = rpx(ref * (1 - s * d / 1e4), s, tk)
        if s > 0 and p >= ap[0] - eps:  # post-only
            p = rpx(ap[0] - tk, 1, tk)
        if s < 0 and p <= bp[0] + eps:
            p = rpx(bp[0] + tk, -1, tk)
        return p

    def place(key, s, px, sz, d, t):
        od = Order(s, px, sz, d, key, t + lat)
        orders.append(od)
        slot[key] = od

    def drop(key, t):
        od = slot.pop(key, None)
        if od is not None and od.cancel_at == INF:
            od.cancel_at = t + lat

    def ratio(acc, i):
        return acc[i] / cfg.tau_s / base()

    def frozen(od, tgt):  # target moved away from the market while flow hits this side
        i = (1 - od.s) >> 1
        return (tgt - od.px) * od.s < 0 and ratio(fs, i) + ratio(lq, i) > cfg.freeze

    for t, k, j in zip(tm, kd, ix):
        dt = (t - t_prev) / 1000.0
        if dt > 0:
            a = math.exp(-dt / cfg.tau_s)
            fs[0] *= a; fs[1] *= a; lq[0] *= a; lq[1] *= a
            fl *= math.exp(-dt / cfg.tau_l)
            V *= math.exp(-dt / cfg.tau_v)
            if m:
                me = m + (me - m) * math.exp(-dt / cfg.tau_m)
            t_prev = t

        dirty = False
        for od in orders:  # cancels take effect before anything else at the same ms
            if od.cancel_at <= t:
                od.sz, dirty = 0.0, True
            elif not od.init and t >= od.live and bp is not None:
                if (od.s > 0 and od.px >= ap[0] - eps) or (od.s < 0 and od.px <= bp[0] + eps):
                    od.sz, dirty = 0.0, True  # ALO would cross -> rejected
                    continue
                vis = level(bp, bs, od.px, 1, eps, fullb) if od.s > 0 else level(ap, az, od.px, -1, eps, fulla)
                if vis < INF:
                    od.ahead = vis
                elif prop:
                    od.ahead = (bs if od.s > 0 else az)[-1]
                od.pv, od.tr, od.init = vis, 0.0, True

        if k == 1:  # ---------------------------------------------------------------- trade print
            g, px, sz = tsd[j], tpx[j], tsz[j]
            i = (1 + g) >> 1  # sell aggressor (g=-1) hits bids (i=0)
            if tlq[j]:
                lq[i] += px * sz
            else:
                fs[i] += px * sz
            fl += px * sz
            for od in orders:
                if od.s != -g or not od.init or od.sz <= 0:
                    continue
                if (od.px - px) * od.s > eps:  # printed through our price
                    fill(od, min(od.sz, sz), t)
                    dirty = True
                elif abs(od.px - px) <= eps:
                    od.tr += sz
                    if od.ahead < INF:
                        f = sz - od.ahead
                        od.ahead = max(0.0, od.ahead - sz)
                        if f > 0:
                            fill(od, min(f, od.sz), t)
                            dirty = True
            if pol and P.tox < INF and i in slot and t >= t_go:
                s = 1 - 2 * i
                if s * q >= 0 and ratio(fs, i) > P.tox and ratio(lq, i) <= P.liq_on:
                    drop(i, t)
        else:  # ------------------------------------------------------------------- L2 snapshot
            b, a_ = NB[j], NA[j]
            bp, bs, ap, az = BP[j, :b].tolist(), BS[j, :b].tolist(), AP[j, :a_].tolist(), AS[j, :a_].tolist()
            fullb, fulla = b >= Lmax, a_ >= Lmax
            m_new = 0.5 * (bp[0] + ap[0])
            if m:
                V += math.log(m_new / m) ** 2
            else:
                me = m_new
            m, hs = m_new, 0.5 * (ap[0] - bp[0])
            vol = math.sqrt(V / cfg.tau_v * 60.0) * 1e4
            i1 = (bs[0] - az[0]) / (bs[0] + az[0])
            sb, sa = sum(bs[:5]), sum(az[:5])
            i5 = (sb - sa) / (sb + sa)
            mdev = (m - me) / m * 1e4
            ref = est.ref(m, hs, i1, i5) if est is not None else m
            bb = base()
            xs = [(math.log1p(lq[0] / cfg.tau_s / bb), math.log1p(fs[0] / cfg.tau_s / bb), vol, i1, -mdev),
                  (math.log1p(lq[1] / cfg.tau_s / bb), math.log1p(fs[1] / cfg.tau_s / bb), vol, -i1, mdev)]
            hist.append((t, tuple(xs)))
            while len(hist) > 1 and hist[1][0] <= t - lat:
                hist.popleft()

            for od in orders:
                if not od.init or od.sz <= 0:
                    continue
                if (od.s > 0 and ap[0] <= od.px + eps) or (od.s < 0 and bp[0] >= od.px - eps):
                    fill(od, od.sz, t)  # book crossed our price: someone traded into us
                    dirty = True
                    continue
                vis = level(bp, bs, od.px, 1, eps, fullb) if od.s > 0 else level(ap, az, od.px, -1, eps, fulla)
                if vis < INF:
                    if prop and od.pv is not None and od.pv < INF and od.ahead < INF:
                        b0 = od.pv - od.tr  # level size that should remain without cancels/adds
                        if b0 > 1e-12:
                            od.ahead -= max(0.0, b0 - vis) * min(1.0, od.ahead / b0)
                    od.ahead = min(od.ahead, vis)
                od.pv, od.tr = vis, 0.0

            if pol:
                qu = q * m / P.size_usd
                qmax, qsum, nq = max(qmax, abs(qu)), qsum + abs(qu), nq + 1

            if t >= t_go:
                if not pol:  # ----------------------------------------------------- probe orders
                    for key, (s, d) in enumerate(probe):
                        tgt, od = target(s, d, ref), slot.get(key)
                        if od is not None and od.sz <= 1e-12:
                            od = None
                        if od is None:
                            place(key, s, tgt, cfg.probe_usd / m, d, t)
                        elif abs(od.px - tgt) > max(0.5, 0.15 * d) * m / 1e4 and not frozen(od, tgt):
                            drop(key, t)
                            place(key, s, tgt, cfg.probe_usd / m, d, t)
                else:  # ---------------------------------------------------------------- policy
                    if abs(qu) > P.q_max + 0.5:  # slack: price moves alone change q_units
                        taker(math.copysign(P.q_max, q) * P.size_usd / m, t)
                    elif t_open is not None and (t - t_open) / 1000.0 > P.t_hold:
                        taker(0.0, t)
                    qu = q * m / P.size_usd
                    for i, s in ((0, 1), (1, -1)):
                        red = s * qu < -1e-9
                        room = 1.0 if red else min(1.0, P.q_max - abs(qu))  # order size in units
                        d = INF
                        if red or room >= 0.5:
                            d = P.d0 + P.k_vol * vol + P.skew * s * qu
                            if not red:
                                if ratio(fs, i) > P.tox:
                                    d = INF
                                if ratio(lq, i) > P.liq_on:
                                    d = P.d_liq
                                if d < INF and est is not None and P.gate > -INF and est.pred(d, xs[i]) < P.gate:
                                    d = INF
                            d = max(d, 0.0)
                        od = slot.get(i)
                        if od is not None and od.sz <= 1e-12:
                            slot.pop(i)
                            od = None
                        tgt = target(s, d, ref) if d < INF else None
                        if red and P.d_exit < INF:  # take-profit anchored to entry (long: ask at entry*(1+d_exit))
                            tp = target(s, P.d_exit, qpx)
                            tgt = tp if tgt is None else (min(tgt, tp) if s < 0 else max(tgt, tp))
                        if tgt is None:
                            drop(i, t)
                            continue
                        sz = (min(1.0, abs(qu)) if red else room) * P.size_usd / m
                        if od is None:
                            place(i, s, tgt, sz, d, t)
                        elif (abs(od.px - tgt) > max(P.requote * m / 1e4, eps) and not frozen(od, tgt)) \
                                or abs(od.sz - sz) > 0.5 * sz:
                            drop(i, t)
                            place(i, s, tgt, sz, d, t)

        if dirty:
            orders = [od for od in orders if od.sz > 1e-12]
            for key in [kk for kk, od in slot.items() if od.sz <= 1e-12]:
                del slot[key]

    daily = None
    if pol and bp is not None:
        taker(0.0, tm[-1])  # flat at day end
        daily = dict(pnl=cash, fees=fee, maker_fills=nm, maker_usd=mvol, taker_usd=tvol, taker_n=ntk,
                     q_max_units=qmax, q_avg_units=qsum / max(nq, 1))
    cols = ['t', 's', 'd', 'px', 'w'] + ['x%d' % c for c in range(NX)]
    R = np.array(recs, float).reshape(-1, len(cols))
    F = {c: R[:, n] for n, c in enumerate(cols)}
    F['t'] = F['t'].astype(np.int64)
    return {'fills': F, 'daily': daily}


def _avg(q0, p0, q1, p):
    """Average entry price after inventory moves q0 -> q1 at price p."""
    if abs(q1) < 1e-12:
        return 0.0
    if abs(q0) < 1e-12 or q0 * q1 < 0:
        return p
    if abs(q1) > abs(q0):
        return (abs(q0) * p0 + (abs(q1) - abs(q0)) * p) / abs(q1)
    return p0


def markouts(D, F, horizons, fee_bp):
    """net_H = side * (mid(t+H) - px) / px - fee, bp. NaN past the day's end."""
    mid = 0.5 * (D['bpx'][:, 0] + D['apx'][:, 0])
    out = {}
    for h in horizons:
        k = np.searchsorted(D['t'], F['t'] + int(h * 1000))
        ok = k < len(mid)
        v = np.full(len(F['t']), np.nan)
        v[ok] = F['s'][ok] * (mid[k[ok]] / F['px'][ok] - 1) * 1e4 - fee_bp
        out['net%d' % h] = v
    return out


# ------------------------------------------------------------------------------------------- estimators
def feat(d, x):
    l = math.log1p(d)
    return (l,) + tuple(x) + (l * x[0],)


class Est:
    """(a) reference price: mid + (b1*imb1 + b5*imb5) * half-spread
       (b) fill-conditional net markout (bp) at Cfg.h_mark: ridge on feat(depth, x)."""

    def __init__(self, beta=(0.0, 0.0), w=None, mu=None, sd=None, ybar=0.0):
        self.beta, self.w, self.mu, self.sd, self.ybar = tuple(beta), w, mu, sd, ybar

    def ref(self, m, hs, i1, i5):
        return m + (self.beta[0] * i1 + self.beta[1] * i5) * hs

    def pred(self, d, x):
        if self.w is None:
            return INF
        return self.ybar + sum(w * (f - u) / s for w, f, u, s in zip(self.w, feat(d, x), self.mu, self.sd))


def ref_xy(D, h_s, step=4):
    """Regressors (imb1*hs, imb5*hs) and target mid(t+h)-mid(t), every `step`-th snapshot."""
    t, bp, ap, bs, az = D['t'], D['bpx'], D['apx'], D['bsz'], D['asz']
    mid, hs = 0.5 * (bp[:, 0] + ap[:, 0]), 0.5 * (ap[:, 0] - bp[:, 0])
    i1 = (bs[:, 0] - az[:, 0]) / (bs[:, 0] + az[:, 0])
    sb, sa = np.nansum(bs[:, :5], 1), np.nansum(az[:, :5], 1)
    i5 = (sb - sa) / (sb + sa)
    k = np.searchsorted(t, t + int(h_s * 1000))
    ok = (k < len(t)) & (np.arange(len(t)) % step == 0)
    return np.c_[i1[ok] * hs[ok], i5[ok] * hs[ok]], mid[k[ok]] - mid[ok]


def fit_ref(xys):
    X = np.concatenate([a for a, _ in xys])
    y = np.concatenate([b for _, b in xys])
    return np.linalg.lstsq(X, y, rcond=None)[0] if len(y) > 10 else (0.0, 0.0)


def fit_markout(F, h, lam=1.0, min_n=200):
    """Weighted ridge of clipped net_h on feat(depth, x) over probe fills F (dict of arrays)."""
    y, w = F['net%d' % h], F['w']
    ok = np.isfinite(y)
    if ok.sum() < min_n:
        return None
    l = np.log1p(F['d'][ok])
    Xx = np.column_stack([F['x%d' % c][ok] for c in range(NX)])
    X = np.column_stack([l, Xx, l * Xx[:, 0]])  # = feat(d, x)
    y, w = y[ok], w[ok]
    lo, hi = np.percentile(y, [1, 99])
    y = np.clip(y, lo, hi)
    mu = np.average(X, 0, w)
    sd = np.sqrt(np.average((X - mu) ** 2, 0, w)) + 1e-9
    Z = (X - mu) / sd
    ybar = np.average(y, weights=w)
    A = Z.T @ (Z * w[:, None]) + lam * np.eye(Z.shape[1])
    beta = np.linalg.solve(A, Z.T @ (w * (y - ybar)))
    return tuple(beta), tuple(mu), tuple(sd), float(ybar)
