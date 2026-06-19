"""
Emergent self-propulsion: a three-component reaction-diffusion "dissipative
soliton" that moves on its own (Purwins / Bode / Liehr type).

    du/dt = Du Lap u + lam u - u^3 - k3 v - k4 w + k1   (+ noise)
    tv dv/dt = Dv Lap v + u - v          (fast, short/medium range inhibitor)
    tw dw/dt = Dw Lap w + u - w          (slow, long-range inhibitor)

A localized spot sits in its own inhibitor wells. When the slow inhibitor w is
slow enough, it cannot follow the spot: a tiny fluctuation makes the spot drift,
the lagging w pushes it to keep going -> a DRIFT bifurcation. The spot then
travels at constant speed in a spontaneously chosen direction. Nothing is moved
by hand -- the motion is a property of the local dynamics. Noise picks/wanders
the heading.

This file scans parameters for the moving regime (diagnostics only).
"""

import numpy as np


def lap(f):
    return (-4.0 * f
            + np.roll(f, 1, 0) + np.roll(f, -1, 0)
            + np.roll(f, 1, 1) + np.roll(f, -1, 1))


def run(p, nsteps, seed=0, record_every=0):
    rng = np.random.default_rng(seed)
    N = p["N"]
    Y, X = np.mgrid[0:N, 0:N].astype(float)
    cx0 = cy0 = N / 2
    r2 = (X - cx0) ** 2 + (Y - cy0) ** 2
    u = p["ubg"] + (p["uhi"] - p["ubg"]) * np.exp(-r2 / (2 * p["R"] ** 2))
    v = u.copy(); w = u.copy()

    dt = p["dt"]
    traj, snaps = [], []
    for s in range(nsteps):
        u = u + dt * (p["Du"] * lap(u) + p["lam"] * u - u ** 3
                      - p["k3"] * v - p["k4"] * w + p["k1"]) \
            + p["noise"] * np.sqrt(dt) * rng.standard_normal(u.shape)
        np.clip(u, -5.0, 5.0, out=u)
        v = v + (dt / p["tv"]) * (p["Dv"] * lap(v) + u - v)
        w = w + (dt / p["tw"]) * (p["Dw"] * lap(w) + u - w)

        if record_every and s % record_every == 0:
            m = np.maximum(u - p["ubg"], 0.0)
            tot = m.sum() + 1e-9
            cx = (m * X).sum() / tot; cy = (m * Y).sum() / tot
            traj.append((cx, cy))
            snaps.append((u.copy(), v.copy(), w.copy()))

    m = np.maximum(u - p["ubg"], 0.0)
    tot = m.sum() + 1e-9
    cx = (m * X).sum() / tot; cy = (m * Y).sum() / tot
    area = (u > 0.5 * (p["uhi"] + p["ubg"])).sum()
    disp = np.hypot(cx - cx0, cy - cy0)
    return dict(cx=cx, cy=cy, disp=disp, area=int(area),
                umax=float(u.max()), umin=float(u.min()),
                traj=traj, snaps=snaps)


BASE = dict(N=128, dt=0.01, Du=1.0, lam=2.0, k1=-0.2, k3=0.5, k4=0.5,
            Dv=8.0, tv=1.0, Dw=16.0, tw=1.0, ubg=-1.07, uhi=0.87, R=8.0,
            noise=0.005)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["scan"]:
        print(f"{'tw':>5} {'k4':>5} {'Dw':>5} | {'disp':>7} {'area':>6} "
              f"{'umax':>6} {'umin':>6}")
        for tw in [2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 15.0]:
            for k4 in [0.5]:
                p = dict(BASE); p["tw"] = tw; p["k4"] = k4
                r = run(p, nsteps=12000, seed=1)
                tag = "MOVES" if (r["area"] > 30 and r["disp"] > 5) else \
                      ("alive" if r["area"] > 30 else "dead")
                print(f"{tw:5.0f} {k4:5.1f} {p['Dw']:5.0f} | {r['disp']:7.1f} "
                      f"{r['area']:6d} {r['umax']:6.2f} {r['umin']:6.2f}  {tag}")
    else:
        # single config, time series -> first confirm a stable spot exists
        p = dict(BASE)
        r = run(p, nsteps=12000, seed=1, record_every=1500)
        print("step    cx     cy    disp   area   umax")
        for i, (cx, cy) in enumerate(r["traj"]):
            d = np.hypot(cx - p["N"] / 2, cy - p["N"] / 2)
            print(f"{i*1500:5d} {cx:6.1f} {cy:6.1f} {d:6.1f}")
        print("final:", {k: r[k] for k in ("disp", "area", "umax", "umin")})
