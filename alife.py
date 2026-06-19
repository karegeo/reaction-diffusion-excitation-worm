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


BASE = dict(N=128, dt=0.006, Du=1.0, lam=6.0, k1=-0.8, k3=1.0, k4=1.0,
            Dv=3.0, tv=1.0, Dw=30.0, tw=1.0, ubg=-2.1, uhi=1.8, R=8.0,
            noise=0.006)


def simulate_worm(p, nsteps, record_every=40, body_decay=0.004, body_gain=0.02):
    """Run the self-propelled soliton and let it paint a slowly-fading BODY
    trail (b). The head (u) moves by itself (drift bifurcation); the body is
    just a memory of where the emergent head has wandered -> a living worm."""
    rng = np.random.default_rng(p.get("rseed", 3))
    N = p["N"]
    Y, X = np.mgrid[0:N, 0:N].astype(float)
    r2 = (X - N / 2) ** 2 + (Y - N / 2) ** 2
    u = p["ubg"] + (p["uhi"] - p["ubg"]) * np.exp(-r2 / (2 * p["R"] ** 2))
    v = u.copy(); w = u.copy(); b = np.zeros_like(u)
    thr = 0.5 * (p["uhi"] + p["ubg"])
    dt = p["dt"]
    frames, trail = [], []
    for s in range(nsteps):
        u = u + dt * (p["Du"] * lap(u) + p["lam"] * u - u ** 3
                      - p["k3"] * v - p["k4"] * w + p["k1"]) \
            + p["noise"] * np.sqrt(dt) * rng.standard_normal(u.shape)
        np.clip(u, -5.0, 5.0, out=u)
        v = v + (dt / p["tv"]) * (p["Dv"] * lap(v) + u - v)
        w = w + (dt / p["tw"]) * (p["Dw"] * lap(w) + u - w)
        b = b + body_gain * np.maximum(u - thr, 0.0) - body_decay * b
        if record_every and s % record_every == 0:
            m = np.maximum(u - p["ubg"], 0.0); tot = m.sum() + 1e-9
            trail.append(((m * X).sum() / tot, (m * Y).sum() / tot))
            frames.append((u.copy(), b.copy()))
    return frames, trail


def render_worm(p, frames, trail, fname="worm_alive.gif"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    thr = 0.5 * (p["uhi"] + p["ubg"])
    bmax = max(f[1].max() for f in frames) + 1e-9

    def rgb(u, b):
        bg = np.array([0.05, 0.06, 0.09]); body = np.array([0.10, 0.45, 0.55])
        img = np.broadcast_to(bg, u.shape + (3,)).copy()
        a = np.clip(b / bmax, 0, 1)[..., None]
        img = img * (1 - a) + body * a
        hot = np.clip((u - thr) / (p["uhi"] - thr + 1e-9), 0, 1)
        h = np.stack([np.clip(1.3 * hot, 0, 1), np.clip(1.5 * hot - .4, 0, 1),
                      np.clip(2.0 * hot - 1.2, 0, 1)], -1)
        ah = hot[..., None]
        return np.clip(img * (1 - ah) + h * ah, 0, 1)

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
    fig.patch.set_facecolor("#0d0e12")
    im = ax.imshow(rgb(*frames[0]), origin="lower", interpolation="bilinear")
    (tr,) = ax.plot([], [], color="#7fe7ff", lw=0.8, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("emergent crawl: the spot propels ITSELF (drift bifurcation); "
                 "noise wanders the heading", color="#aeb6c2", fontsize=8)
    xs, ys = [], []

    def update(i):
        im.set_array(rgb(*frames[i]))
        xs.append(trail[i][0]); ys.append(trail[i][1]); tr.set_data(xs, ys)
        return im, tr

    anim = FuncAnimation(fig, update, frames=len(frames), interval=55)
    anim.save(fname, writer=PillowWriter(fps=20)); plt.close(fig)
    from PIL import Image, ImageSequence
    im2 = Image.open(fname)
    fr = [f.convert("RGB").quantize(colors=96, method=Image.MEDIANCUT)
          for f in ImageSequence.Iterator(im2)]
    fr[0].save(fname, save_all=True, append_images=fr[1:], loop=0,
               duration=55, optimize=True)
    import os
    print(f"wrote {fname} ({os.path.getsize(fname)/1e6:.2f} MB)")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["scan"]:
        print(f"{'tw':>5} {'k4':>5} {'Dw':>5} | {'disp':>7} {'area':>6} "
              f"{'umax':>6} {'umin':>6}")
        # drift comes from a SLOW LOCALIZING inhibitor: the spot rolls out of
        # its own lagging well. Make v slow (large tv); w off (k4=0).
        print(f"{'tv':>5} {'Dv':>5} {'k3':>5} | {'disp':>7} {'area':>6} "
              f"{'umax':>6} {'umin':>6}")
        # robust deep-bistable spot + TIGHT slow localizer v (roll-out -> drift),
        # broad fast w for global stability. Scan how slow v is.
        print(f"{'tv':>5} {'Dv':>5} {'Dw':>5} | {'disp':>7} {'area':>6} "
              f"{'umax':>6} {'umin':>6}")
        for tv in [1.0, 6.0, 12.0, 20.0, 32.0, 48.0]:
            p = dict(BASE); p["N"] = 80; p["tv"] = tv
            r = run(p, nsteps=14000, seed=1)
            tag = "MOVES" if (r["area"] > 30 and r["disp"] > 5) else \
                  ("alive" if r["area"] > 30 else "dead")
            print(f"{tv:5.0f} {p['Dv']:5.0f} {p['Dw']:5.0f} | {r['disp']:7.1f} "
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
