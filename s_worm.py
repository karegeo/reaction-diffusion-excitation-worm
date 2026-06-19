"""
An S-shaped (serpentine) worm that slithers.

Same three fields as worm.py (body phi, excitation u, recovery v) but with two
changes that make it both prettier and *more purely a set of PDEs*:

* the body is initialised as an S-shaped tube;
* the body is advected by a LOCAL velocity  V = c (-grad v)  -- every patch of
  the worm is pushed along the local direction the wave is travelling. There is
  no global "read out the heading" step any more; the coupling is local:

      dphi/dt = Dphi lap(phi) + cohesion + c (grad v . grad phi)

  Because the excitation runs head-ward along the curved body, each segment is
  pushed along the local tangent -> the worm slithers along its own S-curve.

Optional `noise` adds a stochastic kick to the excitation; turn the pacemaker
off and a large enough noise will *spontaneously* nucleate the waves itself.

Run:  python3 s_worm.py        # writes worm_s_shape.gif
"""

import numpy as np
from worm import lap9, grad, compose_rgb


def build_S(Nx, Ny, amp=34.0, x0=45, x1=210, radius=10.0, periods=1.5):
    """An S / serpentine tube: disks strung along a sinusoidal centreline."""
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    X = X.astype(float); Y = Y.astype(float)
    ts = np.linspace(0, 1, 400)
    cxs = x0 + ts * (x1 - x0)
    cys = Ny / 2 + amp * np.sin(2 * np.pi * periods * ts)
    d = np.full((Ny, Nx), 1e9)
    for cx, cy in zip(cxs, cys):
        d = np.minimum(d, np.hypot(X - cx, Y - cy))
    phi = 0.5 * (1.0 - np.tanh((d - radius) / 1.3))
    return X, Y, phi, (cxs[0], cys[0])      # also return the tail end


class S:
    Nx, Ny = 430, 170
    dt = 0.02
    nsteps = 9000
    a, b, eps = 0.75, 0.01, 0.02
    Du = 1.0
    k_leak = 3.0
    Dphi = 0.4
    tau = 0.8
    mu = 0.6
    v_crawl = 1.2          # glide speed (translation along the global heading)
    lp = 0.05              # heading smoothing
    act_ref = None
    pace_every = 700
    pace_radius = 7
    rear_frac = 0.16       # pacemaker fires in the rear-most slice of the body
    noise = 0.0            # stochastic kick to the excitation (try ~0.05)


def simulate_s(p, record_every=60):
    X, Y, phi, _ = build_S(p.Nx, p.Ny, amp=30, x0=35, x1=185,
                           radius=10.0, periods=1.5)
    A0 = phi.sum()
    u = np.zeros_like(phi); v = np.zeros_like(phi)
    body0 = phi > 0.5
    sx = X[body0].min(); sy = Y[body0][np.argmin(X[body0])]
    u[np.hypot(X - sx, Y - sy) < 7] = 1.0          # seed pulse at the tail
    Phx, Phy = 1.0, 0.0
    rng = np.random.default_rng(0)

    frames = []
    for step in range(p.nsteps):
        thr = (v + p.b) / p.a
        fu = (u * (1.0 - u) * (u - thr)) / p.eps
        du = p.Du * lap9(u) + phi * fu - p.k_leak * (1.0 - phi) * u
        if p.noise:
            du = du + p.noise * rng.standard_normal(u.shape) * phi / np.sqrt(p.dt)
        u = np.clip(u + p.dt * du, 0.0, 1.0)
        v = v + p.dt * (phi * (u - v))

        # pacemaker at the current rear of the worm (rear-most along heading)
        if p.pace_every and step % p.pace_every == 0:
            body = phi > 0.5
            if body.any():
                proj = X * Phx + Y * Phy
                pmin = proj[body].min(); span = proj[body].max() - pmin + 1e-9
                rear = body & (proj < pmin + p.rear_frac * span)
                if rear.any():
                    ry, rx = np.argwhere(rear).mean(axis=0)
                    u[(np.hypot(X - rx, Y - ry) < p.pace_radius) & (phi > 0.3)] = 1.0

        # global heading P = <-grad v> over the worm; rigid translation by it
        vx, vy = grad(v)
        wgt = phi * u
        activity = wgt.sum()
        if p.act_ref is None and activity > 0:
            p.act_ref = max(activity, 1.0)
        if activity > 1e-6:
            tx, ty = -(wgt * vx).sum(), -(wgt * vy).sum()
            n = np.hypot(tx, ty) + 1e-9
            Phx += p.lp * (tx / n - Phx)
            Phy += p.lp * (ty / n - Phy)
            Phy *= 0.97
            n2 = np.hypot(Phx, Phy) + 1e-9
            Phx, Phy = Phx / n2, Phy / n2
        go = 0.0 if not p.act_ref else min(activity / p.act_ref, 1.0)

        gx, gy = grad(phi)
        A = phi.sum()
        cohesion = (phi * (1 - phi) * (phi - 0.5 + p.mu * (A0 - A) / A0)) / p.tau
        Vx, Vy = p.v_crawl * go * Phx, p.v_crawl * go * Phy
        adv = -(Vx * gx + Vy * gy)
        phi = np.clip(phi + p.dt * (p.Dphi * lap9(phi) + cohesion + adv), 0, 1)

        if record_every and step % record_every == 0:
            cx = (phi * X).sum() / phi.sum()
            frames.append((phi.copy(), u.copy(), v.copy(), cx))
    return frames, A0, phi.sum()


def render(frames, fname, title, colors=96):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(6.6, 3.7), dpi=110)
    fig.patch.set_facecolor("#0d0e12"); ax.set_facecolor("#0d0e12")
    im = ax.imshow(compose_rgb(*frames[0][:3]), origin="lower",
                   interpolation="bilinear", animated=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color="#aeb6c2", fontsize=8.5)
    anim = FuncAnimation(fig, lambda i: (im.set_array(compose_rgb(*frames[i][:3])),),
                         frames=len(frames), interval=60, blit=False)
    anim.save(fname, writer=PillowWriter(fps=18))
    plt.close(fig)

    from PIL import Image, ImageSequence
    im2 = Image.open(fname)
    fr = [f.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT)
          for f in ImageSequence.Iterator(im2)]
    fr[0].save(fname, save_all=True, append_images=fr[1:], loop=0,
               duration=60, optimize=True)
    import os
    print(f"wrote {fname} ({os.path.getsize(fname)/1e6:.2f} MB)")


def main(scenario="slither"):
    import time
    p = S()
    if scenario == "noise":
        # no pacemaker, no crawl: show that random noise alone ignites the waves
        p.pace_every = 0
        p.noise = 0.05
        p.v_crawl = 0.0
        title = ("no pacemaker -- random noise alone keeps igniting the "
                 "excitable waves on the body")
        fname = "worm_noise.gif"
    else:
        title = ("S-shaped worm gliding along its heading -- the wave runs "
                 "head-ward, refractory tail in magenta")
        fname = "worm_s_shape.gif"

    t0 = time.time()
    frames, A0, A = simulate_s(p, record_every=90)
    dx = frames[-1][3] - frames[0][3]
    print(f"sim {time.time()-t0:.1f}s  area {A0:.0f}->{A:.0f}  "
          f"frames={len(frames)}  centre-of-mass dx={dx:+.1f}")
    render(frames, fname, title)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "slither")
