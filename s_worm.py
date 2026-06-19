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
    Nx, Ny = 300, 170
    dt = 0.02
    nsteps = 9000
    a, b, eps = 0.75, 0.01, 0.02
    Du = 1.0
    k_leak = 3.0
    Dphi = 0.4
    tau = 0.8
    mu = 0.6
    c_local = 4.0          # local advection strength (slither speed)
    pace_every = 850
    pace_radius = 7
    noise = 0.0            # stochastic kick to the excitation (try ~0.05)


def simulate_s(p, record_every=60):
    X, Y, phi, (tail_x, tail_y) = build_S(p.Nx, p.Ny)
    A0 = phi.sum()
    u = np.zeros_like(phi); v = np.zeros_like(phi)
    seed = np.hypot(X - tail_x, Y - tail_y) < 7
    u[seed] = 1.0
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

        # pacemaker at the tail end of the S
        if p.pace_every and step % p.pace_every == 0:
            tip = np.hypot(X - tail_x, Y - tail_y) < p.pace_radius
            u[tip & (phi > 0.3)] = 1.0

        # LOCAL advection in conservative (flux) form so AREA is preserved:
        #   V = c(-grad v),   dphi/dt += -div(phi V)
        vx, vy = grad(v)
        Vx, Vy = -p.c_local * vx, -p.c_local * vy
        adv = -(grad(phi * Vx)[0] + grad(phi * Vy)[1])
        A = phi.sum()
        cohesion = (phi * (1 - phi) * (phi - 0.5 + p.mu * (A0 - A) / A0)) / p.tau
        phi = np.clip(phi + p.dt * (p.Dphi * lap9(phi) + cohesion + adv), 0, 1)

        if record_every and step % record_every == 0:
            frames.append((phi.copy(), u.copy(), v.copy()))
    return frames, A0, phi.sum()


def render(frames, fname, title, colors=96):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(6.6, 3.7), dpi=110)
    fig.patch.set_facecolor("#0d0e12"); ax.set_facecolor("#0d0e12")
    im = ax.imshow(compose_rgb(*frames[0]), origin="lower",
                   interpolation="bilinear", animated=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color="#aeb6c2", fontsize=8.5)
    anim = FuncAnimation(fig, lambda i: (im.set_array(compose_rgb(*frames[i])),),
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
        # no pacemaker: random noise alone spontaneously nucleates the waves
        p.pace_every = 0
        p.noise = 0.05
        title = ("no pacemaker -- random noise alone keeps igniting the "
                 "excitable waves (then the body creeps after them)")
        fname = "worm_noise.gif"
    else:
        title = ("S-shaped worm: the excitation wave runs head-ward along the "
                 "curve, leaving a refractory tail")
        fname = "worm_s_shape.gif"

    t0 = time.time()
    frames, A0, A = simulate_s(p, record_every=90)
    print(f"sim {time.time()-t0:.1f}s  area {A0:.0f}->{A:.0f}  frames={len(frames)}")
    render(frames, fname, title)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "slither")
