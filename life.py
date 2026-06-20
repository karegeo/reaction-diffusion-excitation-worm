"""
Artificial-life worm: EMERGENT self-propulsion, nothing moved by hand.

A cohesive droplet (phase field phi) secretes a slowly-decaying repellent c.
It is pushed down its own c-gradient -- written as a reaction-diffusion
CROSS-DIFFUSION term, +chi div(phi grad c), not an imposed velocity:

    dphi/dt = Dphi lap phi + cohesion/area + chi div(phi grad c) + noise
    dc/dt   = Dc lap c + beta phi - gamma c          (the secreted trail)

Why it moves (emergently):
  * A symmetric droplet builds a symmetric c bump -> no net push. But that state
    is UNSTABLE: a tiny noise fluctuation shifts the droplet, the trail it
    leaves behind raises c there, and grad c then pushes it further the same
    way. Symmetry breaks spontaneously and it self-propels.
  * It cannot return where it has just been (high c there) -- exactly the
    "refractory tail forbids going backward" idea, now producing locomotion.
  * Noise keeps the heading wandering, like something alive.
Mass is ~conserved: the cross-diffusion term is a flux divergence (moves phi,
doesn't create it); the area term only counters phase-field shrinkage.
"""

import numpy as np


def lap(f):                                   # zero-flux (Neumann) boundaries
    fp = np.pad(f, 1, mode="edge")
    return (fp[:-2, 1:-1] + fp[2:, 1:-1] + fp[1:-1, :-2] + fp[1:-1, 2:]
            - 4.0 * f)


def grad(f):
    fp = np.pad(f, 1, mode="edge")
    fx = 0.5 * (fp[1:-1, 2:] - fp[1:-1, :-2])
    fy = 0.5 * (fp[2:, 1:-1] - fp[:-2, 1:-1])
    return fx, fy


class Cfg:
    N = 160
    dt = 0.1
    Dphi = 0.6
    tau = 0.9
    mu = 1.5           # area (mass) restoring
    Dc = 1.5
    beta = 0.7         # repellent secretion
    gamma = 0.015      # repellent decay (1/gamma = trail memory)
    chi = 3.6          # propulsion (chemotactic repulsion from own trail)
    noise = 0.02
    R0 = 12.0
    bgain = 0.06       # body-trail deposition (the visible snake body)
    bdecay = 0.022     # body-trail decay (sets snake length)


def simulate(c, nsteps, record_every=0, rseed=0):
    rng = np.random.default_rng(rseed)
    N = c.N
    Y, X = np.mgrid[0:N, 0:N].astype(float)
    r2 = (X - N / 2) ** 2 + (Y - N / 2) ** 2
    phi = 0.5 * (1 - np.tanh((np.sqrt(r2) - c.R0) / 2.0))
    cc = np.zeros_like(phi)
    bb = np.zeros_like(phi)            # visible body trail (the snake's body)
    A0 = phi.sum()

    def centroid():
        tot = phi.sum() + 1e-9
        return (phi * X).sum() / tot, (phi * Y).sum() / tot

    frames, traj = [], []
    for s in range(nsteps):
        cx, cy = grad(cc)
        cross = c.chi * (grad(phi * cx)[0] + grad(phi * cy)[1])   # +chi div(phi grad c)
        A = phi.sum()
        cohesion = (phi * (1 - phi) * (phi - 0.5 + c.mu * (A0 - A) / A0)) / c.tau
        phi = phi + c.dt * (c.Dphi * lap(phi) + cohesion + cross) \
            + c.noise * np.sqrt(c.dt) * rng.standard_normal(phi.shape) * phi
        np.clip(phi, 0.0, 1.0, out=phi)
        cc = cc + c.dt * (c.Dc * lap(cc) + c.beta * phi - c.gamma * cc)
        bb = bb + c.dt * (c.bgain * np.maximum(phi - 0.5, 0.0) - c.bdecay * bb)

        if record_every and s % record_every == 0:
            frames.append((phi.copy(), bb.copy()))
            traj.append(centroid())
    return frames, traj, A0, phi.sum()


def render(cfg, frames, traj, fname="worm_alive.gif"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    bmax = max(f[1].max() for f in frames) + 1e-9

    def rgb(phi, b):
        bg = np.array([0.05, 0.06, 0.09])
        img = np.broadcast_to(bg, phi.shape + (3,)).copy()
        body = (np.clip(b / bmax, 0, 1) ** 0.5)[..., None]         # snake body (trail)
        img = img * (1 - body) + np.array([0.12, 0.64, 0.72]) * body
        core = np.clip(phi, 0, 1)[..., None]                       # bright head
        img = img * (1 - core) + np.array([0.88, 0.98, 1.0]) * core
        return np.clip(img, 0, 1)

    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=110)
    fig.patch.set_facecolor("#0d0e12")
    im = ax.imshow(rgb(*frames[0]), origin="lower", interpolation="bilinear")
    (tr,) = ax.plot([], [], color="#8fefff", lw=0.9, alpha=0.55)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("artificial life: the droplet flees its own secreted trail; "
                 "noise wanders the heading — nothing steered by hand",
                 color="#aeb6c2", fontsize=7.6)
    xs, ys = [], []

    def update(i):
        im.set_array(rgb(*frames[i]))
        xs.append(traj[i][0]); ys.append(traj[i][1]); tr.set_data(xs, ys)
        return im, tr

    anim = FuncAnimation(fig, update, frames=len(frames), interval=55)
    anim.save(fname, writer=PillowWriter(fps=20)); plt.close(fig)
    from PIL import Image, ImageSequence
    im2 = Image.open(fname)
    fr = [f.convert("RGB").quantize(colors=80, method=Image.MEDIANCUT)
          for f in ImageSequence.Iterator(im2)]
    fr[0].save(fname, save_all=True, append_images=fr[1:], loop=0,
               duration=55, optimize=True)
    import os
    print(f"wrote {fname} ({os.path.getsize(fname)/1e6:.2f} MB)")


if __name__ == "__main__":
    c = Cfg()
    frames, traj, A0, A = simulate(c, nsteps=9000, record_every=90)
    tr = np.array(traj)
    path = np.sum(np.hypot(*np.diff(tr, axis=0).T))
    print(f"area {A0:.0f}->{A:.0f} ({100*(A-A0)/A0:+.1f}%)  "
          f"path length={path:.0f}  frames={len(frames)}")
    render(c, frames, traj)
