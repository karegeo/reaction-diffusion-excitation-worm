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


def lap(f):                                   # periodic (torus): no walls to hug
    return (-4.0 * f + np.roll(f, 1, 0) + np.roll(f, -1, 0)
            + np.roll(f, 1, 1) + np.roll(f, -1, 1))


def grad(f):
    fx = 0.5 * (np.roll(f, -1, 1) - np.roll(f, 1, 1))
    fy = 0.5 * (np.roll(f, -1, 0) - np.roll(f, 1, 0))
    return fx, fy


class Cfg:
    N = 200
    dt = 0.1
    Dphi = 0.6
    tau = 0.9
    mu = 1.5           # area (mass) restoring
    Dc = 1.5
    beta = 0.7         # repellent secretion
    gamma = 0.035      # repellent decay (1/gamma = trail memory)
    chi = 3.6          # propulsion (chemotactic repulsion from own trail)
    noise = 0.02
    R0 = 12.0
    bgain = 0.5        # body deposition (the permanent, growing snake body)
    kappa_b = 0.35     # gentle long-term self-avoidance (c does the propelling)


def simulate(c, nsteps, record_every=0, rseed=0):
    rng = np.random.default_rng(rseed)
    N = c.N
    Y, X = np.mgrid[0:N, 0:N].astype(float)
    r2 = (X - N / 2) ** 2 + (Y - N / 2) ** 2
    phi = 0.5 * (1 - np.tanh((np.sqrt(r2) - c.R0) / 2.0))
    cc = np.zeros_like(phi)
    bb = np.zeros_like(phi)            # visible body trail (the snake's body)
    A0 = phi.sum()

    def centroid():                                   # circular mean (periodic)
        ax = np.angle((phi * np.exp(2j * np.pi * X / N)).sum())
        ay = np.angle((phi * np.exp(2j * np.pi * Y / N)).sum())
        return (ax % (2 * np.pi)) * N / (2 * np.pi), (ay % (2 * np.pi)) * N / (2 * np.pi)

    frames, traj = [], []
    for s in range(nsteps):
        cx, cy = grad(cc + c.kappa_b * bb)        # flee secreted chemical AND own body
        cross = c.chi * (grad(phi * cx)[0] + grad(phi * cy)[1])
        A = phi.sum()
        cohesion = (phi * (1 - phi) * (phi - 0.5 + c.mu * (A0 - A) / A0)) / c.tau
        phi = phi + c.dt * (c.Dphi * lap(phi) + cohesion + cross) \
            + c.noise * np.sqrt(c.dt) * rng.standard_normal(phi.shape) * phi
        np.clip(phi, 0.0, 1.0, out=phi)
        cc = cc + c.dt * (c.Dc * lap(cc) + c.beta * phi - c.gamma * cc)
        bb = bb + c.dt * c.bgain * np.maximum(phi - 0.5, 0.0) * (1.0 - bb)  # permanent, saturating

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
        body = np.clip((b / bmax - 0.12) / 0.4, 0, 1)[..., None]   # crisp snake body
        img = img * (1 - body) + np.array([0.12, 0.66, 0.74]) * body
        core = np.clip((phi - 0.5) / 0.5, 0, 1)[..., None]         # bright head
        img = img * (1 - core) + np.array([0.90, 0.98, 1.0]) * core
        return np.clip(img, 0, 1)

    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=110)
    fig.patch.set_facecolor("#0d0e12")
    im = ax.imshow(rgb(*frames[0]), origin="lower", interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("artificial life: a self-propelled head (it flees its own "
                 "secreted trail) grows a body & wanders — nothing steered by hand",
                 color="#aeb6c2", fontsize=7.0)

    def update(i):
        im.set_array(rgb(*frames[i]))
        return (im,)

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
    frames, traj, A0, A = simulate(c, nsteps=8000, record_every=80)
    tr = np.array(traj)
    path = np.sum(np.hypot(*np.diff(tr, axis=0).T))
    print(f"area {A0:.0f}->{A:.0f} ({100*(A-A0)/A0:+.1f}%)  "
          f"path length={path:.0f}  frames={len(frames)}")
    render(c, frames, traj)
