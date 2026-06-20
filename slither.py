"""
A genuinely *slithering* S-shaped worm (undulatory swimmer).

Why this is separate from worm.py / s_worm.py
----------------------------------------------
A body that merely glides (s_worm.py) translates rigidly. Real slithering is
different: a *travelling wave of body curvature* (the muscle/CPG activation that
the excitable layer stands for) plus **anisotropic drag** (the body slips along
its length far more easily than sideways) produces net forward thrust. Without
the drag anisotropy a free undulating body cannot move at all -- the forward and
backward pushes cancel exactly. With it, the worm swims.

Model (resistive-force theory, the standard low-Reynolds swimmer):
  * the body shape is set by a travelling curvature wave  kappa(s,t)
    -- this is the excitation wave running head->tail, here as the muscle signal;
  * each segment feels drag  f = -(xi_par (v.t)t + xi_perp (v.n)n),  xi_perp > xi_par;
  * the worm is force- and torque-free, so its rigid motion (U, Omega) each
    instant solves a 3x3 linear balance.  Integrate -> it slithers.

Run:  python3 slither.py        # writes worm_slither.gif
"""

import numpy as np


class Cfg:
    N = 110                 # body nodes
    L = 170.0               # body length
    kappa_amp = 0.055       # curvature-wave amplitude (how hard it bends)
    n_waves = 1.6           # number of body waves (the S/serpentine shape)
    period = 34.0           # temporal period of the muscle wave
    xi_perp = 6.0           # normal drag  (sideways: hard to move)
    xi_par = 1.0            # tangential drag (lengthwise: easy to slip)  -> anisotropy
    dt = 0.1
    nsteps = 1700
    radius = 8.0


def body_shape(c, t):
    """Body-frame node positions from the travelling curvature wave, centred."""
    s = np.linspace(0.0, 1.0, c.N)
    ds = c.L / (c.N - 1)
    kappa = c.kappa_amp * np.sin(2 * np.pi * (c.n_waves * s) + 2 * np.pi * t / c.period)
    kappa -= kappa.mean()             # zero net bend -> swims straight, no drift-turn
    psi = np.concatenate([[0.0], np.cumsum(kappa[:-1]) * ds])   # tangent angle
    x = np.concatenate([[0.0], np.cumsum(np.cos(psi[:-1])) * ds])
    y = np.concatenate([[0.0], np.cumsum(np.sin(psi[:-1])) * ds])
    X = np.stack([x, y], axis=1)
    X -= X.mean(axis=0)                                         # origin at centroid
    act = np.sin(2 * np.pi * (c.n_waves * s) - 2 * np.pi * t / c.period)
    return X, act


def simulate(c, record_every=12):
    alpha = 0.0                       # body orientation
    Xc = np.array([0.0, 0.0])         # centroid (lab)
    Xb_prev, _ = body_shape(c, -c.dt)

    frames = []
    traj = []
    for step in range(c.nsteps):
        t = step * c.dt
        Xb, act = body_shape(c, t)
        ub = (Xb - Xb_prev) / c.dt                 # deformation velocity (body frame)
        Xb_prev = Xb

        ca, sa = np.cos(alpha), np.sin(alpha)
        R = np.array([[ca, -sa], [sa, ca]])
        r = Xb @ R.T                               # lab positions rel. to centroid
        w = ub @ R.T                               # lab deformation velocity
        P = np.stack([-r[:, 1], r[:, 0]], axis=1)  # perp(r), the rotation lever

        # lab tangent / normal at each node
        tang = np.gradient(r, axis=0)
        tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)
        norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)

        def Dapply(vec):                           # anisotropic drag tensor * vec
            return (c.xi_par * np.sum(vec * tang, axis=1)[:, None] * tang
                    + c.xi_perp * np.sum(vec * norm, axis=1)[:, None] * norm)

        Dw, DP = Dapply(w), Dapply(P)
        SumD = (c.xi_par * tang.T @ tang) + (c.xi_perp * norm.T @ norm)   # 2x2
        M = np.zeros((3, 3))
        M[:2, :2] = SumD
        M[:2, 2] = DP.sum(axis=0)
        M[2, :2] = DP.sum(axis=0)
        M[2, 2] = np.sum(P * DP)
        b = np.zeros(3)
        b[:2] = -Dw.sum(axis=0)
        b[2] = -np.sum(P * Dw)
        Ux, Uy, Om = np.linalg.solve(M, b)

        Xc = Xc + np.array([Ux, Uy]) * c.dt
        alpha = alpha + Om * c.dt

        if step % record_every == 0:
            frames.append((Xc + r, act.copy()))
            traj.append(Xc.copy())
    return frames, np.array(traj)


def render(c, frames, fname="worm_slither.gif"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon

    allnodes = np.concatenate([f[0] for f in frames])
    x0, x1 = allnodes[:, 0].min() - 20, allnodes[:, 0].max() + 20
    y0, y1 = allnodes[:, 1].min() - 20, allnodes[:, 1].max() + 20

    fig, ax = plt.subplots(figsize=(8.0, 8.0 * (y1 - y0) / (x1 - x0)), dpi=110)
    fig.patch.set_facecolor("#0d0e12"); ax.set_facecolor("#0d0e12")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("undulatory slither: a travelling curvature wave + anisotropic "
                 "drag = net forward swimming", color="#aeb6c2", fontsize=9)

    tube = Polygon(frames[0][0], closed=True, facecolor="#11616f",
                   edgecolor="none", zorder=1)
    ax.add_patch(tube)
    lc = LineCollection([], cmap="magma", zorder=2, linewidth=4)
    lc.set_clim(0, 1); ax.add_collection(lc)
    (trail,) = ax.plot([], [], color="#5fd3e6", lw=1.0, alpha=0.6, zorder=0)
    cxs, cys = [], []

    def tube_xy(nodes):
        tg = np.gradient(nodes, axis=0)
        tg /= (np.linalg.norm(tg, axis=1, keepdims=True) + 1e-12)
        nm = np.stack([-tg[:, 1], tg[:, 0]], axis=1)
        return np.vstack([nodes + c.radius * nm, (nodes - c.radius * nm)[::-1]])

    def update(i):
        nodes, act = frames[i]
        tube.set_xy(tube_xy(nodes))
        segs = np.stack([nodes[:-1], nodes[1:]], axis=1)
        lc.set_segments(segs)
        lc.set_array(0.5 * (act[:-1] + act[1:]) * 0.5 + 0.5)
        cx, cy = nodes.mean(axis=0)
        cxs.append(cx); cys.append(cy); trail.set_data(cxs, cys)
        return tube, lc, trail

    anim = FuncAnimation(fig, update, frames=len(frames), interval=55, blit=False)
    anim.save(fname, writer=PillowWriter(fps=20))
    plt.close(fig)
    from PIL import Image, ImageSequence
    im = Image.open(fname)
    fr = [f.convert("RGB").quantize(colors=96, method=Image.MEDIANCUT)
          for f in ImageSequence.Iterator(im)]
    fr[0].save(fname, save_all=True, append_images=fr[1:], loop=0,
               duration=55, optimize=True)
    import os
    print(f"wrote {fname} ({os.path.getsize(fname)/1e6:.2f} MB)")


if __name__ == "__main__":
    c = Cfg()
    frames, traj = simulate(c)
    disp = traj[-1] - traj[0]
    # mass check: the body never grows or decays. Total length (= mass, for a
    # constant cross-section) must stay fixed -- the motion is pure transport.
    lengths = [np.sum(np.linalg.norm(np.diff(n, axis=0), axis=1)) for n, _ in frames]
    print(f"frames={len(frames)}  net displacement = "
          f"({disp[0]:+.1f}, {disp[1]:+.1f})  |d|={np.hypot(*disp):.1f}  "
          f"(= {np.hypot(*disp)/c.L:.2f} body-lengths)")
    print(f"body length (mass) over run: min={min(lengths):.2f} "
          f"max={max(lengths):.2f}  -> change {100*(max(lengths)-min(lengths))/c.L:.3f}%")
    render(c, frames)
