"""
A worm that *crawls in one direction* using reaction-diffusion + an excitable layer.

Three coupled 2D fields:

    phi(x,y)  -- the worm BODY (a phase field, ~1 inside the worm, ~0 outside)
    u(x,y)    -- EXCITATION  (fast activator, FitzHugh-Nagumo / Barkley kinetics)
    v(x,y)    -- RECOVERY    (slow inhibitor -> the refractory tail)

Why this crawls instead of growing isotropically or drifting sideways
---------------------------------------------------------------------
* The excitable layer (u,v) supports a travelling PULSE with a refractory
  tail (high v). Refractoriness means the pulse cannot propagate back into
  tissue it just crossed -> once a direction is picked it is LOCKED IN.
  That is the symmetry-breaking the plain reaction-diffusion worm was missing.
* The excitation is CONFINED to the body: its kinetics are gated by phi and it
  leaks away outside, so the worm is a self-organising excitable waveguide.
* A pacemaker at the worm's TAIL fires periodic pulses head-ward (like a
  central pattern generator / the oscillatory chemistry of a BZ gel).
* The pulse defines a POLARITY  P = <-grad v>  (points the way the wave goes).
  The body then grows at the head and retracts at the tail along P
  -> the centre of mass translocates: the worm crawls.

Run:  python3 worm.py            # writes worm_crawl.gif and com.png
"""

import numpy as np

# ----------------------------------------------------------------------------
# Differential operators (finite differences, zero-flux / Neumann boundaries)
# ----------------------------------------------------------------------------
def lap9(f):
    """Isotropic 9-point Laplacian (h=1). Less grid anisotropy for the waves."""
    fp = np.pad(f, 1, mode="edge")
    N = fp[:-2, 1:-1]; S = fp[2:, 1:-1]; E = fp[1:-1, 2:]; W = fp[1:-1, :-2]
    NE = fp[:-2, 2:]; NW = fp[:-2, :-2]; SE = fp[2:, 2:]; SW = fp[2:, :-2]
    C = fp[1:-1, 1:-1]
    return (4.0 * (N + S + E + W) + (NE + NW + SE + SW) - 20.0 * C) / 6.0


def grad(f):
    """Central-difference gradient. Returns (d/dx, d/dy) = (cols, rows)."""
    fp = np.pad(f, 1, mode="edge")
    fx = (fp[1:-1, 2:] - fp[1:-1, :-2]) * 0.5
    fy = (fp[2:, 1:-1] - fp[:-2, 1:-1]) * 0.5
    return fx, fy


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def capsule_field(X, Y, x0, x1, yc, radius, sharp=1.2):
    """Smooth indicator (~1 inside, ~0 outside) of a horizontal capsule
    (a segment from (x0,yc) to (x1,yc) thickened by `radius`)."""
    t = np.clip((X - x0) / (x1 - x0 + 1e-9), 0.0, 1.0)
    px = x0 + t * (x1 - x0)
    d = np.hypot(X - px, Y - yc)            # distance to the segment
    return 0.5 * (1.0 - np.tanh((d - radius) / sharp))


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
class P:
    Nx, Ny = 380, 120
    dt = 0.02
    nsteps = 8000

    # Barkley excitable kinetics:  fu = u(1-u)(u-(v+b)/a)/eps ,  fv = u-v
    a, b, eps = 0.75, 0.01, 0.02
    Du = 1.0            # excitation diffusion (the wave travels)
    k_leak = 3.0        # how fast excitation dies OUTSIDE the body (confinement)

    # Body (phase field) dynamics
    Dphi = 0.4          # line tension / interface width (low -> slow rounding)
    tau = 0.8           # cohesion timescale (keeps phi ~0 or ~1, sharp edge)
    mu = 0.6            # volume (area) conservation strength
    v_crawl = 1.1       # crawl speed per unit excitation activity

    # Polarity / drive
    lp = 0.05           # low-pass smoothing of the polarity direction
    act_ref = None      # reference "wave activity" (set from first pulse)

    # Pacemaker (tail)
    pace_every = 850    # steps between pulses (one clean pulse at a time)
    pace_radius = 7
    rear_frac = 0.14    # inject in the rear-most this-fraction of the body


def build_initial(p):
    X, Y = np.meshgrid(np.arange(p.Nx), np.arange(p.Ny))
    X = X.astype(float); Y = Y.astype(float)
    yc = p.Ny / 2.0
    phi = capsule_field(X, Y, x0=35, x1=120, yc=yc, radius=11.0)
    u = np.zeros_like(phi)
    v = np.zeros_like(phi)
    # symmetry-breaking kick at the LEFT end -> first pulse heads +x
    seed = (np.hypot(X - 43, Y - yc) < 7)
    u[seed] = 1.0
    return X, Y, phi, u, v


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def simulate(p, record_every=0):
    X, Y, phi, u, v = build_initial(p)
    A0 = phi.sum()                      # target area (mass) of the worm
    Phx, Phy = 1.0, 0.0                 # polarity direction (start +x)

    frames, coms, times = [], [], []
    com_x0 = (phi * X).sum() / phi.sum()

    for step in range(p.nsteps):
        # ---- excitable layer (gated by the body, leaks outside) ----
        thr = (v + p.b) / p.a
        fu = (u * (1.0 - u) * (u - thr)) / p.eps
        u = u + p.dt * (p.Du * lap9(u) + phi * fu - p.k_leak * (1.0 - phi) * u)
        v = v + p.dt * (phi * (u - v))
        np.clip(u, 0.0, 1.0, out=u)

        # ---- pacemaker: periodically fire a pulse at the worm's tail ----
        # The tail is the rear-most slice in x (worm crawls +x); keying the
        # pacemaker to x (not the full polarity) avoids a y-steering runaway.
        if step % p.pace_every == 0:
            body = phi > 0.5
            if body.any():
                proj = X                            # rear == smallest x
                pmin = proj[body].min()
                span = proj[body].max() - pmin + 1e-9
                rear = body & (proj < pmin + p.rear_frac * span)
                if rear.any():
                    ry, rx = np.argwhere(rear).mean(axis=0)
                    tip = np.hypot(X - rx, Y - ry) < p.pace_radius
                    u[tip & (phi > 0.3)] = 1.0

        # ---- polarity from the excitation:  P = <-grad v> over the pulse ----
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
            Phy *= 0.97                    # gentle straightening (heading inertia)
            n2 = np.hypot(Phx, Phy) + 1e-9
            Phx, Phy = Phx / n2, Phy / n2
        go = 0.0 if not p.act_ref else min(activity / p.act_ref, 1.0)

        # ---- body: a phase field that is ADVECTED along the polarity P -------
        # The excitation provides a velocity V = v_crawl * activity * P; this
        # translates the whole worm (so it keeps its shape and crawls) instead
        # of just growing a head. Cohesion + area term keep phi sharp & ~const.
        gx, gy = grad(phi)
        A = phi.sum()
        cohesion = (phi * (1.0 - phi) * (phi - 0.5 + p.mu * (A0 - A) / A0)) / p.tau
        Vx = p.v_crawl * go * Phx
        Vy = p.v_crawl * go * Phy
        adv = -(Vx * gx + Vy * gy)

        phi = phi + p.dt * (p.Dphi * lap9(phi) + cohesion + adv)
        np.clip(phi, 0.0, 1.0, out=phi)

        # ---- record ----
        if record_every and step % record_every == 0:
            cx = (phi * X).sum() / phi.sum()
            cy = (phi * Y).sum() / phi.sum()
            frames.append((phi.copy(), u.copy(), v.copy(), Phx, Phy, cx, cy))
            coms.append((cx, cy))
            times.append(step * p.dt)

    final = dict(phi=phi, u=u, v=v, A0=A0, A=phi.sum(),
                 com_x0=com_x0, com_x=(phi * X).sum() / phi.sum(),
                 com_y=(phi * Y).sum() / phi.sum())
    return frames, coms, times, final


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def compose_rgb(phi, u, v):
    """Composite the three fields into one RGB image:
    body = teal, refractory tail (v) = magenta glow, excitation pulse = hot."""
    h, w = phi.shape
    bg = np.array([0.05, 0.06, 0.09])
    body = np.array([0.10, 0.42, 0.52])
    refr_col = np.array([0.55, 0.12, 0.48])
    img = np.broadcast_to(bg, (h, w, 3)).copy()
    a = phi[..., None]
    img = img * (1 - a) + body * a
    vr = (np.clip((v - 0.03) / 0.32, 0, 1) * phi)[..., None]
    img = img * (1 - 0.8 * vr) + refr_col * (0.8 * vr)
    uu = np.clip(u, 0, 1)
    hot = np.stack([np.clip(1.3 * uu, 0, 1),
                    np.clip(1.5 * uu - 0.35, 0, 1),
                    np.clip(2.2 * uu - 1.2, 0, 1)], axis=-1)
    au = (uu * (phi > 0.25))[..., None] * 0.95
    img = img * (1 - au) + hot * au
    return np.clip(img, 0, 1)


def make_gif(p, frames, times, fname="worm_crawl.gif", stride=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = frames[::stride]; times = times[::stride]
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=110)
    fig.patch.set_facecolor("#0d0e12"); ax.set_facecolor("#0d0e12")
    phi0, u0, v0, Phx, Phy, cx, cy = frames[0]
    im = ax.imshow(compose_rgb(phi0, u0, v0), origin="lower",
                   interpolation="bilinear", animated=True)
    (trail,) = ax.plot([], [], color="#7fe7ff", lw=1.0, alpha=0.7)
    quiv = ax.quiver([cx], [cy], [Phx], [Phy], color="#ffe08a", scale=12,
                     width=0.006, headwidth=4)
    txt = ax.text(0.015, 0.9, "", transform=ax.transAxes, color="#dfe6f0",
                  fontsize=10, family="monospace")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("excitable wave (hot) drives the body (teal) — refractory tail (magenta) locks the heading",
                 color="#aeb6c2", fontsize=8.5, pad=6)
    xs, ys = [], []
    cx0 = frames[0][5]

    def update(i):
        phi, u, v, Phx, Phy, cx, cy = frames[i]
        im.set_array(compose_rgb(phi, u, v))
        xs.append(cx); ys.append(cy)
        trail.set_data(xs, ys)
        quiv.set_offsets([[cx, cy]]); quiv.set_UVC([Phx], [Phy])
        txt.set_text(f"t={times[i]:6.1f}   crawled dx = {cx - cx0:+6.1f}")
        return im, trail, quiv, txt

    anim = FuncAnimation(fig, update, frames=len(frames), interval=55, blit=False)
    anim.save(fname, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"wrote {fname}  ({len(frames)} frames)")


def plot_com(coms, times, fname="com.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cx = [c[0] for c in coms]; cy = [c[1] for c in coms]
    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    ax.plot(times, cx, color="#1f77b4", label="centre-of-mass x")
    ax.plot(times, cy, color="#d62728", label="centre-of-mass y")
    ax.set_xlabel("time"); ax.set_ylabel("position"); ax.legend()
    ax.set_title("Directed crawl: x advances steadily, y stays fixed")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    print(f"wrote {fname}")


if __name__ == "__main__":
    import time
    p = P()
    t0 = time.time()
    frames, coms, times, final = simulate(p, record_every=60)
    print(f"sim wall={time.time()-t0:.1f}s  area {final['A0']:.0f}->{final['A']:.0f}"
          f"  COM_x {final['com_x0']:.0f}->{final['com_x']:.0f}"
          f"  (dx={final['com_x']-final['com_x0']:+.0f})  COM_y={final['com_y']:.1f}")
    make_gif(p, frames, times)
    plot_com(coms, times)
