"""
Side-by-side comparison:  classic reaction-diffusion "worm"  vs.  the
excitable-layer worm.

LEFT  -- Gray-Scott reaction-diffusion in its "worms" regime. The canonical
         RD worm: it grows, branches and splits in *all* directions. There is
         no heading -- it is a spreading colony, not a crawler.

RIGHT -- the excitable-layer worm (worm.py). The extra excitable field gives
         it a polarity, so it travels along its own body axis: it crawls.

This is exactly the gap the excitation layer fills:
    plain RD  -> isotropic growth (or a front that drifts sideways)
    + excitation -> a locked-in heading -> directed crawling.

Run:  python3 compare.py        # writes comparison.gif
"""

import numpy as np
import worm


# ----------------------------------------------------------------------------
# Classic Gray-Scott reaction-diffusion (the "original" RD worm)
# ----------------------------------------------------------------------------
def lap5(f):
    return (-4.0 * f
            + np.roll(f, 1, 0) + np.roll(f, -1, 0)
            + np.roll(f, 1, 1) + np.roll(f, -1, 1))


def gray_scott(N=200, steps=10000, record_every=80, F=0.054, k=0.062,
               Du=0.16, Dv=0.08, seed=0):
    """Gray-Scott in the worms regime. Returns a list of V-field snapshots."""
    rng = np.random.default_rng(seed)
    U = np.ones((N, N)); V = np.zeros((N, N))
    # a few seed blobs -> they grow into branching worms
    for _ in range(8):
        cx, cy = rng.integers(40, N - 40, size=2)
        r = 6
        U[cx - r:cx + r, cy - r:cy + r] = 0.50
        V[cx - r:cx + r, cy - r:cy + r] = 0.25
    V += 0.02 * rng.random((N, N))

    frames = []
    for s in range(steps):
        uvv = U * V * V
        U += Du * lap5(U) - uvv + F * (1.0 - U)
        V += Dv * lap5(V) + uvv - (F + k) * V
        if s % record_every == 0:
            frames.append(V.copy())
    return frames


# ----------------------------------------------------------------------------
# Render the two side by side
# ----------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    print("running Gray-Scott (classic RD worm) ...")
    gs = gray_scott(N=170, steps=10000, record_every=140)

    print("running excitable-layer worm ...")
    p = worm.P()
    p.nsteps = 6000
    wm, _, _, _ = worm.simulate(p, record_every=85)

    n = min(len(gs), len(wm))
    gs = [gs[int(i * (len(gs) - 1) / (n - 1))] for i in range(n)]
    wm = [wm[int(i * (len(wm) - 1) / (n - 1))] for i in range(n)]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 3.0), dpi=88)
    fig.patch.set_facecolor("#0d0e12")
    for ax in (axL, axR):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#0d0e12")

    imL = axL.imshow(gs[0], cmap="magma", vmin=0, vmax=0.45, animated=True,
                     interpolation="bilinear")
    axL.set_title("classic RD (Gray-Scott)\ngrows & splits in all directions",
                  color="#d98", fontsize=9)

    phi0, u0, v0, *_ = wm[0]
    imR = axR.imshow(worm.compose_rgb(phi0, u0, v0), origin="lower",
                     animated=True, interpolation="bilinear")
    axR.set_title("RD + excitation\ncrawls along one direction",
                  color="#8cf", fontsize=9)

    def update(i):
        imL.set_array(gs[i])
        phi, u, v, *_ = wm[i]
        imR.set_array(worm.compose_rgb(phi, u, v))
        return imL, imR

    anim = FuncAnimation(fig, update, frames=n, interval=66, blit=False)
    fig.tight_layout()
    anim.save("comparison.gif", writer=PillowWriter(fps=15))
    plt.close(fig)

    # re-encode with an adaptive 128-colour palette to keep the file small
    from PIL import Image, ImageSequence
    im = Image.open("comparison.gif")
    fr = [f.convert("RGB").quantize(colors=128, method=Image.MEDIANCUT)
          for f in ImageSequence.Iterator(im)]
    fr[0].save("comparison.gif", save_all=True, append_images=fr[1:],
               loop=0, duration=66, optimize=True)
    import os
    print(f"wrote comparison.gif  ({n} frames, "
          f"{os.path.getsize('comparison.gif')/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
