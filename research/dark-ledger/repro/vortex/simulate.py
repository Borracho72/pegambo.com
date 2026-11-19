"""Circular-spectrum proper complex Gaussian random wave; lambda = 1."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np


def sample_field(rng, x, n):
    # Offset guards against artificial orientation locking across realizations.
    theta = (2 * np.pi / n) * (np.arange(n) + rng.random())
    kx, ky = 2 * np.pi * np.cos(theta), 2 * np.pi * np.sin(theta)
    w = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2*n)
    ex = np.exp(1j * np.outer(kx, x))
    ey = np.exp(1j * np.outer(x, ky))
    return (ey * w) @ ex, kx, ky, w


def detect(f, x):
    """CCW principal phase increments; subpixel bilinear complex zero."""
    phase = np.angle(f)
    dx, dy = np.diff(phase, axis=1), np.diff(phase, axis=0)
    wx = (dx < -np.pi).astype(np.int8) - (dx > np.pi)
    wy = (dy < -np.pi).astype(np.int8) - (dy > np.pi)
    q = wx[:-1, :] + wy[:, 1:] - wx[1:, :] - wy[:, :-1]
    iy, ix = np.where(q != 0)
    a = f[iy, ix]
    b = f[iy, ix + 1] - a
    c = f[iy + 1, ix] - a
    d = f[iy + 1, ix + 1] - a - b - c
    u = np.ones(len(a)) / 2
    v = u.copy()
    for _ in range(15):
        val = a + u*b + v*c + u*v*d
        du, dv = b + v*d, c + u*d
        det = (du.conj()*dv).imag
        safe = np.abs(det) > 1e-25
        u[safe] += np.clip((dv.conj()*val).imag[safe]/det[safe], -.175, .175)
        v[safe] -= np.clip((du.conj()*val).imag[safe]/det[safe], -.175, .175)
        np.clip(u, 0, 1, out=u)
        np.clip(v, 0, 1, out=v)
    bad = np.abs(a + u*b + v*c + u*v*d) > 1e-8
    u[bad], v[bad] = .5, .5
    h = x[1] - x[0]
    return np.c_[x[ix] + h*u, x[iy] + h*v, q[iy, ix]], int(bad.sum())


def refine(p, kx, ky, w, fine_step):
    """Correct bilinear positions against the true wave, not grid samples."""
    p = p.copy()
    valid = np.ones(len(p), dtype=bool)
    for s in range(0, len(p), 256):
        group = p[s:s+256]
        start = group[:, :2].copy()
        xy = start.copy()
        active = np.ones(len(xy), dtype=bool)
        for _ in range(12):
            a = np.flatnonzero(active)
            if not len(a):
                break
            phase = np.exp(1j*(xy[a, 0, None]*kx + xy[a, 1, None]*ky))*w
            val = phase.sum(axis=1)
            vx = 1j * (phase*kx).sum(axis=1)
            vy = 1j * (phase*ky).sum(axis=1)
            det = (vx.conj()*vy).imag
            good = (np.abs(val) < 1e-11) & (det*group[a, 2] > 0)
            active[a[good]] = False
            update = np.c_[(vy.conj()*val).imag, -(vx.conj()*val).imag]
            update /= np.where(np.abs(det)>1e-16, det, np.inf)[:, None]
            update *= np.minimum(1, .4*fine_step/np.maximum(1e-20, np.linalg.norm(update,axis=1)))[:,None]
            xy[a[~good]] += update[~good]
        bad = active | (np.max(np.abs(xy-start),axis=1) > 1.25*fine_step)
        valid[s:s+len(group)] = ~bad
        group[~bad, :2] = xy[~bad]
    return p, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=int, default=64)
    ap.add_argument("--start", type=int, default=0, help="resume at field index, seed is unchanged")
    ap.add_argument("--seed", type=int, default=614902)
    ap.add_argument("--width", type=float, default=54)
    ap.add_argument("--fine", type=float, default=.025)
    ap.add_argument("--directions", type=int, default=640)
    ap.add_argument("--dir", type=Path, default=Path(__file__).with_name("fields"))
    a = ap.parse_args()
    a.dir.mkdir(parents=True, exist_ok=True)
    intervals = round(a.width/a.fine)
    if intervals % 4 or not np.isclose(intervals*a.fine, a.width):
        raise ValueError("width/fine must be an integer divisible by 4")
    x = np.linspace(-a.width/2, a.width/2, intervals+1)
    for i, child in enumerate(np.random.SeedSequence(a.seed).spawn(a.fields)):
        if i < a.start:
            continue
        f, kx, ky, w = sample_field(np.random.default_rng(child), x, a.directions)
        points, bad = zip(*(detect(f[::stride, ::stride], x[::stride]) for stride in (1, 2, 4)))
        roots, valid = refine(points[0], kx, ky, w, a.fine)
        nb = int((~valid).sum())
        np.savez_compressed(a.dir/f"field_{i:03d}.npz", fine=points[0], medium=points[1],
                            coarse=points[2], roots=roots, valid_roots=valid, bad=bad, bad_refine=nb,
                            fine_step=a.fine, width=a.width, seed=a.seed, directions=a.directions)
        print(i, [len(p) for p in points], bad, nb, flush=True)


if __name__ == "__main__":
    main()
