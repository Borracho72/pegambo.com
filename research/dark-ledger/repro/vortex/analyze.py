"""Block bootstrap (random-wave realizations); guard-zone Palm/pair estimators."""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from theory import correlations, cubic_coefficient, high_precision

STEP = .025
G_EDGES = np.arange(0, 1.5 + STEP/2, STEP)
G_MID = (G_EDGES[:-1] + G_EDGES[1:])/2
G_AREA = np.pi*np.diff(G_EDGES**2)
FITS = [(0.0375, .1375), (.0625, .1625), (.0875, .1875),
        (.0875, .2875), (.1375, .2875), (.1875, .3375),
        (.2375, .3875), (.2875, .4375), (.3875, .5375), (.4375, .6375)]
AT_POINT = np.array([[.09, .11], [.075, .125], [.095, .105]])
M_EDGES = np.arange(0, 1.5+.002, .005)
M_MID = (M_EDGES[:-1] + M_EDGES[1:])/2
N_EDGES = np.linspace(0, 5, 1001)
N_PDF_EDGES = np.linspace(0, 5, 101)
MARGIN = 3.


def info(p, width, all_sign_nn=True):
    core_ids = np.flatnonzero(np.all(np.abs(p[:, :2]) < width/2-MARGIN, axis=1))
    inside = p[core_ids]
    xy, X = p[:, :2], inside[:, :2]
    t, ref = cKDTree(xy), cKDTree(X)
    nn, j = t.query(X, k=2)
    nnd = nn[:, 1]
    isopp = inside[:, 2] != p[j[:, 1], 2]
    if np.max(nnd) >= MARGIN:
        raise ValueError("NN exceeds guard zone")
    pair = ref.sparse_distance_matrix(t, G_EDGES[-1], output_type="ndarray")
    rr = pair["v"]
    ss = (rr > 0) & (inside[pair["i"], 2] == p[pair["j"], 2])
    same = rr[ss]
    hist_all = np.histogram(rr[rr > 0], G_EDGES)[0]
    hist_same = np.histogram(same, G_EDGES)[0]
    local = np.array([np.count_nonzero((lo <= same) & (same < hi)) for lo, hi in AT_POINT])
    del pair, rr, ss, same
    out = dict(n=len(X), nopp=int(isopp.sum()), hist_all=hist_all, hist_same=hist_same,
               local=local, nnd=nnd, nnd_opp=nnd[isopp], nnd_same=nnd[~isopp],
               nplus=int(np.count_nonzero(inside[:, 2] > 0)),
               q_spec_sorted=np.unique(inside[:, 2]).astype(int).tolist())
    target_same = np.empty(len(X))
    target_different = np.empty(len(X))
    if all_sign_nn:
        neigh_ids = j[:, 1]
        other_neigh_ids = t.query(xy[neigh_ids], k=2)[1][:, 1]
        mutual = other_neigh_ids == core_ids
        pairs = np.unique(np.sort(np.c_[core_ids,neigh_ids],axis=1),axis=0)
        out.update(n_mut=int(mutual.sum()), n_mut_opp=int((isopp & mutual).sum()),
                   n_undir=len(pairs), n_undir_opp=int((p[pairs[:,0],2]!=p[pairs[:,1],2]).sum()))
        for sign in (-1, 1):
            g = inside[:, 2] == sign
            sg = p[:, 2] == sign
            target_same[g] = cKDTree(xy[sg]).query(X[g], k=2)[0][:, 1]
            target_different[g] = cKDTree(xy[~sg]).query(X[g], k=1)[0]
        if max(np.max(target_same), np.max(target_different)) >= MARGIN:
            raise ValueError("marked NN exceeds guard zone")
        out["nnd_target_same"] = target_same
        out["nnd_target_opp"] = target_different
    return out


def summ(points, res):
    good = res[np.isfinite(res)]
    return dict(estimate=float(points) if np.isfinite(points) else None,
                se=float(good.std(ddof=1)) if len(good)>1 else None,
                ci95=np.percentile(good,[2.5,97.5]).tolist() if len(good) else None,
                valid_replicates=int(len(good)))


def all_summ(arr):
    return summ(arr[0], arr[1:])


def estim_mtx(rng, n, reps):
    return np.vstack((np.ones(n), rng.multinomial(n, np.full(n, 1/n), size=reps)))


def mode(hist, sigma):
    y = gaussian_filter1d(hist.astype(float), sigma/np.diff(M_EDGES)[0], axis=-1, mode="constant")
    i = np.argmax(y, axis=1)
    row = np.arange(len(y))
    i = np.clip(i, 1, len(M_MID)-2)
    prev, peak, nex = y[row, i-1], y[row, i], y[row, i+1]
    offset = (prev - nex) / (2*np.minimum(-1e-16, prev-2*peak+nex))
    return M_MID[i] + offset*np.diff(M_EDGES)[0]


def slopes(hists, interval, weighted):
    ii = (G_MID >= (interval[0]-1e-12)) & (G_MID <= (interval[1]+1e-12))
    xx = np.log(G_MID[ii])
    hc = hists[:, ii]
    yy = np.log(np.maximum(1, hc)/G_AREA[ii])
    weights = hc if weighted else np.asarray(hc > 0, float)
    weights = weights/np.maximum(1, weights.sum(axis=1, keepdims=True))
    nx = xx - np.sum(weights*xx, axis=1, keepdims=True)
    nv = yy - np.sum(weights*yy, axis=1, keepdims=True)
    denom = np.sum(weights*nx**2, axis=1)
    return np.divide(np.sum(weights*nx*nv, axis=1), denom,
                     out=np.full(len(denom), np.nan), where=denom>0), hc[0].astype(int).tolist()


def basic(stats, B, width):
    def pooled(key):
        return B @ np.asarray([s[key] for s in stats])
    N = pooled("n")
    h = pooled("hist_same")
    ar = np.pi*np.diff(AT_POINT**2,axis=1)[:,0]
    g100 = pooled("local") / (N[:,None]*ar[None,:]*(np.pi/2))
    out = dict(n_points=int(N[0]), positive_fraction=float(pooled("nplus")[0]/N[0]),
               density=all_summ(N/(len(stats)*(width-2*MARGIN)**2)),
               opposite_nn_fraction=all_summ(pooled("nopp")/N),
               opposite_nn_fraction_mutual=all_summ(pooled("n_mut_opp")/pooled("n_mut")),
               opposite_nn_fraction_unique_undirected=all_summ(pooled("n_undir_opp")/pooled("n_undir")),
               g_same_at_point=[dict(bin=ab.tolist(), n_pairs=int(pooled("local")[0,i]),
                                     **all_summ(g100[:,i])) for i,ab in enumerate(AT_POINT)],
               g_same_loglog_slopes={}, nearest_mode={}, g_hist={})
    for window in FITS:
        key = f"{window[0]}-{window[1]}"
        u, cnt = slopes(h, window, False)
        w, _ = slopes(h, window, True)
        out["g_same_loglog_slopes"][key] = dict(unweighted=all_summ(u),
                                              count_weighted=all_summ(w), n_pairs_by_bin=cnt)
    for label in ("nnd", "nnd_opp", "nnd_same", "nnd_target_opp", "nnd_target_same"):
        hist = B @ np.array([np.histogram(s[label], M_EDGES)[0] for s in stats])
        avg = (B @ np.array([s[label].sum() for s in stats]))/(B @ np.array([len(s[label]) for s in stats]))
        out["nearest_mode"][label] = dict(mean=all_summ(avg))
        for sigma in (.0075, .0125, .02):
            out["nearest_mode"][label][str(sigma)] = all_summ(mode(hist,sigma))
    gh = pooled("hist_all")
    div = N[:,None]*np.pi*G_AREA[None,:]
    g, gs = gh/div, 2*h/div
    for k, val in zip(("all", "same", "opposite", "signed"), (g,gs,2*g-gs,gs-g)):
        out["g_hist"][k] = dict(estimate=val[0].tolist(), se=val[1:].std(axis=0,ddof=1).tolist(),
                                ci95=np.percentile(val[1:],[2.5,97.5],axis=0).T.tolist())
    return out


def overlap(group, width, m):
    out = info(np.concatenate(group,axis=0), width, False)
    nn = np.sort(out.pop("nnd")*np.pi*np.sqrt(m))
    out.update(nn_cdf_count=np.searchsorted(nn, N_EDGES, side="right"),
               nn_pdf_count=np.histogram(nn,N_PDF_EDGES)[0])
    return out


def poisson_metrics(stats_blocks, B, m):
    # Each item in stats_blocks contains one or more disjoint unions.
    data = {k:B @ np.asarray([sum(s[k] for s in block) for block in stats_blocks])
            for k in ("n", "hist_all", "hist_same", "nn_cdf_count", "nn_pdf_count")}
    n = data["n"][:,None]
    nn_ref = 1-np.exp(-N_EDGES**2)
    nn_delta = data["nn_cdf_count"]/n - nn_ref[None,:]
    d_pdf = data["nn_pdf_count"]/n - np.diff(1-np.exp(-N_PDF_EDGES**2))[None,:]
    div = n*m*np.pi*G_AREA[None,:]
    g = data["hist_all"]/div
    gs = 2*data["hist_same"]/div
    d = {"NN KS": np.max(np.abs(nn_delta),axis=1),
         "NN Wasserstein-1 scaled": np.trapezoid(np.abs(nn_delta),x=N_EDGES,axis=1),
         "NN Cramer-von Mises squared": np.trapezoid(nn_delta**2,x=nn_ref,axis=1),
         "NN histogram total variation": .5*np.sum(np.abs(d_pdf),axis=1),
         "NN histogram relative RMS": np.sqrt(np.sum(d_pdf**2,axis=1)/np.sum(np.diff(1-np.exp(-N_PDF_EDGES**2))**2))}
    for rmax in (.5,1.,1.5):
        mask = G_MID < rmax
        for lab, vals in zip(("all","same","opposite","charge"), (g-1,gs-1,2*g-gs-1,gs-g)):
            d[f"g {lab} RMS [0,{rmax}]"] = np.sqrt(np.mean(vals[:,mask]**2,axis=1))
            d[f"g {lab} L1 [0,{rmax}]"] = np.mean(np.abs(vals[:,mask]),axis=1)
        d[f"g all+charge RMS euclidean [0,{rmax}]"] = np.sqrt(np.mean(((g-1)**2+(gs-g)**2)[:,mask],axis=1))
    out = {k:all_summ(v) for k,v in d.items()}
    out.update(n_points=int(n[0,0]), independent_blocks=len(stats_blocks))
    return out


def area_means(radii):
    x,w = np.polynomial.legendre.leggauss(32)
    vals = []
    for lo, hi in radii:
        r = (lo+hi)/2+(hi-lo)/2*x
        vals.append(float(np.dot(w*r, [correlations(p)[2] for p in r])/(lo+hi)))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path(__file__).with_name("fields"))
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("results.json"))
    ap.add_argument("--reps", type=int, default=1600)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verified-only", action="store_true",
                    help="discard unverified Newton roots (reported in separate sensitivity run)")
    ap.add_argument("--skip-overlay", action="store_true")
    a = ap.parse_args()
    files = sorted(a.dir.glob("field_*.npz"))[:a.limit]
    if len(files) < 8 or len(files)%8:
        raise ValueError("Number of input fields must be a multiple of 8")
    rng = np.random.default_rng(459208)
    B = estim_mtx(rng, len(files), a.reps)
    trees = {}
    roots = []
    fail, bilbad = 0, np.zeros(3,int)
    for name in ("roots","fine","medium","coarse"):
        stats = []
        for f in files:
            with np.load(f) as fdata:
                width = float(fdata["width"])
                fine = float(fdata["fine_step"])
                directions = int(fdata["directions"])
                seed = int(fdata["seed"])
                arr = fdata[name]
                if name == "roots":
                    if a.verified_only:
                        okay = fdata["valid_roots"] if "valid_roots" in fdata else np.any(
                            arr[:, :2] != fdata["fine"][:, :2], axis=1)
                        arr = arr[okay]
                    roots.append(arr)
                    fail += int(fdata["bad_refine"])
                    bilbad += fdata["bad"]
                ss = info(arr,width)
                assert ss["q_spec_sorted"] == [-1,1]
                stats.append(ss)
        trees[name] = basic(stats,B,width)
        print(name, trees[name]["n_points"], trees[name]["opposite_nn_fraction"], flush=True)

    allpoiss = {}
    PB = estim_mtx(rng, len(files)//8, a.reps)
    for m in (() if a.skip_overlay else (1,2,4,8)):
        blocks = []
        for b in range(0,len(files),8):
            blocks.append([overlap(roots[h:h+m],width,m) for h in range(b,b+8,m)])
        allpoiss[str(m)] = poisson_metrics(blocks,PB,m)
        print("overlay",m,allpoiss[str(m)]["NN KS"],flush=True)

    t = np.array([correlations(r) for r in G_MID])
    tt = {}
    for lo, hi in FITS:
        ind = (G_MID >= lo-1e-12) & (G_MID <= hi+1e-12)
        tt[f"{lo}-{hi}"] = float(np.polyfit(np.log(G_MID[ind]),np.log(t[ind,2]),1)[0])
    approx = {}
    for r in (.1,.05,.02,.01,.005,.002,.001):
        v = high_precision(r)
        approx[str(r)] = dict(g=float(v[0]),gQ=float(v[1]),g_same=float(v[2]),
                              g_opposite=float(v[3]),g_same_div_r_cubed=float(v[2]/r**3))
    # Exact pair dilution theory evaluated on the same physical grid, plus literature.
    g_exact_bins = t[:,0]
    div_theory = {}
    for m in (1,2,4,8):
        dev = (g_exact_bins-1)/m
        div_theory[str(m)] = {f"g all RMS [0,{rmax}]":float(np.sqrt(np.mean(dev[G_MID<rmax]**2)))
                              for rmax in (.5,1.,1.5)}
    results = dict(
        simulation=dict(seed=seed,directions=directions,fields=len(files),width=width,
                        margin=MARGIN,fine_step=fine,bootstrap_reps=a.reps,bootstrap_seed=459208,
                        failed_true_refine_in_full_windows=fail,failed_bilinear_fine_medium_coarse=bilbad.tolist(),
                        verified_roots_only=a.verified_only,numpy=np.__version__,scipy=scipy.__version__),
        simulations=trees,poisson_overlay=allpoiss,pair_r_grid=G_MID.tolist(),
        exact_berry_dennis=dict(density_lambda_minus2=np.pi, limits=dict(g=.25,gQ=-.25,g_same=0,g_opposite=.5),
                               g_same_asymptotic_exponent=3,g_same_asymptotic_coefficient_lambda=cubic_coefficient(),
                               g_same_area_means=area_means(AT_POINT),high_precision=approx,
                               g_hist_sample_at_centers=t.tolist(),loglog_slopes=tt,poisson_overlay=div_theory))
    a.out.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
