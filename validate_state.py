# Out-of-band validation of a trained state (2026-10-30): many more validators than a training run
# can carry, each fitted to convergence with the shared model frozen, so states are compared at
# convergence rather than wherever their run left the in-run validators.
#
# The validators (validator_manifest.py: archive calibrators not used in training, plus the 15
# in-run ones) are built by retrain.py itself as UniqueAbPointFit exposures, so their parameters are
# keyed exactly as in training: position, flux and wavefront per exposure; spectrum per star and
# filter; focus per program, filter and dither position; rotation per program. Everything else
# (network, pixel maps, blurs, mask metrology) comes from the state and is held. Exposures that share
# a key are fitted together (a "component"). Each exposure has its own compiled loss and gradient,
# summed on the host (a joint compiled objective would hold every exposure's intermediates on the
# GPU at once); the sum is minimised by SciPy's L-BFGS in coordinates whitened by the Gauss-Newton
# Fisher matrix at the start (a unit step is one standard deviation along each eigenvector).
#
# Reported per exposure, before and after the fit:
#   loss       the training loss: mean over pixels of the chi^2 summed over the slopes (grows with
#              NGROUPS, so compare like with like)
#   chi2_slope loss / (NGROUPS - 1): the reduced chi^2 per slope, 1 for a perfect model
#   bands      retrain_fns.residual_row: rms of the per-pixel mean z in flux-ranked bands (core50,
#              core300, mid, bg), each with what a perfect model would leave (noise_*)
# and the transfer test (VAL_TRANSFER=1): for two stars of one program observed on the same date
# in the same filter and dither position, B's own fitted wavefront is replaced by A's and only B's
# position, flux and spectrum are refitted, as a science target inherits its calibrator's
# wavefront. B's focus and rotation are shared with A already (same keys), fitted jointly with both.
#
#   run with the state's usual RETRAIN_* environment (RETRAIN_BEST_STATE, the basis, the network
#   config, RETRAIN_ROTATION_KEY ...) and the amigo it was trained with (PYTHONPATH), plus
#   VAL_OUT=<dir>        results: shard_<i>_of_<n>.json (rows) and _maps.npz (mean-z maps)
#   VAL_MANIFEST         validator list (default outputs/validation/validators.json)
#   VAL_SHARD=i/n        this task's share of the components (default SLURM_ARRAY_TASK_ID/COUNT, else all)
#   VAL_PROGRAMS=a,b     only these programs; VAL_FILTERS=a,b only these filters
#   VAL_STEPS (300)      L-BFGS iterations per component; VAL_TRANSFER_STEPS (100) per transfer
#   VAL_WHITEN (1)       0 skips the Fisher whitening (plumbing tests on a CPU only)
#   VAL_AB_INIT          new validators' starting wavefront: "calibrator" (default; the trained one
#                        of the same filter and dither position, else 04481's POS1) or "zero"
#   VAL_TRANSFER (1)     run the transfer test
#   VAL_PLOTS (1)        a summarise_fit figure per validator at its fitted values, in <VAL_OUT>/fits
import json
import os
import sys
import time
from collections import defaultdict

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
os.environ.setdefault("RETRAIN_INIT_PLOTS", "0")
os.environ.setdefault("RETRAIN_RESIDUAL_STATS", "0")
os.environ.setdefault(
    "RETRAIN_BINARY", "0"
)  # the binary's 30 exposures are not needed here

MANIFEST = os.environ.get(
    "VAL_MANIFEST", "/fred/oz440/max/outputs/validation/validators.json"
)
OUT = os.environ["VAL_OUT"]
STEPS = int(os.environ.get("VAL_STEPS", 300))
TRANSFER_STEPS = int(os.environ.get("VAL_TRANSFER_STEPS", 100))
AB_INIT = os.environ.get("VAL_AB_INIT", "calibrator")
TRANSFER = os.environ.get("VAL_TRANSFER", "1") == "1"
PLOTS = os.environ.get("VAL_PLOTS", "1") == "1"
PROGRAMS = [
    p for p in os.environ.get("VAL_PROGRAMS", "").replace(":", ",").split(",") if p
]  # ":" too (sbatch --export)
FILTERS = [
    f for f in os.environ.get("VAL_FILTERS", "").replace(":", ",").split(",") if f
]
WHITEN = os.environ.get("VAL_WHITEN", "1") == "1"
if "VAL_SHARD" in os.environ:
    SHARD, NSHARD = (int(v) for v in os.environ["VAL_SHARD"].split("/"))
else:
    SHARD, NSHARD = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)), int(
        os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)
    )
t_start = time.time()
log = lambda msg: print(f"[{(time.time() - t_start) / 60:6.1f} min] {msg}", flush=True)

# ---- this shard's validators: components from the manifest, by the keys retrain.py will use ----------
defocus_key = os.environ.get("RETRAIN_DEFOCUS_KEY", "position")
rotation_key = os.environ.get("RETRAIN_ROTATION_KEY", "")
rotation_split = [
    p
    for p in os.environ.get("RETRAIN_ROTATION_SPLIT", "").replace(":", ",").split(",")
    if p
]
# RETRAIN_VAL_PTT_KEY (2026-11-17): validators' PTT per exposure (default), per visit (program, observation, POS; shared
# across filters) or per visit and filter; with RETRAIN_FILTER_PTT=fixed every exposure also reads the state's fixed
# per-filter term (the plan for the pipeline: PTT per visit plus the base model's per-filter term)
val_ptt_key = os.environ.get("RETRAIN_VAL_PTT_KEY", "exposure")


def shared_keys(r):
    keys = [f"spectra:{r['star']}_{r['filter']}"]
    if defocus_key == "position":
        keys.append(f"defocus:{r['program']}_{r['filter']}_{r['pos']}")
    elif defocus_key == "program":
        keys.append(f"defocus:{r['program']}_{r['filter']}")
    if rotation_key in ("program", "observation"):
        split = rotation_key == "observation" or r["program"] in rotation_split
        keys.append(
            f"rotations:{r['program']}" + (f"_{r['observation']}" if split else "")
        )
    if val_ptt_key == "visit":
        keys.append(f"aberrations:{r['program']}_{r['observation']}_{r['pos']}")
    elif val_ptt_key == "visit_filter":
        keys.append(
            f"aberrations:{r['program']}_{r['observation']}_{r['filter']}_{r['pos']}"
        )
    return keys


def components(items, keys_of):
    """Groups items linked by any shared key (union-find)."""
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner = {}
    for i, item in enumerate(items):
        for k in keys_of(item):
            if k in owner:
                parent[find(i)] = find(owner[k])
            else:
                owner[k] = i
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(items[i])
    return list(groups.values())


manifest = json.load(open(MANIFEST))["exposures"]
if PROGRAMS:
    manifest = [r for r in manifest if r["program"] in PROGRAMS]
if FILTERS:
    manifest = [r for r in manifest if r["filter"] in FILTERS]
comps = sorted(
    components(manifest, shared_keys),
    key=lambda c: -sum(1 + r["ngroups"] / 30 for r in c),
)
load = [0.0] * NSHARD
mine = []
for c in comps:  # largest first, each to the least loaded shard
    s = load.index(min(load))
    load[s] += sum(1 + r["ngroups"] / 30 for r in c)
    if s == SHARD:
        mine += c
log(
    f"shard {SHARD}/{NSHARD}: {len(mine)} of {len(manifest)} validators in "
    f"{sum(1 for c in comps if c[0] in mine)} of {len(comps)} components"
)
if not mine:
    sys.exit()

# ---- build the model with retrain.py, the manifest's files in place of its validators -------------------
_retrain_path = os.path.join(_dir, "retrain.py")
_src = open(_retrain_path).read()
_m1, _m2 = (
    "# %% [markdown]\n# ## Calibration files",
    "# %% [markdown]\n# ## Checking initial fits",
)
part1, rest = _src.split(_m1)
part2 = (_m1 + rest).split(_m2)[0]
g = {"__name__": "__validate__"}  # no __file__: retrain.py saves nothing
exec(compile(part1, _retrain_path, "exec"), g)
# retrain.py turns on the persistent compile cache, which keeps a file for every compiled function;
# each validator compiles its own once per job, and /fred has a file-count quota
g["jax"].config.update("jax_enable_compilation_cache", False)

import numpy as onp
from astropy.io import fits


def _add_badpix(file):
    """retrain.py's mask, plus the pixels this file cannot evaluate. add_badpix replaces the file's own
    BADPIX with the static mask; training data has no non-finite pixels, but the deepest validators
    (01509 HR-7258) have a few saturated ones whose covariance is NaN. nanmean drops them from the
    loss but not from its gradient, which comes back NaN (2026-11-02)."""
    slopes, cov = file["SLOPE"].data, onp.asarray(file["SLOPE_COV"].data, float)
    var = onp.diagonal(cov, axis1=0, axis2=1)
    finite = onp.isfinite(cov).all((0, 1))
    bad = ~(onp.isfinite(slopes).all(0) & finite & (var > 0).all(-1))
    # pixels whose slope covariance is not positive definite (2026-11-17: 16 of the 120, all 01509 F480M, 1-5
    # pixels for HD-93237 and ~200-290 for HR-7258): a full-covariance chi^2 is unbounded below there. Masked in
    # both likelihoods, so diagonal and full-covariance results cover the same pixels.
    G = cov.shape[0]
    if G > 1:
        C = onp.transpose(
            onp.where(finite[None, None], cov, onp.eye(G)[:, :, None, None]),
            (2, 3, 0, 1),
        )
        bad |= onp.linalg.eigvalsh(C)[..., 0] <= 0
    file = g["add_badpix"](file)
    file["BADPIX"].data[bad] = 1
    return file


g["val_files"] = [_add_badpix(fits.open(r["path"])) for r in mine]
# a new program's focus starts from the mean of the state's focus in its filter (retrain.py falls back
# to a per-filter key, which states keyed by position do not have)
_defocus = g["best_state"]["defocus"]
for filt in ("F380M", "F430M", "F480M"):
    if filt not in _defocus:
        vals = [float(v) for k, v in _defocus.items() if filt in k]
        _defocus[filt] = g["np"].asarray(sum(vals) / len(vals))
exec(compile(part2, _retrain_path, "exec"), g)
# VAL_FOURIER_PROGRAM=<calibrator program> (2026-11-20): every validator takes that program's Fourier set instead
# of the closest-dated one (does the high-order wavefront change between epochs?). Unset: closest-dated, as before.
FOURIER_PROGRAM = os.environ.get("VAL_FOURIER_PROGRAM")
if FOURIER_PROGRAM:
    if FOURIER_PROGRAM not in g["fourier_programs"]:
        raise ValueError(
            f"VAL_FOURIER_PROGRAM={FOURIER_PROGRAM} is not one of {g['fourier_programs']}"
        )
    g["nearest_fourier_program"] = {
        p: FOURIER_PROGRAM for p in g["nearest_fourier_program"]
    }
    log(f"Fourier set {FOURIER_PROGRAM} for every validator (VAL_FOURIER_PROGRAM)")
# VAL_FOURIER_FROM=<program>,<program>... (2026-11-20): the closest-dated rule chooses only among these calibrator
# programs, e.g. 04481,08330 for a state trained with an extra calibrator program (RETRAIN_EXTRA_CAL), so its
# validators take the same Fourier sets as before and only the shared parameters differ
FOURIER_FROM = [
    p for p in os.environ.get("VAL_FOURIER_FROM", "").replace(":", ",").split(",") if p
]
if FOURIER_FROM:
    if FOURIER_PROGRAM or not set(FOURIER_FROM) <= set(g["fourier_programs"]):
        raise ValueError(
            f"VAL_FOURIER_FROM={FOURIER_FROM}: not with VAL_FOURIER_PROGRAM, and only of {g['fourier_programs']}"
        )
    _dates = g["program_dates"]
    g["nearest_fourier_program"] = {
        p: min(FOURIER_FROM, key=lambda c: abs(_dates[c] - d))
        for p, d in _dates.items()
    }
    log(f"Fourier sets chosen among {FOURIER_FROM} only (VAL_FOURIER_FROM)")

import jax
import jax.numpy as np
import numpy as onp
import equinox as eqx
from scipy.optimize import minimize
from retrain_fns import residual_row

model, exposures, best_state = g["model"], g["val_exposures"], g["best_state"]
PTT = g[
    "PTT"
]  # the piston/tip/tilt parameter's name (retrain.py): "ptt_coeffs" with the base-model parts
if g.get("hier_wf"):
    raise NotImplementedError(
        "validate_state.py does not handle RETRAIN_WF=hierarchical"
    )
USE_COV = bool(
    g.get("use_cov")
)  # RETRAIN_USE_COV: full slope covariance in the fits (2026-11-17)
# VAL_TRIDIAGONAL=1 (needs RETRAIN_USE_COV=1): each pixel's slope covariance keeps the file's variances d on its
# diagonal and gets the read-noise anticorrelation of adjacent slopes (they share a read) on its first off-diagonals,
# zero beyond: entry (i, i+1) = -min(r, d_i / 2, d_{i+1} / 2), r = minus the mean first off-diagonal of the file's
# SLOPE_COV (clipped >= 0). Diagonally dominant with strict end rows, so positive definite by construction, unlike
# SLOPE_COV cut to its tridiagonal band (not positive definite in 121k of 768k validator pixels,
# outputs/checks/v02_merge/tridiagonal_check.py). The variances are exactly the diagonal and full runs' (amigo reads
# them off the covariance diagonal, for the metrics too); the 2026-11-20 version (r * tridiag(-1, 2, -1) +
# diag(photon >= 0)) raised d_i to 2r wherever d_i < 2r, 32% of slopes, which biased that suite's chi2 and noise
# references (tridiagonal_variance_check.py). The bad-pixel mask stays the full-covariance one, so the pixels match
# the diagonal and full runs.
TRIDIAGONAL = os.environ.get("VAL_TRIDIAGONAL", "0") == "1"
if TRIDIAGONAL:
    if not USE_COV:
        raise ValueError("VAL_TRIDIAGONAL=1 needs RETRAIN_USE_COV=1")

    from amigo.stats import tridiagonal_covariance as _tridiagonal

    exposures = [exp.set("cov", _tridiagonal(np.asarray(exp.cov))) for exp in exposures]
COVARIANCE = "tridiagonal" if TRIDIAGONAL else "full" if USE_COV else "diagonal"
log(f"likelihood: {COVARIANCE} slope covariance")
rows_by_key = {}
for r, exp in zip(mine, exposures):
    rows_by_key[exp.key] = r
    if exp.POS != r["pos"]:
        log(
            f"warning: {exp.key} is {exp.POS} in the model but {r['pos']} in the manifest"
        )

# fitted parameters, as the validators' in training (retrain.py validator_params)
shared_params = (
    ["spectra"]
    + (["defocus"] if g["own_defocus"] else [])
    + (["rotations"] if g["fit_rotation"] else [])
)
own_params = ["positions", "fluxes", PTT]
if val_ptt_key != "exposure":  # the PTT is then shared by the visit's exposures
    own_params.remove(PTT)
    shared_params.append(PTT)

# new validators' wavefronts start from a trained one of the same filter (and dither position)
trained_ab = set(best_state.get(PTT, {}))
abs_ = dict(model.get(PTT))
# with a fixed per-filter term, a start is the calibrator's PTT minus its filter's term (so the total starts at the
# calibrator's); a key shared by several filters (per visit) starts from the mean over its exposures
term = dict(model.params["filter_ptt"]) if "filter_ptt" in model.params else {}
starts = defaultdict(list)
for exp in exposures:
    key = exp.get_key(PTT)
    if key in trained_ab or AB_INIT == "zero":
        continue
    for source in (
        f"04481_{exp.filter}_{exp.POS}",
        f"08330_{exp.filter}_{exp.POS}",
        f"04481_{exp.filter}_POS1",
    ):
        if source in abs_:
            starts[key].append(
                abs_[source] - term[exp.filter] if term else abs_[source]
            )
            break
for key, values in starts.items():
    abs_[key] = sum(values) / len(values)
model = model.set(PTT, abs_)
log(
    f"{len(exposures)} validators built; fitting {own_params} per exposure and {shared_params} per key; "
    f"new wavefronts start from {AB_INIT}; PTT per {val_ptt_key}"
    + (
        f"; fixed per-filter PTT term, nm rms over the pupil "
        + ", ".join(
            f"{f} {float(np.sqrt(np.sum(np.asarray(v) ** 2))):.2f}"
            for f, v in sorted(term.items())
        )
        if term
        else ""
    )
)


# ---- per-exposure functions of that exposure's own parameter vector ---------------------------------
def names_of(exp, params):
    return [(p, exp.get_key(p)) for p in params]


def set_params(model, names, shapes, x):
    i = 0
    for (param, key), shape in zip(names, shapes):
        n = int(onp.prod(shape)) if shape else 1
        model = model.set(param, {**model.get(param), key: x[i : i + n].reshape(shape)})
        i += n
    return model


@eqx.filter_jit
def _value_and_grad(x, model, exp, names, shapes):
    loss = lambda x: -np.nanmean(exp.mv_zscore(set_params(model, names, shapes, x)))
    return jax.value_and_grad(loss)(x)


def _residuals(x, model, exp, names, shapes, mask):
    m = set_params(model, names, shapes, x)
    s, d = exp.to_vec(exp(m)), exp.to_vec(exp.slopes)
    if (
        exp.covariance != "diagonal"
    ):  # whitened by each pixel's Cholesky factor, so sum r^2 = (s - d)^T C^-1 (s - d) as in mv_zscore
        C = exp.to_vec(exp.cov)
        C = np.where(mask[:, None, None], C, np.eye(C.shape[-1]))
        L = np.linalg.cholesky(C)
        r = jax.vmap(lambda L, v: jax.scipy.linalg.solve_triangular(L, v, lower=True))(
            L, s - d
        )
        return np.where(mask[:, None], r, 0.0).ravel()
    var = jax.vmap(np.diag)(exp.to_vec(exp.cov))
    return np.where(mask[:, None], (s - d) / np.sqrt(var), 0.0).ravel()


_residuals_jit = eqx.filter_jit(_residuals)


@eqx.filter_jit
def _jvps(x, model, exp, names, shapes, mask, tangents):
    f = lambda x: _residuals(x, model, exp, names, shapes, mask)
    return jax.vmap(lambda t: jax.jvp(f, (x,), (t,))[1])(tangents)


class Exp:
    """One validator: its parameter names, current values and compiled loss / Gauss-Newton Fisher."""

    def __init__(self, exp, model):
        self.exp, self.key = exp, exp.key
        self.names = names_of(exp, own_params + shared_params)
        self.shapes = tuple(tuple(onp.shape(model.get(p)[k])) for p, k in self.names)
        d, var = exp.to_vec(exp.slopes), jax.vmap(np.diag)(exp.to_vec(exp.cov))
        self.mask = np.all(np.isfinite(d) & np.isfinite(var) & (var > 0), axis=1)

    def x(self, model):
        return onp.concatenate(
            [onp.ravel(onp.asarray(model.get(p)[k], float)) for p, k in self.names]
        )

    def value_and_grad(self, x, model):
        v, gr = _value_and_grad(
            np.asarray(x), model, self.exp, tuple(self.names), self.shapes
        )
        return float(v), onp.asarray(gr, float)

    def fisher(self, x, model, cols, chunk=8):
        """Gauss-Newton Fisher of the loss over the local coordinates `cols`, (len(cols), len(cols))."""
        # the Gauss-Newton residuals must reproduce the training loss, or the whitening is for another problem
        r = onp.asarray(
            _residuals_jit(
                np.asarray(x),
                model,
                self.exp,
                tuple(self.names),
                self.shapes,
                self.mask,
            ),
            float,
        )
        loss, _ = self.value_and_grad(x, model)
        if abs(onp.sum(r**2) / float(self.mask.sum()) - loss) > 1e-6 * abs(loss):
            log(
                f"warning: {self.key} residuals give loss {onp.sum(r**2) / float(self.mask.sum()):.6f}, the model {loss:.6f}"
            )
        tangents = onp.eye(len(x))[cols]
        J = onp.concatenate(
            [
                onp.asarray(
                    _jvps(
                        np.asarray(x),
                        model,
                        self.exp,
                        tuple(self.names),
                        self.shapes,
                        self.mask,
                        np.asarray(tangents[i : i + chunk]),
                    ),
                    float,
                )
                for i in range(0, len(cols), chunk)
            ]
        )
        return 2.0 * (J @ J.T) / float(self.mask.sum())  # loss = sum r^2 / n_pixels


def whitener(F, floor=1e-6):
    F = 0.5 * (F + F.T)
    lam, V = onp.linalg.eigh(F)
    lam = onp.clip(lam, floor * lam.max(), None)
    return V / onp.sqrt(lam)[None, :]


def fit(items, model, steps, free=None):
    """Minimises the summed loss of `items` (Exp) over their parameters (all, or the names in `free`),
    in whitened coordinates. Returns the model with the fitted values, and the result.
    """
    index, starts = {}, []
    for it in items:  # one coordinate block per distinct (param, key)
        for (p, k), shape in zip(it.names, it.shapes):
            if (p, k) not in index and (free is None or (p, k) in free):
                index[(p, k)] = (sum(len(s) for s in starts), shape)
                starts.append(onp.ravel(onp.asarray(model.get(p)[k], float)))
    x0 = onp.concatenate(starts)
    # each item's local vector = its fixed values, with the free blocks taken from the global vector
    maps = []
    for it in items:
        xl, src, dst, i = it.x(model), [], [], 0
        for (p, k), shape in zip(it.names, it.shapes):
            n = int(onp.prod(shape)) if shape else 1
            if (p, k) in index:
                g0 = index[(p, k)][0]
                src += list(range(g0, g0 + n))
                dst += list(range(i, i + n))
            i += n
        maps.append((xl, onp.array(src, int), onp.array(dst, int)))
    if WHITEN:
        F = onp.zeros((len(x0), len(x0)))
        for it, (xl, src, dst) in zip(items, maps):
            F[onp.ix_(src, src)] += it.fisher(xl, model, dst)
        P = whitener(F)
    else:
        P = onp.eye(len(x0))

    def local(xg, xl, src, dst):
        y = xl.copy()
        y[dst] = xg[src]
        return y

    def fun(z):
        xg = x0 + P @ z
        f, grad = 0.0, onp.zeros_like(x0)
        for it, (xl, src, dst) in zip(items, maps):
            v, gl = it.value_and_grad(local(xg, xl, src, dst), model)
            f += v
            grad[src] += gl[dst]
        return f, P.T @ grad

    res = minimize(
        fun,
        onp.zeros(P.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options=dict(maxiter=steps, ftol=1e-12, gtol=1e-8, maxcor=30),
    )
    xg = x0 + P @ res.x
    for (p, k), (i0, shape) in index.items():
        n = int(onp.prod(shape)) if shape else 1
        model = model.set(
            p, {**model.get(p), k: np.asarray(xg[i0 : i0 + n].reshape(shape))}
        )
    return model, res


def metrics(it, model):
    exp = it.exp
    slopes = onp.asarray(exp(model), float)
    data, variance = onp.asarray(exp.slopes, float), onp.asarray(exp.variance, float)
    support = onp.zeros(data.shape[-2:], bool)
    support[tuple(onp.asarray(s) for s in exp.support)] = True
    z = onp.where(support[None], (data - slopes) / onp.sqrt(variance), onp.nan)
    row, mz = residual_row(exp.key, z, data, variance)
    loss = float(-np.nanmean(exp.mv_zscore(model)))
    # loss is the fitted likelihood's; loss_diag the diagonal one (= loss when RETRAIN_USE_COV=0), so states fitted
    # under either likelihood can be compared on both
    row.update(
        loss_diag=row["loss"], loss=loss, chi2_slope=loss / (int(exp.ngroups) - 1)
    )
    return row, mz


# ---- fit each component ------------------------------------------------------------------------------
items = [Exp(e, model) for e in exposures]
by_key = {it.key: it for it in items}
results, maps = [], {}
for comp in components(
    items, lambda it: [f"{p}:{k}" for p, k in it.names if p in shared_params]
):
    t0 = time.time()
    before = {it.key: metrics(it, model) for it in comp}
    model, res = fit(comp, model, STEPS)
    after = {it.key: metrics(it, model) for it in comp}
    desc = ", ".join(
        sorted(
            {
                f"{rows_by_key[it.key]['program']} {rows_by_key[it.key]['star']}"
                for it in comp
            }
        )
    )
    log(
        f"component of {len(comp)} ({desc}): loss {sum(b[0]['loss'] for b in before.values()):.3f} -> "
        f"{sum(a[0]['loss'] for a in after.values()):.3f} in {res.nit} iterations ({res.message}), {(time.time() - t0) / 60:.1f} min"
    )
    for it in comp:
        exp, r = it.exp, rows_by_key[it.key]
        (b, _), (a, mz) = before[it.key], after[it.key]
        fitted = {
            f"{p}": onp.asarray(model.get(p)[k], float).tolist()
            for p, k in it.names
            if p != PTT
        }
        ab = onp.asarray(model.get(PTT)[exp.get_key(PTT)], float)
        results.append(
            dict(
                key=it.key,
                file=r["file"],
                program=r["program"],
                observation=r["observation"],
                star=r["star"],
                filter=exp.filter,
                pos=exp.POS,
                ngroups=int(exp.ngroups),
                nints=int(exp.nints),
                date=r["date"],
                pwcpos=r["pwcpos"],
                peak_dn=r["peak_dn"],
                beyond_training_depth=r["beyond_training_depth"],
                current=r["current"],
                processed=r["processed"],
                n_component=len(comp),
                iterations=int(res.nit),
                converged=bool(res.success),
                before=b,
                after=a,
                fitted=fitted,
                ab_rms_nm=float(onp.sqrt(onp.mean(ab**2))),
            )
        )
        maps[it.key] = mz.astype(onp.float32)

# ---- transfer: B with A's wavefront, B's position, flux and spectrum refitted ---------------------------
transfers = []
if TRANSFER:
    pairs = defaultdict(list)
    for res_row in results:
        pairs[
            (res_row["program"], res_row["date"], res_row["filter"], res_row["pos"])
        ].append(res_row)
    for group in pairs.values():
        first = {}  # one exposure per star (the first): each ordered pair of stars once
        for r in sorted(group, key=lambda r: r["key"]):
            first.setdefault(r["star"], r)
        if len(first) < 2:
            continue
        for ra in first.values():
            for rb in first.values():
                if ra["star"] == rb["star"]:
                    continue
                A, B = by_key[ra["key"]], by_key[rb["key"]]
                ab_a = model.get(PTT)[A.exp.get_key(PTT)]
                m_t = model.set(PTT, {**model.get(PTT), B.exp.get_key(PTT): ab_a})
                free = {
                    n for n in B.names if n[0] in ("positions", "fluxes", "spectra")
                }
                m_t, res = fit([B], m_t, TRANSFER_STEPS, free=free)
                row, _ = metrics(B, m_t)
                transfers.append(
                    dict(
                        calibrator=ra["key"],
                        target=rb["key"],
                        program=rb["program"],
                        date=rb["date"],
                        filter=rb["filter"],
                        pos=rb["pos"],
                        calibrator_star=ra["star"],
                        target_star=rb["star"],
                        own=rb["after"],
                        transfer=row,
                        iterations=int(res.nit),
                    )
                )
                log(
                    f"transfer {ra['star']} -> {rb['star']} ({rb['program']} {rb['filter']} {rb['pos']}): "
                    f"loss own {rb['after']['loss']:.3f}, transferred {row['loss']:.3f}"
                )

os.makedirs(OUT, exist_ok=True)
stem = os.path.join(OUT, f"shard_{SHARD:02d}_of_{NSHARD:02d}")
with open(stem + ".json", "w") as f:
    json.dump(
        dict(
            state=os.environ.get("RETRAIN_BEST_STATE"),
            manifest=MANIFEST,
            steps=STEPS,
            ab_init=AB_INIT,
            use_cov=USE_COV,
            own_params=own_params,
            shared_params=shared_params,
            val_ptt_key=val_ptt_key,
            filter_ptt=os.environ.get("RETRAIN_FILTER_PTT", "0"),
            fourier_program=FOURIER_PROGRAM
            or ("nearest of " + ",".join(FOURIER_FROM) if FOURIER_FROM else "nearest"),
            covariance=COVARIANCE,
            extra_cal=os.environ.get("RETRAIN_EXTRA_CAL"),
            validators=results,
            transfers=transfers,
        ),
        f,
        indent=1,
    )
onp.savez(stem + "_maps.npz", **maps)
log(f"saved {stem}.json ({len(results)} validators, {len(transfers)} transfers)")
# every fitted value, wavefronts included, so the fits can be redrawn without refitting
onp.savez(
    stem + "_params.npz",
    **{
        f"{p}:{k}": onp.asarray(model.get(p)[k], float)
        for it in items
        for p, k in it.names
    },
)

# ---- one summarise_fit figure per validator, named so the folder sorts by program and star -------------
if PLOTS:
    from amigo.plotting import summarise_fit

    fits_dir = os.path.join(OUT, "fits")
    for r in results:
        exp = by_key[r["key"]].exp
        try:
            summarise_fit(model, exp, save_path=fits_dir)
            os.replace(
                os.path.join(fits_dir, f"{exp.key}.png"),
                os.path.join(
                    fits_dir,
                    f"{r['program']}_{r['star']}_{r['filter']}_{r['pos']}_{exp.key}.png",
                ),
            )
        except Exception as e:  # the results are saved already
            log(f"summarise_fit failed for {exp.key}: {e!r}")
    log(f"fit figures in {fits_dir}")
