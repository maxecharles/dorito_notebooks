import numpy as onp
from jax import numpy as np, random as jr, tree as jtu
import jax
# from dorito.stats import apply_regularisers
import dLux as dl
from dLux import utils as dlu
from amigo.model_fits import PointFit
from amigo.optical_models import AMIOptics
import pandas as pd
import sys
import amigo
import matplotlib as mpl
from matplotlib import pyplot as plt
import os


def ff_reg(model, exposure, args={}, ff_std=0.035):
    ff_norm = model.FF - 1
    return np.mean((ff_norm / ff_std) ** 2)


def nl_reg(model, exposure, args={}, nl_std=0.025):
    nl_norm = model.non_linearity - model.non_linearity.mean()
    return np.mean((nl_norm / nl_std) ** 2)


def sep_reg(model, exposure, args={}, prior=0.1259, scale=0.01):
    """
    Prior on the binary separation of EZ Aquarii AC-B on Nov 11th 2025.
    """
    if exposure.star != "V-EZ-Aqr":
        return np.array(0)
    sep = model.separations[exposure.get_key("separations")]
    return -jax.scipy.stats.norm.logpdf(sep, loc=prior, scale=scale)


def pa_reg(model, exposure, args={}, prior=66.31, scale=6.6):
    """
    Prior on the binary position angle of EZ Aquarii AC-B on Nov 11th 2025.
    """
    if exposure.star != "V-EZ-Aqr":
        return np.array(0)
    pa = model.pas[exposure.get_key("pas")]
    return -jax.scipy.stats.norm.logpdf(pa, loc=prior, scale=scale)


def linear_penalty(y, eps=1e-8):
    """
    Mean squared residual from the best-fit line through y,
    assuming equispaced samples (x = 0, 1, ..., n-1).
    0 = perfectly linear (any slope, including flat).
    """
    n = y.shape[0]  # number of timesteps
    x = np.arange(n)

    y = (y - np.min(y)) / (np.max(y) - np.min(y) + eps)  # for range offset
    # y = (y - np.mean(y)) / (np.mean(y) + eps)  # for mean offset

    x_centered = x - (n - 1) / 2.0  # can just use midpoint for x mean
    y_centered = y - np.mean(y)

    Sxy = np.sum(x_centered * y_centered)
    Sxx = np.sum(x_centered ** 2)
    Syy = np.sum(y_centered ** 2)

    ss_res = Syy - (Sxy ** 2) / Sxx
    return ss_res / n


def linear_reg(model, exposure, args, return_im=False):

    # getting slopes out of args
    slopes = args["slopes"]
    slope_vec = slopes.reshape(slopes.shape[0], -1)

    # calculating linear penalty per pixel
    lin_pen = jax.vmap(linear_penalty, in_axes=1)(slope_vec)

    if return_im:
        return lin_pen.reshape(slopes.shape[-2], slopes.shape[-1])
    return np.mean(lin_pen)


def apply_regularisers(model, exposure, args):
    """Apply registered regularisers stored in ``args['reg_dict']``.

    The expected format of ``args['reg_dict']`` is a mapping to pairs
    ``(coeff, fun)`` where ``fun(model, exposure)`` returns a scalar regulariser
    value.
    """

    if "reg_dict" not in args.keys():
        return 0.0

    # evaluating the regularisation term with each for each regulariser
    priors = [coeff * fun(model, exposure, args=args) for coeff, fun in args["reg_dict"].values()]

    # summing the different regularisers
    return np.array(priors).sum()


def loss_fn(model, exposure, args={}):

    # calculating likelihood
    z_vec, slopes = exposure.mv_zscore(model, return_slopes=True)
    likelihood = -np.nanmean(z_vec)

    # applying regularisers to calculate prior
    # NOTE: The regularisers read the model parameters directly (eg ff_reg, nl_reg), so
    # they need the same gradient stopping that `simulate` applies to calibrators
    reg_model = model
    if exposure.calibrator:
        reg_model = exposure.nuke_dark_grads(exposure.nuke_pixel_grads(model))
    reg_args = {**args, "slopes": slopes}
    prior = apply_regularisers(reg_model, exposure, reg_args)

    # summing to posterior
    posterior = likelihood + prior

    aux = (likelihood, prior)
    return posterior, aux


def args_fn(model, args, epoch):
    """
    Custom args function to handle the learning
    rate warm-up and temperature decay.
    """
    args["l2"] = args["l2_schedule"][epoch]
    return model, args


def cosine_warmup(t, t0, n_max):
    # Make the cosine curve
    x = (t - t0) * np.pi / n_max
    half_cos = 0.5 * (1 + (np.cos(x + np.pi)))

    # Set all values > n to 1.
    half_cos = np.where(t > n_max + t0, 1, half_cos)
    half_cos = np.where(t < t0, 0.0, half_cos)
    return half_cos


def temp_decay(t, T0, k, TF=0):
    return (T0 - TF) * np.exp(-k * t) + TF


def tempered_adam(lr, updates_per_epoch, warmup_epochs, T0=0.0, TF=0.0, decay=0.0, seed=0):
    """
    Adam with the retrain warm-up and temperature schedules built in.

    Noise is added after Adam's normalisation, so the temperature is in units of Adam's
    per-weight step (~1), not raw gradient units, and it can't be divided back out the
    way gradient-level noise is. Schedules are in epochs: count / updates_per_epoch.
    """
    import optax

    def epoch(count):
        return count / updates_per_epoch

    def lr_fn(count):
        return lr * cosine_warmup(epoch(count), 0.0, warmup_epochs)

    def add_noise():
        def init_fn(params):
            return {"count": np.zeros([], np.int32), "key": jr.PRNGKey(seed)}

        def update_fn(updates, state, params=None):
            key, subkey = jr.split(state["key"])
            leaves, treedef = jtu.flatten(updates)
            keys = jr.split(subkey, len(leaves))
            T = temp_decay(epoch(state["count"]), T0, decay, TF)
            leaves = [u + T * jr.normal(k, u.shape, u.dtype) for u, k in zip(leaves, keys)]
            return jtu.unflatten(treedef, leaves), {"count": state["count"] + 1, "key": key}

        return optax.GradientTransformation(init_fn, update_fn)

    return optax.chain(optax.scale_by_adam(), add_noise(), optax.scale_by_learning_rate(lr_fn))


def get_warmup(args):
    return cosine_warmup(args["t"], args["t0"], args["n_max"])


def get_temperature(args):
    return temp_decay(t=args["t"], T0=args["T0"], k=args["k"], TF=args["TF"])


def grads_fn(model, grads, args):

    if "nn_weights" not in grads.params.keys():
        return grads, args

    # Get the parameters
    grad_params = grads.params

    # Get the key and update args with new key
    key, subkey = jr.split(args["key"], 2)
    args["key"] = subkey

    # Adds a temperature to the NN gradients
    values = grad_params["nn_weights"]

    rand_vals = get_temperature(args) * jr.normal(key, values.shape)
    values += rand_vals

    # Add the learning rate warm-up (we also warm up the temperature here)
    values *= args["max_lr"] * get_warmup(args)

    # Increment the t parameter
    args["t"] += 1.0 / args["n_batch"]

    # Update with the new values
    grad_params["nn_weights"] = values
    grads = grads.set("params", grad_params)
    return grads, args


def aux_fn(batch_key, aux_dict, aux):
    # Aux should have exposure keys, with values (likelihood, prior)
    for exp_key, val in aux.items():
        aux_key = (batch_key, exp_key)
        aux_dict["loglike"][aux_key].append(onp.array(val[0]))
        # NOTE "l2_reg" is just all priors
        # TODO Edit Trainer class so this is not hardcoded
        aux_dict["l2_reg"][aux_key].append(onp.array(val[1]))
    return aux_dict


def batch_looper_fn(looper, loss_dict):
    """tqdm description for the plain Trainer: mean loss plus cal/flat splits."""
    last = {k: float(onp.asarray(v[-1])) for k, v in loss_dict.items() if len(v) > 0}
    groups = {"Cal": [], "Flat": []}
    for key, value in last.items():
        for label in groups:
            if label.lower() in key:
                groups[label].append(value)
    desc = f"Loss: {onp.mean(list(last.values())):.2f}"
    for label, values in groups.items():
        if len(values) > 0:
            desc += f" | {label}: {onp.mean(values):.2f}"
    looper.set_description(desc)


def looper_fn(loss_dict, aux_dict):

    cal_losses, flat_losses, val_losses = {}, {}, {}
    for key, value in aux_dict["loglike"].items():
        batch_key, exp_key = key
        if "cal" in batch_key:
            cal_losses[key] = value
        if "flat" in batch_key:
            flat_losses[key] = value
        if "val" in batch_key:
            val_losses[key] = value

    # Only the last 1-2 values (per exposure) are ever printed, so avoid
    # rebuilding a device array from the full, ever-growing history every
    # epoch (that's O(epochs) work per call, O(epochs^2) over a full run).
    # Plain host-side numpy is also the right tool here regardless, since
    # this is pure bookkeeping for a printed string, not part of the
    # differentiable model.
    def last_and_diff(values):
        last = onp.mean([v[-1] for v in values])
        diff = last - onp.mean([v[-2] for v in values]) if len(next(iter(values))) > 1 else None
        return last, diff

    def append_str(print_str, label, values):
        if len(values) == 0:
            return print_str
        last, diff = last_and_diff(values)
        print_str += f"{label}{last:.2f}"
        if diff is not None:
            print_str += f" \u0394 {diff:.2f}"
        return print_str

    print_str = ""
    print_str = append_str(print_str, "Cal: ", cal_losses.values())
    print_str = append_str(print_str, " | Val: ", val_losses.values())
    print_str = append_str(print_str, " | Flat: ", flat_losses.values())

    # NOTE "l2_reg" is just all priors
    # TODO Edit Trainer class so this is not hardcoded
    prior = onp.array(jtu.leaves(aux_dict["l2_reg"]))
    if len(prior) > 0:
        print_str += f" | Prior: {prior[-1]:.2f}"
        if len(prior) > 1:
            print_str += f" \u0394 {onp.diff(prior)[-1]:.2f}"

    return print_str

    
class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# Visualising metadata in a table
def summarise_files(files):
    prog_ids = []
    fnames = []
    targets = []
    filts = []
    diths = []
    ngroups = []
    time = []
    pis = []
    cals = []

    for file in files:
        header = file[0].header

        prog_ids.append(header["PROGRAM"][1:])
        fnames.append(header["FILENAME"][:25])
        targets.append(header["TARGPROP"])
        filts.append(header["FILTER"])
        diths.append(f"{header["PATT_NUM"]}/{header["NUMDTHPT"]}")
        ngroups.append(f"{header["NGROUPS"]}/{header["NINTS"]}")
        time.append(header["DATE-BEG"])
        pis.append(header["PI_NAME"])
        try:
            cals.append(header["IS_PSF"])
        except KeyError:
            cals.append("FLAT")

    df = pd.DataFrame(
        {
            "program": prog_ids,
            # "filename": fnames,
            "target": targets,
            "filter": filts,
            "dither": diths,
            "g/i": ngroups,
            "date": time,
            "PI": pis,
            "CAL": cals,
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.assign(date=pd.to_datetime(df["date"]).dt.strftime("%d-%m-%Y"))

    with pd.option_context(
        "display.expand_frame_repr",
        False,
        "display.max_columns",
        None,
        "display.max_rows",
        None,
        "display.width",
        1000,
    ):
        print(df)

    # df.to_excel("cal_data.xlsx", index=False)

def get_cmap(cmap_name: str):
    cmap = mpl.colormaps[cmap_name]
    cmap.set_bad("k", 0.5)
    return cmap

    
def get_transmission_mask(optics, params, static_optics=False):
    """
    Build the pupil transmission mask for a given saved-params dict,
    using that state's own distortion/primary_beam rather than
    whatever's currently sitting on result.model.optics.
    """
    if static_optics:
        return optics.transmission

    pupil_mask = optics.pupil_mask
    if "distortion" in params:
        pupil_mask = pupil_mask.set("distortion", params["distortion"])
    if "primary_beam" in params:
        pupil_mask = pupil_mask.set("primary_beam", params["primary_beam"])
    optics = optics.set("pupil_mask", pupil_mask)

    return optics.calc_mask(optics.wf_npixels, optics.diameter)


def remove_hole_ptt(opd, mask, corners, size=180):
    """
    Subtract each hole's own least-squares piston / tip / tilt from a dense OPD,
    fitted over that hole's aperture (mask > 0.5). A pupil-wide basis (the
    eigenbasis) has no per-hole piston/tip/tilt coefficients to zero, so the
    "flat" OPD is made this way instead.
    """
    opd, mask = onp.asarray(opd), onp.asarray(mask)
    flat = opd.copy()
    yy, xx = onp.mgrid[:size, :size]
    for j, i in onp.asarray(corners):
        win = (slice(i, i + size), slice(j, j + size))
        m = mask[win] > 0.5
        P = onp.stack([onp.ones(m.sum()), xx[m], yy[m]], 1)
        coef = onp.linalg.lstsq(P, opd[win][m], rcond=None)[0]
        flat[win] -= (coef[0] + coef[1] * xx + coef[2] * yy) * m
    return flat


# How well each exposure is fitted where the PSF is bright and where it is faint. The all-pixel
# sigma of the z-scores (the statistic in the AMIGO paper's figures) is dominated by the
# background and cannot see the PSF core (2026-10-09). So the per-pixel mean z-score (the mean over
# slopes of (data - model) / sigma: the middle panel of summarise_fit and of the paper's Figs 11, 20
# and 21) is summarised as an rms in bands of pixels ranked by their flux, each against what a
# perfect model would leave, and for CAL 4481 against the paper's own maps (read back from its
# figures by paper_residual_maps.py, the five dithers = keys ending _1 ... _5).
PAPER_RESIDUAL_MAPS = "/fred/oz440/max/outputs/degeneracy/paper_residual_maps.npz"
PAPER_TAGS = {"04481_001_04_03_": "F380M", "04481_001_06_03_": "F430M", "04481_001_02_03_": "F480M"}
RESIDUAL_BANDS = (("core50", 0, 50), ("core300", 0, 300), ("mid", 300, 1000), ("bg", 3000, None))


def mean_z_noise(variance):
    """The rms that a perfect model would leave in the mean over slopes of z, per pixel:
    neighbouring slopes share a read (covariance -r), photon noise is independent. r, the read
    variance, is taken from the faintest pixels, where variance = 2 r."""
    n = variance.shape[0]
    faint = onp.nanmedian(variance, axis=0)
    r = 0.5 * onp.nanmedian(faint[faint <= onp.nanpercentile(faint, 20)])
    s = onp.sqrt(variance)
    cross = (1 / (s[:-1] * s[1:])).sum(0)
    return onp.sqrt(onp.clip(n - 2 * r * cross, 1e-12, None)) / n


def residual_row(key, z, data, variance, paper_maps=None):
    """One exposure's numbers from its z-scores, data and variance (slopes, 80, 80); z is NaN
    wherever the likelihood is not evaluated. Returns the row and the mean-z map."""
    z = onp.asarray(z, float)
    ok = onp.isfinite(z).all(0)
    mz = onp.where(ok, onp.nanmean(onp.where(ok[None], z, 0.0), 0), onp.nan)
    paper = None
    for tag, filt in PAPER_TAGS.items():
        if paper_maps is not None and key.startswith(tag) and key[len(tag):].isdigit():
            paper = paper_maps[f"{filt}_mean_z"][int(key[len(tag):]) - 1]
    use = ok & (onp.isfinite(paper) if paper is not None else True)
    flux = onp.where(use, onp.nansum(onp.asarray(data, float), 0), -onp.inf).ravel()
    order = onp.argsort(-flux)[: int(use.sum())]  # brightest first
    noise = mean_z_noise(onp.asarray(variance, float))
    rms = lambda x: float(onp.sqrt(onp.nanmean(x**2)))
    row = dict(key=key, n_pix=int(use.sum()), sigma=float(onp.nanstd(z[:, ok])),
               loss=float(onp.mean(onp.sum(z[:, ok] ** 2, 0))))
    for band, a, b in RESIDUAL_BANDS:
        idx = order[a:b]
        row[band] = rms(mz.ravel()[idx])
        row[f"noise_{band}"] = rms(noise.ravel()[idx])
        row[f"paper_{band}"] = rms(paper.ravel()[idx]) if paper is not None else onp.nan
    return row, mz


def residual_stats(model, exposures, save_path=None, name="residual_stats", title=""):
    """Residual statistics of (kind, exposure) pairs at `model`: printed, and saved as
    <name>.npz (rows and mean-z maps) and <name>.txt in save_path."""
    try:
        paper_maps = dict(onp.load(PAPER_RESIDUAL_MAPS, allow_pickle=True))
    except Exception as e:
        print(f"No paper residual maps ({e}); comparing without them")
        paper_maps = None
    rows, maps = [], {}
    for kind, exp in exposures:
        slopes = onp.asarray(exp(model), float)
        data, variance = onp.asarray(exp.slopes, float), onp.asarray(exp.variance, float)
        support = onp.zeros(data.shape[-2:], bool)
        support[tuple(onp.asarray(s) for s in exp.support)] = True
        z = onp.where(support[None], (data - slopes) / onp.sqrt(variance), onp.nan)
        row, mz = residual_row(exp.key, z, data, variance, paper_maps)
        row.update(kind=kind, program=exp.program, filter=exp.filter, ngroups=int(exp.ngroups))
        rows.append(row)
        maps[exp.key] = mz.astype(onp.float32)
    lines = format_residual_rows(rows, title)
    print("\n".join(lines))
    if save_path is not None:
        onp.savez(os.path.join(save_path, f"{name}.npz"), rows=onp.array(rows, dtype=object), **maps)
        with open(os.path.join(save_path, f"{name}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
    return rows


def format_residual_rows(rows, title=""):
    f3 = lambda x: "  -  " if not onp.isfinite(x) else f"{x:5.3f}"
    lines = [f"Residual statistics{': ' + title if title else ''}. rms of the per-pixel mean z-score (mean over slopes of"
             " (data - model) / sigma) over the pixels ranked by flux: brightest 50 and 300, 300-1000, faintest half (bg);"
             " 'noise' is what a perfect model would leave; 'paper' is the AMIGO paper's fit of CAL 4481. sigma is the"
             " all-pixel z-score sigma of the paper's figures, loss the training loss.",
             f"{'kind':4s} {'key':20s} {'filter':6s} {'loss':>7s} {'sigma':>5s} | {'core50':>6s} {'core300':>7s} {'mid':>5s} {'bg':>5s}"
             f" | noise {'core300':>7s} {'bg':>5s} | paper {'core300':>7s} {'bg':>5s}"]
    for r in rows:
        lines.append(f"{r['kind']:4s} {r['key']:20s} {r['filter']:6s} {r['loss']:7.3f} {r['sigma']:5.3f} | {f3(r['core50']):>6s}"
                     f" {f3(r['core300']):>7s} {f3(r['mid']):>5s} {f3(r['bg']):>5s} | {'':5s} {f3(r['noise_core300']):>7s}"
                     f" {f3(r['noise_bg']):>5s} | {'':5s} {f3(r['paper_core300']):>7s} {f3(r['paper_bg']):>5s}")
    lines.append("Means per program and filter (core300 and bg; ratio to the noise floor in brackets):")
    for prog in sorted({r["program"] for r in rows}):
        for filt in ("F380M", "F430M", "F480M"):
            sel = [r for r in rows if r["program"] == prog and r["filter"] == filt]
            if not sel:
                continue
            m = lambda k: float(onp.mean([r[k] for r in sel]))
            text = (f"  {prog} {filt} ({len(sel):2d}): core300 {m('core300'):.3f} ({m('core300') / m('noise_core300'):.1f}x noise),"
                    f" bg {m('bg'):.3f} ({m('bg') / m('noise_bg'):.1f}x)")
            if onp.isfinite(m("paper_core300")):
                text += (f"; paper core300 {m('paper_core300'):.3f}, bg {m('paper_bg'):.3f};"
                         f" ours/paper core300 {m('core300') / m('paper_core300'):.2f}, bg {m('bg') / m('paper_bg'):.2f}")
            lines.append(text)
    return lines


def with_carried(params, carry_params):
    """params plus whatever the run does not use but a later run may need (e.g. the binary's
    parameters in a run without the binary): missing parameters, and missing keys of dict ones."""
    params = dict(params)
    for key, value in carry_params.items():
        if key not in params:
            params[key] = value
        elif isinstance(value, dict) and isinstance(params[key], dict):
            params[key] = {**value, **params[key]}
    return params


def summarise_fn(
    result,
    save_path,
    val_flag=False,
    cal_flag=True,
    binary_flag=False,
    save_flag=False,
    flat_flag=False,
    dark_flag=False,
    static_optics=False,
    cal_exposures=[],
    val_exposures=[],
    flat_exposures=[],
    dark_exposures=[],
    calpsf_exposures=[],
    calbin_exposures=[],
    badpix=None,
    n_batch=None,
    optimisers={},
    amigo_files_path="",
    history_stride=1,
    carry_params={},
    ):

    inferno_r = get_cmap("inferno_r")
    inferno = get_cmap("inferno")
    seismic = get_cmap("seismic")

    badpix_bool = badpix.astype(bool)

    params_to_save = [
        "fluxes",
        "positions",
        "aberrations",
        "nn_weights", 
        "distortion",
        "primary_beam",
        "defocus",
        "sigma",
        "dark_current",
        "non_linearity",
        "FF",
        "spectra",
    ]

    if flat_flag:
        params_to_save += ["flat_coeffs"]
    
    if binary_flag:
        params_to_save += ["pas", "separations", "contrasts"]

    # only there when a run fits them (retrain.py's RETRAIN_ROTATION_KEY / RETRAIN_SIGMA_KEY)
    if "rotations" in result.model.params.keys():
        params_to_save += ["rotations"]
    if "sigmas" in result.model.params.keys():
        params_to_save += ["sigmas"]

    ################### SAVING RESULTS ###################
    if val_flag:
        # Unwrapping into cal, val, and flat
        aux_history = jtu.map(
            lambda x: np.array(x), result.aux, is_leaf=lambda x: isinstance(x, list)
        )
        
        cal, val, flat = [], [], []
        for (batch_key, exp_key), value in aux_history["loglike"].items():
            if "cal" in batch_key:
                cal.append(value)
            if "val" in batch_key:
                val.append(value)
            if "flat" in batch_key:
                flat.append(value)
        cal = np.array(cal)
        val = np.array(val)
        flat = np.array(flat)
    
        # Finding BEST STATE from the fit
        mean_val = np.array(val).mean(0)  # mean loss for validators
        best = mean_val.min()  # best state is where the validator loss was minimum
        idx = np.where(mean_val == best)[0][0]
        test_aux = jtu.map(
            lambda x: x[: idx + 1], result.aux, is_leaf=lambda x: isinstance(x, list)
        )
        print(f"Best: {idx}")
        print(looper_fn(result.losses, test_aux))

    optics = result.model.optics  # base optics; pupil_mask gets overridden per-state below

    # best_state/final_state save to different locations/filenames depending on save_flag,
    # but are otherwise identical, so this is shared rather than duplicated per-branch.
    out_dir = save_path if save_flag else amigo_files_path
    prefix = "" if save_flag else "scratch_"

    try:
        best_params = result.best_state.params
        if "nn_weights" in best_params.keys():
            # result.best_batch["nn_weights"] is already averaged over the winning epoch's
            # own batches (ValBatchedTrainer does this at the point best_val updates, same
            # convention as final_state's `history[-n_batch:].mean(0)` below -- falling back
            # to a plain snapshot if that averaging itself ever fails), so it's already the
            # correct (8640,) vector -- no further reduction needed. This used to be
            # `.mean(0)`'d again here, back when best_batch was still a single unaveraged
            # snapshot; that averaged over the 8640 weights themselves instead of over
            # batches, collapsing it to a 0-d scalar and crashing any run that loads
            # best_state.npy (Trainer.grads_fn's lax_slice needs a 1-d vector).
            best_params["nn_weights"] = np.array(result.best_batch["nn_weights"])
        # best_state only holds the optimised params, so anything frozen for this run
        # (e.g. nn_weights under RETRAIN_FREEZE_NN) is filled in from the model; without
        # this, a run started from this best_state would silently fall back to defaults.
        for key in params_to_save:
            if key not in best_params.keys():
                best_params[key] = result.model.get(key)
        best_params["transmission"] = get_transmission_mask(
            optics, best_params, static_optics=static_optics
        )
        best_params = with_carried(best_params, carry_params)
        np.save(
            os.path.join(out_dir, f"{prefix}best_state.npy"),
            best_params,
            allow_pickle=True,
        )
    except Exception as e:
        print(f"Saving best state failed: {e}")

    final_params = {key: result.model.get(key) for key in params_to_save}
    if "nn_weights" in result.state.params.keys():
        # Normally n_batch (one batch_history entry per cal/flat batch). Under
        # amigo-multigpu's RETRAIN_MULTI_GPU=1 + val_flag=True path, nn_weights
        # instead gets one accumulated-gradient update per EPOCH, so
        # batch_history only has 1 entry/epoch there -- meta_data says which,
        # falling back to n_batch for results without it (plain Trainer, or an
        # older ValBatchedTrainer that predates this field).
        n_hist = result.meta_data.get("batch_updates_per_epoch", n_batch)
        final_params["nn_weights"] = np.array(
            result.history["nn_weights"]  # all batches (or the single update) of the final epoch
        )[-n_hist:].mean(0)
    final_params["transmission"] = get_transmission_mask(
        optics, final_params, static_optics=static_optics
    )
    final_params = with_carried(final_params, carry_params)
    np.save(
        os.path.join(out_dir, f"{prefix}final_state.npy"),
        final_params,
        allow_pickle=True,
    )

    
    ################## PLOTTING LOSSES ###################

    losses = list(result.losses.values())[0]
    n_epoch = len(losses)
    start = int(0.1 * n_epoch)
    stop = -1
    if stop < 0:
        stop = n_epoch + stop
    
    if val_flag:
        
        if start >= cal.shape[-1]:
            start = 1
        
        if stop < 0:
            stop = cal.shape[-1] + stop
        xs = np.arange(start, stop)
        
        plt.figure(figsize=(18, 4))
        ax = plt.subplot(1, 3, 1)
        if cal_flag or binary_flag:
            plt.plot(xs, np.array(cal).mean(0)[xs])
            ax.set(title="Calibrators", xlabel="Epochs", ylabel="Loss")
        
        ax = plt.subplot(1, 3, 2)
        ax.set(title="Validators", xlabel="Epochs", ylabel="Loss")
        plt.plot(xs, np.array(val).mean(0)[xs])
        
        ax = plt.subplot(1, 3, 3)
        ax.set(title="Flat", xlabel="Epochs", ylabel="Loss")
        if flat_flag:
            plt.plot(xs, np.array(flat).mean(0)[xs])
        
        plt.tight_layout()
        if save_flag:
            plt.savefig(os.path.join(save_path, "mean_losses.png"), dpi=200)
            plt.close()
        else:
            plt.show()
        
        ###
        
        plt.figure(figsize=(18, 4))
        ax = plt.subplot(1, 3, 1)
        ax.set(title="Calibrators", xlabel="Epochs", ylabel="Loss")
        [plt.plot(xs, ys[xs]) for ys in cal]
        
        ax = plt.subplot(1, 3, 2)
        ax.set(title="Validators", xlabel="Epochs", ylabel="Loss")
        [plt.plot(xs, ys[xs]) for ys in val]
        
        ax = plt.subplot(1, 3, 3)
        ax.set(title="Flat", xlabel="Epochs", ylabel="Loss")
        if flat_flag:
            [plt.plot(xs, ys[xs]) for ys in flat]
        
        plt.tight_layout()
        if save_flag:
            plt.savefig(os.path.join(save_path, "all_losses.png"), dpi=200)
            plt.close()
        else:
            plt.show()
    
    else:
        # Plot every batch plus the mean (what tqdm reports), not just the first batch
        all_losses = onp.array([onp.asarray(v) for v in result.losses.values()])
        n_epoch = all_losses.shape[-1]
        start = start if start < n_epoch else 0
        xs = onp.arange(start, n_epoch)

        plt.figure(figsize=(16, 5))
        ax = plt.subplot(1, 2, 1)
        ax.set(title="Mean loss (all batches)", xlabel="Epochs", ylabel="Loss")
        plt.plot(all_losses.mean(0))
        ax = plt.subplot(1, 2, 2)
        ax.set(title=f"Per-batch loss (from epoch {start})", xlabel="Epochs", ylabel="Loss")
        colours = plt.get_cmap("tab20")(onp.linspace(0, 1, 20))
        for i, (key, ys) in enumerate(zip(result.losses.keys(), all_losses)):
            plt.plot(xs, ys[xs], label=str(key), color=colours[i % 20], ls="-" if i < 20 else "--")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "losses.png"))
            onp.savez(
                os.path.join(save_path, "losses.npz"),
                **{str(k): onp.asarray(v) for k, v in result.losses.items()},
            )
        plt.close()
        

    ################### PLOTTING HISTORY AND SUMMARISE FIT ###################
    # The same histories as numbers, for everything but the network and the pixel maps:
    # {param: array or {key: array}}, first axis = one entry every `history_stride` epochs.
    # Lets how parameters move together be looked at afterwards (degeneracy_analysis.py).
    try:
        big = ("nn_weights", "FF", "non_linearity", "dark_current")
        param_history = {
            key: jtu.map(lambda x: onp.asarray(x), val, is_leaf=lambda x: isinstance(x, list))
            for key, val in result.history.params.items()
            if key not in big
        }
        onp.save(
            os.path.join(out_dir, f"{prefix}param_history.npy"),
            {"stride": history_stride, "params": param_history},
            allow_pickle=True,
        )
    except Exception as e:
        print(f"Saving parameter histories failed: {e}")

    amigo.plotting.plot(result.history, save_path=save_path, stride=history_stride)
    
    exposures_lists = []
    exp_types = []
    if cal_flag or binary_flag:
        exp_types += ["cal"]
        exposures_lists += [cal_exposures]
    if val_flag:
        exp_types += ["val"]
        exposures_lists += [val_exposures]
    if flat_flag:
        exp_types += ["flat"]
        exposures_lists += [flat_exposures]
    if dark_flag:
        exp_types += ["dark"]
        exposures_lists += [dark_exposures]

    for exp_type, exps in zip(exp_types, exposures_lists):
        print(5*"\n")
        print(exp_type)
        # if not (cal_flag or binary_flag) and exp_type != "flat":
        #     continue
        if save_path is not None:
            this_save_path = os.path.join(save_path, exp_type)
            os.mkdir(this_save_path)
        else:
            this_save_path = None
        for exp in exps:
            exp.print_summary()
            amigo.plotting.summarise_fit(result.model, exp, save_path=this_save_path)

    # core / background residual statistics of every calibrator and validator (residual_stats)
    try:
        residual_stats(
            result.model,
            ([("cal", e) for e in cal_exposures] if (cal_flag or binary_flag) else [])
            + ([("val", e) for e in val_exposures] if val_flag else []),
            save_path=save_path,
            title=os.path.basename(os.path.normpath(save_path)) if save_path else "",
        )
    except Exception as e:
        print(f"Residual statistics failed: {e}")

    ################### PIXEL SENSITIVITY AND NON-LINEARITY ###################
    
    FF = result.model.FF.at[badpix_bool].set(np.nan)
    non_lin = result.model.non_linearity[0].at[badpix_bool].set(np.nan)
    
    ff_xs = np.linspace(np.nanmin(FF), np.nanmax(FF), 100) - 1
    ff_l2 = np.exp(-((ff_xs / 0.035) ** 2))
    
    med_nl = np.nanmedian(non_lin)
    nl_xs = np.linspace(np.nanmin(non_lin), np.nanmax(non_lin), 100)
    nl_l2 = np.exp(-(((nl_xs - med_nl) / 0.025) ** 2))
    
    
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # Pixel sensitivity
    ax1.set_title("Sensitivity (FF)")
    im1 = ax1.imshow(FF, seismic, norm=mpl.colors.CenteredNorm(1))
    fig.colorbar(im1, ax=ax1)
    
    # Linear (imshow)
    ax2.set_title("Non-linearity")
    im2 = ax2.imshow(non_lin, inferno_r, vmax=None)
    fig.colorbar(im2, ax=ax2)
    
    # FF histogram
    ax3.set_title("FF")
    ax3.hist(FF[::2].flatten(), bins=100, density=True)
    ax3.hist(FF[1::2].flatten(), bins=100, alpha=0.75, density=True)
    ax3.plot(ff_xs + 1, 25 * ff_l2, label="L2", c="k")
    
    # Linear histogram
    ax4.set_title("non-linearity")
    ax4.hist(non_lin[::2].flatten(), bins=100)
    ax4.hist(non_lin[1::2].flatten(), bins=100, alpha=0.75)
    ax4.plot(nl_xs, 150 * nl_l2, label="L2", c="k")
    
    fig.tight_layout()
    if save_flag:
        plt.savefig(os.path.join(save_path, "flats.png"), dpi=300)
        plt.close()
    else:
        plt.show()
    
    
    ##################### DARKS #####################
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    # NOTE: previously used the badpix of whichever exposure happened to be
    # last in the "PLOTTING HISTORY" loop above; that loop can be empty
    # (cal_flag/val_flag/flat_flag all False), so use the passed-in badpix
    # explicitly instead of relying on the leftover loop variable.
    dark_current = np.where(badpix_bool, np.nan, result.model.dark_current)
    
    im = ax[0].imshow(dark_current, inferno)
    ax[0].set(title="Per Pixel Dark Current")
    ax[0].axis("off")
    fig.colorbar(im)
    
    ax[1].hist(dark_current.ravel(), bins=100)
    ax[1].set(title="Histogram")
    ax[1].axvline(np.nanmean(dark_current), label=f"mean={np.nanmean(dark_current):.3f}", color="red")
    ax[1].axvline(np.nanmedian(dark_current), label=f"median={np.nanmedian(dark_current):.3f}", color="yellow")
    ax[1].legend()
    
    fig.tight_layout()
    if save_flag:
        plt.savefig(os.path.join(save_path, "darks.png"), dpi=300)
        plt.close()
    else:
        plt.show()

    
    ################### WAVEFRONT ###################

    pupil_mask = optics.pupil_mask
    
    rms = lambda x: np.sqrt(np.nanmean(np.square(x)))
        
    if "aberrations" in optimisers.keys():
        for key, coeffs in result.state.aberrations.items():
    
            # trying to get the program and filter strings
            key_split = key.split("_")
            if len(key_split) > 2:
                filt, prog = key_split[0], key_split[1]
            elif len(key_split) == 2:
                prog, filt = key_split
    
            # plotting
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    
            full_abb = pupil_mask.set("abb_coeffs", coeffs).calc_aberrations()

            if static_optics:
                mask = optics.transmission
            else:
                mask = pupil_mask.calc_mask(optics.wf_npixels, optics.diameter)

            # Per-hole bases: zero each hole's piston/tip/tilt modes. Pupil-wide
            # eigenbasis: fit and remove each hole's piston/tip/tilt instead.
            if coeffs.ndim == 2:
                flat_abb = pupil_mask.set("abb_coeffs", coeffs.at[:, :3].set(0)).calc_aberrations()
            else:
                flat_abb = remove_hole_ptt(full_abb, mask, pupil_mask.corners)

            full_abb = np.where(mask < 1.0, np.nan, 1e9 * full_abb)
            flat_abb = np.where(mask < 1.0, np.nan, 1e9 * flat_abb)
        
            full_abb -= np.nanmean(full_abb)
            flat_abb -= np.nanmean(flat_abb)
        
            ax[0].set_title(f"{key} — Full OPD (RMS: {rms(full_abb):.2f} nm)")
            im_top = ax[0].imshow(full_abb, cmap=seismic, norm=mpl.colors.CenteredNorm())
            fig.colorbar(im_top, ax=ax[0], label="OPD (nm)")
        
            ax[1].set_title(f"{key} — FLAT OPD (RMS: {rms(flat_abb):.2f} nm)")
            im_bot = ax[1].imshow(flat_abb, cmap=seismic, norm=mpl.colors.CenteredNorm())
            fig.colorbar(im_bot, ax=ax[1], label="OPD (nm)")
    
    
            fig.tight_layout()
            if save_flag:
                if save_path is not None:
                    this_save_path = os.path.join(save_path, "wavefronts/")
                    if not os.path.exists(this_save_path):
                        os.mkdir(this_save_path)
                else:
                    this_save_path = None
                    
                plt.savefig(os.path.join(this_save_path, f"{prog}_{filt}_{key}.png"), dpi=300)
                plt.close()
            else:
                plt.show()
            

    
    ################### PUPIL AND BEAM DISTORTIONS ###################

    if not static_optics:
        null_pupil_mask = pupil_mask.multiply("primary_beam", 0.0).multiply("distortion", 0.0)
        null_mask = null_pupil_mask.calc_mask(optics.wf_npixels, optics.diameter)
        
        mask = pupil_mask.calc_mask(optics.wf_npixels, optics.diameter)
        
    else:
        raw_optics = AMIOptics(static=False)
        null_pupil_mask = raw_optics.pupil_mask.multiply("primary_beam", 0.0).multiply("distortion", 0.0)
        null_mask = null_pupil_mask.calc_mask(optics.wf_npixels, optics.diameter)

        mask = optics.transmission
        
    fig, ax = plt.subplots(figsize=(3, 2))
    
    ax.set_title("Pupil & Beam Distortions")
    im = ax.imshow(mask - null_mask, cmap="berlin", norm=mpl.colors.CenteredNorm())
    fig.colorbar(im, ax=ax)
    
    fig.tight_layout()
    if save_flag:
        plt.savefig(os.path.join(save_path, "pupil_beam.png"), dpi=300)
        plt.close()
    else:
        plt.show()

        

    
    ################### SLOPE PLOTS ###################

    # Create the cal exposure dictionaries
    cal_dict = {}
    for exp in calpsf_exposures:
        if exp.filter not in cal_dict:
            cal_dict[exp.filter] = []
        cal_dict[exp.filter].append(exp)
    
    # Create the val exposure dictionaries
    val_dict = {}
    for exp in val_exposures:
        if exp.filter not in val_dict:
            val_dict[exp.filter] = []
        val_dict[exp.filter].append(exp)


    from amigo.misc import convert_adjacent_to_true
    
    if save_flag:
        this_save_path = os.path.join(save_path, "slope_res")
        os.mkdir(this_save_path)
    
    exp_types = ["cal", "val"] if val_flag else ["cal"]
    exp_dicts = [cal_dict, val_dict] if val_flag else [cal_dict]
    
    for typ, dic in zip(exp_types, exp_dicts):
        for idx, (filt, _) in enumerate(dic.items()):
            n = 4
            k = 2 * n + 1
        
            for exp in dic[filt]:
    
                print(exp)
        
                slopes = exp(result.model)
                im = slopes.sum(0)
                peak_pix = im == np.nanmax(im)
                peak_map = convert_adjacent_to_true(peak_pix, corners=True, n=n)

                slope_vec = slopes.reshape(slopes.shape[0], -1)    
                linpen = jax.vmap(linear_penalty, in_axes=1)(slope_vec).reshape(slopes.shape[-2], slopes.shape[-1])
        
                ######### plotting #########
                max_loc = np.argwhere(peak_pix)[0][::-1]
                
                fig, ax = plt.subplots(1, 2, figsize=(6, 2))
                c = ax[0].imshow(im, "cividis", norm=mpl.colors.PowerNorm(0.5))
                ax[0].set(title=f"Cropping: {filt}")
                fig.colorbar(c, ax=ax[0], label="Counts")
                ax[1].set(title=f"Linear Penalty: {filt}")
                c = ax[1].imshow(linpen, "plasma", norm=mpl.colors.PowerNorm(.5))
                fig.colorbar(c, ax=ax[1])
                for i, color in zip([0, 1], ["r", "g"]):
                    square = mpl.patches.Rectangle(max_loc - np.array([k/2, k/2]), k, k, color=color, fill=False)
                    ax[i].axis("off")
                    ax[i].add_patch(square)
                if save_flag:
                    plt.savefig(os.path.join(this_save_path, f"{typ}_crop_{filt}.png"), dpi=300)
                    plt.close()
                else:
                    plt.show()
                ############################
        
                ind = np.where(peak_map)
                model_slopes = slopes[:, *ind].reshape(-1, k, k)
                data_slopes = exp.slopes[:, *ind].reshape(-1, k, k)
                data_std = exp.variance[:, *ind].reshape(-1, k, k) ** 0.5
        
                xs = np.arange(len(slopes))
        
                fig, axes = plt.subplots(k, k, figsize=(3 * k, 3 * k), sharex="col")
                # fig.suptitle(filt)
        
                for i in range(k):
                    for j in range(k):
                        ax = axes[i, j]
        
                        if i == k - 1:
                            ax.set_xlabel("Slope Index")
                        else:
                            ax.tick_params(labelbottom=False)
                        if j == 0:
                            ax.set_ylabel("Slope (e- / group)")
        
                        ax.errorbar(
                            xs, data_slopes[:, i, j], yerr=data_std[:, i, j],
                            marker='o', capsize=5, label="Data",
                        )
                        ax.errorbar(
                            xs, model_slopes[:, i, j],
                            marker='x', capsize=5, label="Model"
                        )
                        ax.legend()
        
                # fig.tight_layout()
                if save_flag:
                    plt.savefig(os.path.join(this_save_path, f"{typ}_slope_{filt}.png"), dpi=300)
                    plt.close()
                else:
                    plt.show()

    # Defensive: every figure above should already be closed when save_flag, but
    # this got missed for 4 of them (mean_losses/all_losses/flats/darks -- all now
    # fixed) without ever erroring, just silently leaking. Belt and braces against
    # the same mistake creeping back in on a future edit -- summarise_fn runs once
    # per checkpoint on long runs, so a missed close() compounds badly. No-op if
    # save_flag is False (plt.show() in a real notebook keeps the figure around on
    # purpose, so don't close those).
    if save_flag:
        plt.close("all")


nn_setup_options = [
    {"hidden_width": 16, "n_hidden_layers": 3},
    {"hidden_width": 14, "n_hidden_layers": 3},
    {"hidden_width": 12, "n_hidden_layers": 3},
    {"hidden_width": 10, "n_hidden_layers": 3},
    {"hidden_width": 8, "n_hidden_layers": 3},
    {"hidden_width": 6, "n_hidden_layers": 3},
    {"hidden_width": 16, "n_hidden_layers": 2},
    {"hidden_width": 14, "n_hidden_layers": 2},
    {"hidden_width": 12, "n_hidden_layers": 2},
    {"hidden_width": 10, "n_hidden_layers": 2},
    {"hidden_width": 8, "n_hidden_layers": 2},
    {"hidden_width": 6, "n_hidden_layers": 2},
]
