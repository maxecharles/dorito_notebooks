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
    reg_args = {**args, "slopes": slopes}
    prior = apply_regularisers(model, exposure, reg_args)

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
            best_params["nn_weights"] = np.array(
                result.best_batch["nn_weights"]
            ).mean(0)
        best_params["transmission"] = get_transmission_mask(
            optics, best_params, static_optics=static_optics
        )
        np.save(
            os.path.join(out_dir, f"{prefix}best_state.npy"),
            best_params,
            allow_pickle=True,
        )
    except Exception as e:
        print(f"Saving best state failed: {e}")

    final_params = {key: result.model.get(key) for key in params_to_save}
    if "nn_weights" in result.state.params.keys():
        final_params["nn_weights"] = np.array(
            result.history["nn_weights"]  # all batches of final epoch
        )[-n_batch:].mean(0)
    final_params["transmission"] = get_transmission_mask(
        optics, final_params, static_optics=static_optics
    )
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
    amigo.plotting.plot(result.history, save_path=save_path)
    
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
            flat_abb = pupil_mask.set("abb_coeffs", coeffs.at[:, :3].set(0)).calc_aberrations()
            
            if static_optics:
                mask = optics.transmission
            else:
                mask = pupil_mask.calc_mask(optics.wf_npixels, optics.diameter)
        
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
