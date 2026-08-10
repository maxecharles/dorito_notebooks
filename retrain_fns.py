import numpy as onp
from jax import numpy as np, random as jr, tree as jtu
from dorito.stats import apply_regularisers
import dLux as dl
from dLux import utils as dlu
from amigo.model_fits import ModelFit, PointFit
from amigo.optical_models import AMIOptics
import pandas as pd
import sys
import amigo
import matplotlib as mpl
from matplotlib import pyplot as plt
import os


class DarkFit(ModelFit):

    def __init__(self, file, fit_one_on_fs=False, **kwargs):
        file[0].header["IS_PSF"] = False

        super().__init__(file, **kwargs)
        self.star = "NIS_DARK"
        self.observation = "DARK"
        self.program = "DARK"
        self.fit_one_on_fs = fit_one_on_fs
        self.fit_reflectivity = False
        self.fit_bias = False
        self.validator = False

    def print_summary(self):
        print(
            f"File {self.key}\n"
            f"Star {self.star}\n"
            f"nints {self.nints}\n"
            f"ngroups {len(self.slopes)+1}\n"
        )

    def initialise_params(self, optics, vis_model=None, one_on_fs_order=1):
        params = {}

        im = np.where(self.badpix, np.nan, self.slopes[0])
        # psf = np.where(np.isnan(im), 0.0, im)

        # dark current
        params["dark_A"] = (
            self.get_key("dark_A"),
            np.array(0.443),
        )

        # One on fs
        if self.fit_one_on_fs:
            params["one_on_fs"] = (
                self.get_key("one_on_fs"),
                np.zeros((self.ngroups, 80, one_on_fs_order + 1)),
            )

        return params

    @property
    def key(self):
        return "_".join(["dark", str(self.ngroups)])

    def model_illuminance(self, model):
        """
        There is no illuminance! Haha!
        """
        # Get the pixel scale (arcseconds)
        pixel_scale = model.optics.psf_pixel_scale / model.optics.oversample
        npix = model.optics.psf_npixels * model.optics.oversample

        # illuminance is just zeros
        illuminance = np.zeros((npix, npix))

        # Make the object and return
        return dl.PSF(illuminance, dlu.arcsec2rad(pixel_scale))

    def simulate(self, model, return_slopes=False):
        illuminance = self.model_illuminance(model)
        ramp = self.model_ramp(illuminance, model)
        ramp = self.model_read(ramp, model)

        if return_slopes:
            return ramp.set("data", np.diff(ramp.data, axis=0))
        return ramp

        
class BinaryFit(PointFit):

    sub_exps: dict
    unique_params: list

    def __init__(self, file, unique_params=None, calibrator=True):

        super().__init__(file)

        # OVERIDE self.calbrator
        self.calibrator = calibrator

        self.sub_exps = {
            "A": PointFit(file),
            "B": PointFit(file),
        }

        if unique_params is None:
            unique_params = [
                "spectra",
            ]
        self.unique_params = unique_params

    def initialise_params(self, optics, one_on_fs_order=1):
        params = super().initialise_params(optics, one_on_fs_order)
        params["pas"] = (self.get_key("pas"), np.array(0.0))  # degrees
        params["separations"] = (self.get_key("separations"), np.array(0.1))
        params["contrasts"] = (self.get_key("contrasts"), np.array(0.5))
        for param, (key, value) in params.items():
            if param in self.unique_params:
                params[param] = key, np.array(2 * [value])  # one for each source

        return params

    def get_key(self, param):
        if param in ["pas"]:
            # return self.key
            return self.star
        if param in ["separations"]:
            return self.star
        if param in ["contrasts"]:
            return "_".join([self.star, self.filter])
        return super().get_key(param)

    def map_param(self, param):
        if param in ["pas", "separations", "contrasts"]:
            return f"{param}.{self.get_key(param)}"
        return super().map_param(param)

    def model_interferogram(self, model):

        mean_pos = model.positions[self.get_key("positions")]
        total_flux = 10 ** model.fluxes[self.get_key("fluxes")]

        # Converting position angle from deg to radians
        # and offsetting by JWST Parallactic Angle
        pa = dlu.deg2rad(model.pas[self.get_key("pas")] - self.parang)
        separation = model.separations[self.get_key("separations")]
        contrast = model.contrasts[self.get_key("contrasts")]

        # separation vector d in radians
        d = separation * np.array([np.cos(pa), np.sin(pa)])  # in radians

        posA = mean_pos - d / 2  # brighter source
        posB = mean_pos + d / 2  # dimmer source

        logfluxA = np.log10(contrast * total_flux)
        logfluxB = np.log10((1 - contrast) * total_flux)

        modelA = model.set(self.map_param("positions"), posA).set(
            self.map_param("fluxes"), logfluxA
        )
        modelB = model.set(self.map_param("positions"), posB).set(
            self.map_param("fluxes"), logfluxB
        )

        # unpacking the unique parameters for each source
        for param in self.unique_params:
            pA, pB = model.get(self.map_param(param))
            modelA = modelA.set(self.map_param(param), pA)
            modelB = modelB.set(self.map_param(param), pB)

        models = [modelA, modelB]

        # TODO Vectorise this?
        illuminances = []
        for m in [modelA, modelB]:
            psf = self.model_psf(m)
            illuminance = self.model_illuminance(psf, m)
            illuminances.append(illuminance.data)

        illuminance = dl.PSF(np.array(illuminances).sum(axis=0), psf.pixel_scale)

        ramp = self.model_ramp(illuminance, model)
        ramp = self.model_read(ramp, model)

        return ramp

    def simulate(self, model, return_slopes: bool = True):
        model = self.nuke_pixel_grads(model)
        ramp = self.model_interferogram(model)
        if return_slopes:
            return ramp.set("data", np.diff(ramp.data, axis=0))
        return ramp


def ff_reg(model, exposure, ff_std=0.035):
    ff_norm = model.FF - 1
    return np.mean((ff_norm / ff_std) ** 2)


def nl_reg(model, exposure, nl_std=0.025):
    nl_norm = model.non_linearity - model.non_linearity.mean()
    return np.mean((nl_norm / nl_std) ** 2)


def loss_fn(model, exposure, args={}):

    # calculating likelihood
    likelihood = -np.nanmean(exposure.mv_zscore(model))

    # applying regularisers to calculate prior
    prior = apply_regularisers(model, exposure, args)

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


def temp_decay(T0, k, t):
    return T0 * np.exp(-k * t)


def get_warmup(args):
    return cosine_warmup(args["t"], args["t0"], args["n_max"])


def get_temperature(args):
    return temp_decay(args["T0"], args["k"], args["t"])


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

    # Add the learning rate warm-up (we also warm up the temperature here)
    values *= args["max_lr"] * get_warmup(args)

    rand_vals = get_temperature(args) * jr.normal(key, values.shape)
    values += rand_vals

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

    print_str = ""
    if len(cal_losses) > 0:
        print_str += "Cal: "

        losses = np.array(list(cal_losses.values())).mean(0)
        print_str += f"{losses[-1]:.2f}"
        if len(losses) > 1:
            print_str += f" \u0394 {np.diff(losses)[-1]:.2f}"

    if len(val_losses) > 0:
        print_str += " | Val: "

        losses = np.array(list(val_losses.values())).mean(0)
        print_str += f"{losses[-1]:.2f}"
        if len(losses) > 1:
            print_str += f" \u0394 {np.diff(losses)[-1]:.2f}"

    if len(flat_losses) > 0:
        print_str += " | Flat: "
        losses = np.array(list(flat_losses.values())).mean(0)
        print_str += f"{losses[-1]:.2f}"
        if len(losses) > 1:
            print_str += f" \u0394 {np.diff(losses)[-1]:.2f}"

    # NOTE "l2_reg" is just all priors
    # TODO Edit Trainer class so this is not hardcoded
    prior = np.array(jtu.leaves(aux_dict["l2_reg"]))
    if len(prior) > 0:
        print_str += f" | Prior: {prior[-1]:.2f}"
        if len(prior) > 1:
            print_str += f" \u0394 {np.diff(prior)[-1]:.2f}"

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
    


def summarise_fn(
    result,
    save_path,
    val_flag=False,
    binary_flag=False,
    save_flag=False,
    flat_flag=False,
    flat_only=False,
    static_optics=False,
    cal_exposures=[],
    val_exposures=[],
    flat_exposures=[],
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
    
    if save_flag:
        try:
            # --- best_state.npy ---
            # model_params from the epoch with best validation loss,
            # with nn_weights replaced by the mean across all batches
            # of that epoch (since nn_weights is a batched parameter,
            # best_state only contains its value at the final batch)
            best_params = result.best_state.params
            best_params["nn_weights"] = np.array(
                result.best_batch["nn_weights"]
            ).mean(0)
            np.save(
                os.path.join(save_path, "best_state.npy"),
                best_params,
                allow_pickle=True,
            )
    
        except Exception as e:
            print(f"Saving best state failed: {e}")
        
        # --- final_state.npy ---
        # model_params from the final epoch of training,
        # independent of validation loss
        final_params = {key: result.model.get(key) for key in params_to_save}
        final_params["nn_weights"] = np.array(
            result.history["nn_weights"]  # all batches of final epoch
        )[-n_batch:].mean(0)
        np.save(
            os.path.join(save_path, "final_state.npy"),
            final_params,
            allow_pickle=True,
        )
    else:
        try:
            # --- best_state.npy ---
            # model_params from the epoch with best validation loss,
            # with nn_weights replaced by the mean across all batches
            # of that epoch (since nn_weights is a batched parameter,
            # best_state only contains its value at the final batch)
            best_params = result.best_state.params
            best_params["nn_weights"] = np.array(
                result.best_batch["nn_weights"]
            ).mean(0)
            np.save(
                os.path.join(amigo_files_path, "scratch_best_state.npy"),
                best_params,
                allow_pickle=True,
            )
    
        except Exception as e:
            print(f"Saving best state failed: {e}")
        
        # --- final_state.npy ---
        # model_params from the final epoch of training,
        # independent of validation loss
        final_params = {key: result.model.get(key) for key in params_to_save}
        final_params["nn_weights"] = np.array(
            result.history["nn_weights"]  # all batches of final epoch
        )[-n_batch:].mean(0)
        np.save(
            os.path.join(amigo_files_path, "scratch_final_state.npy"),
            final_params,
            allow_pickle=True,
        )
            
    

    
    ################## PLOTTING LOSSES ###################

    start, stop = 50, -1
    
    if val_flag:
        #
        # epochs = cal.shape[-1]
        # start, stop = 500, -1
        
        if start >= cal.shape[-1]:
            start = 1
        
        if stop < 0:
            stop = cal.shape[-1] + stop
        xs = np.arange(start, stop)
        
        # print(np.array(cal).mean(0)[xs].shape)
        
        plt.figure(figsize=(18, 4))
        ax = plt.subplot(1, 3, 1)
        plt.plot(xs, np.array(cal).mean(0)[xs])
        ax.set(title="Calibrators", xlabel="Epochs", ylabel="Loss")
        
        ax = plt.subplot(1, 3, 2)
        ax.set(title="Validators", xlabel="Epochs", ylabel="Loss")
        if val_flag:
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
        if val_flag:
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
        amigo.plotting.plot_losses(list(result.losses.values())[0], start=start, save_path=save_path)
        

    ################### PLOTTING HISTORY AND SUMMARISE FIT ###################
    amigo.plotting.plot(result.history, save_path=save_path)
    
    exposures_lists = [cal_exposures]
    exp_types = ["cal"]
    if val_flag:
        exp_types += ["val"]
        exposures_lists += [val_exposures]
    if flat_flag:
        exp_types += ["flat"]
        exposures_lists += [flat_exposures]
        
    for exp_type, exps in zip(exp_types, exposures_lists):
        print(5*"\n")
        print(exp_type)
        if flat_only and exp_type != "flat":
            continue
        if save_path is not None:
            this_save_path = os.path.join(save_path, exp_type)
            os.mkdir(this_save_path)
        else:
            this_save_path = None
        for exp in exps:
            exp.print_summary()
            amigo.plotting.summarise_fit(result.model, exp, save_path=this_save_path)

    ################### PIXEL SENSITIVITY AND NON-LINEARITY ###################

    print(badpix)
    
    badpix_bool = badpix.astype(bool)
    
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
        plt.savefig(os.path.join(save_path, "pixels.png"), dpi=300)
    plt.show()


    ################### WAVEFRONT ###################
    try:
    
        optics = result.model.optics
        pupil_mask = result.model.optics.pupil_mask
        
        rms = lambda x: np.sqrt(np.nanmean(np.square(x)))
        
        if "aberrations" in optimisers.keys():
            for prog in ["4481", "8330", "1093", "1843", "1242"]:
                print(prog)
                cal_aberrations = {key: val for key, val in result.state.aberrations.items() if prog in key}  # NOTE USE ALL 
                
                fig, axes = plt.subplots(2, 3, figsize=(12, 6))
                for i, key in enumerate(cal_aberrations):
                
                    coeffs = result.model.aberrations[key]
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
                
                    ax_top = axes[0, i]
                    ax_bot = axes[1, i]
                
                    v = np.nanmax(np.abs(full_abb))
                    ax_top.set_title(f"{key} — Full OPD (RMS: {rms(full_abb):.2f} nm)")
                    im_top = ax_top.imshow(full_abb, cmap=seismic, vmin=-v, vmax=v)
                    fig.colorbar(im_top, ax=ax_top, label="OPD (nm)")
                
                    v = np.nanmax(np.abs(flat_abb))
                    ax_bot.set_title(f"{key} — FLAT OPD (RMS: {rms(flat_abb):.2f} nm)")
                    im_bot = ax_bot.imshow(flat_abb, cmap=seismic, vmin=-v, vmax=v)
                    fig.colorbar(im_bot, ax=ax_bot, label="OPD (nm)")
                
                fig.tight_layout()
                if save_flag:
                    plt.savefig(os.path.join(save_path, f"wavefront_{prog}.png"), dpi=300)
                    plt.close()
                else:
                    plt.show()
                    
    except Exception as e:
        print(f"Plotting wavefront failed: {e}")
        

    
    ################### PUPIL AND BEAM DISTORTIONS ###################

    if not static_optics:
        null_pupil_mask = result.model.optics.pupil_mask.multiply("primary_beam", 0.0).multiply("distortion", 0.0)
        null_mask = null_pupil_mask.calc_mask(optics.wf_npixels, optics.diameter)
        
        pupil_mask = result.model.optics.pupil_mask
        mask = pupil_mask.calc_mask(optics.wf_npixels, optics.diameter)
        
    else:
        raw_optics = AMIOptics(static=False)
        null_pupil_mask = raw_optics.pupil_mask.multiply("primary_beam", 0.0).multiply("distortion", 0.0)
        null_mask = null_pupil_mask.calc_mask(optics.wf_npixels, optics.diameter)

        mask = result.model.optics.transmission
        
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
            n = 2
            k = 2 * n + 1
        
            for exp in dic[filt]:
    
                print(exp)
        
                slopes = exp(result.model)
                im = slopes.sum(0)
                peak_pix = im == np.nanmax(im)
                peak_map = convert_adjacent_to_true(peak_pix, corners=True, n=n)
        
                ######### plotting #########
                max_loc = np.argwhere(peak_pix)[0][::-1]
                square = mpl.patches.Rectangle(max_loc - np.array([2.5, 2.5]), 5, 5, color="r", fill=False)
        
                fig, ax = plt.subplots(figsize=(3, 2))
                c = ax.imshow(im, "cividis", norm=mpl.colors.PowerNorm(0.5))
                ax.add_patch(square)
                ax.set(title=f"Cropping: {filt}")
                ax.axis("off")
                fig.colorbar(c, ax=ax, label="Counts")
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
