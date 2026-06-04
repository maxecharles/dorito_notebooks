import numpy as onp
from jax import numpy as np, random as jr, tree as jtu
from dorito.stats import apply_regularisers
import dLux as dl
from dLux import utils as dlu
from amigo.model_fits import ModelFit, PointFit
import pandas as pd


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