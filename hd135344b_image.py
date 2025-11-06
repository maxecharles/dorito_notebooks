# %%
# jax ecosystem
import jax

jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", True)
print(jax.local_devices()[0].device_kind)

from jax import numpy as np, tree as jtu
from zodiax.optimisation import sgd, adam
import amigo
import dorito

# other helpful libraries
import os

# matplotlib ecosystem
import matplotlib.pyplot as plt
import matplotlib as mpl

# matplotlib parameters
# plt.style.use(["science", "bright", "no-latex"])

plt.rcParams["image.cmap"] = "inferno"
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = "lower"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 8
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"

inferno = mpl.colormaps["inferno"]
viridis = mpl.colormaps["viridis"]
seismic = mpl.colormaps["seismic"]
coolwarm = mpl.colormaps["coolwarm"]

inferno.set_bad("k", 0.5)
viridis.set_bad("k", 0.5)
seismic.set_bad("k", 0.5)
coolwarm.set_bad("k", 0.5)

# %%
from socket import gethostname

if gethostname() == "glinton":
    path = "/media/morgana1/snert/max/"
elif gethostname() == "AJQ4YHQH9TX":
    path = "/Volumes/morgana1/snert/max/"
else:
    path = "/fred/oz440/max/"

source_name = "HD135344B"
data_path = os.path.join(path, f"data/JWST/{source_name}/calslope/")
uncal_path = os.path.join(path, f"data/JWST/{source_name}/uncal/")
amigo_cache = os.path.join(path, "data/amigo_files/")

cache = os.path.join(amigo_cache, "v_0.0.10/")
output_path = os.path.join(amigo_cache, f"outputs/{source_name}/")

EXP_TYPE = "NIS_AMI"
FILTERS = [
    "F480M",
    "F430M",
    "F380M",
    "F277W",
]

# Bind file path, type and exposure type
file_fn = lambda data_path, filters=FILTERS, **kwargs: amigo.files.get_files(
    data_path,
    "calslope",
    EXP_TYPE=EXP_TYPE,
    FILTER=filters,
    **kwargs,
)

# %% [markdown]
#
# # Loading in data

# %%
files = sorted(
    file_fn(data_path), key=lambda hdu: hdu[0].header.get("EXPMID", float("inf"))
)
sci_files = []
cal_files = []

for file in files:

    # manual bad pixel correction
    file["BADPIX"].data[58, 67] = 1
    file["BADPIX"].data[71, 22] = 1
    file["BADPIX"].data[65, 41] = 1
    file["BADPIX"].data[35, 70] = 1
    file["BADPIX"].data[70, 55] = 1
    file["BADPIX"].data[5, 5] = 1
    file["BADPIX"].data[-4, 37] = 1
    file["BADPIX"].data[51, 27] = 1
    file["BADPIX"].data[28, 18] = 1
    file["BADPIX"].data[32, 10] = 1

    file["BADPIX"].data[:, :3] = 1
    file["BADPIX"].data[:, -3:] = 1
    file["BADPIX"].data[:3, :] = 1
    file["BADPIX"].data[-3:, :] = 1

    file["BADPIX"].data[20:23, 4] = 1
    file["BADPIX"].data[12:15, -5] = 1
    file["BADPIX"].data[16:19, 31] = 1
    file["BADPIX"].data[60, 61] = 1

    # if file[0].header["TARGPROP"] == "TD-PDS-70":
    #     # file["BADPIX"].data[36:66, :25] = 1  # BACKGROUND STAR?
    #     file["BADPIX"].data[17, 70] = 1
    #     file["BADPIX"].data[19, 53] = 1
    #     file["BADPIX"].data[5, 22] = 1
    #     file["BADPIX"].data[19, 41] = 1
    #     # file["BADPIX"].data[:66, :25] = 1  # BACKGROUND STAR?
    #     file["BADPIX"].data[76, 45] = 1
    #     file["BADPIX"].data[59, 30] = 1

    if not bool(file[0].header["IS_PSF"]):
        sci_files.append(file)
    elif bool(file[0].header["IS_PSF"]):
        # file[0].header["TARGPROP"] = "HD 228337"
        cal_files.append(file)
    else:
        print(f"Unkown target: {file[0].header['TARGPROP']}")

# dorito.misc.truncate_files(sci_files, 30)

# %%
from astropy.time import Time

t0 = Time(0, format="mjd")
for file in files:
    h = file[0].header
    t = Time(h["EXPMID"], format="mjd")

    if t.ymdhms[2] != t0.ymdhms[2]:
        print("\n" + 30 * "-" + "\n")

    print(
        f'{h["TARGPROP"]} {h["FILTER"]}, Dither {h["PATT_NUM"]}/{h["NUMDTHPT"]}, Roll {h["ROLL_REF"]:.1f}deg, {h["XPOSURE"] / 60:.1f}min, {t.iso}, Groups: {h["NGROUPS"]}, Ints: {h["NINTS"]}, PWCPOS {h["PWCPOS"]}'
    )
    t0 = t

# %% [markdown]
# ## Building the model

# %%
from dorito.model_fits import PointResolvedFit


class PRRotFit(PointResolvedFit):

    def get_key(self, param):

        match param:
            case "rotation":
                return self.filter

        return super().get_key(param)

    def map_param(self, param):

        # Map the appropriate parameter to the correct key
        if param in ["rotation"]:
            return f"{param}.{self.get_key(param)}"

        # Else its global
        return super().map_param(param)

    def initialise_params(self, optics, coeffs, contrast):
        """
        Initialise the parameters for the resolved source model fit.
        The log distribution is set to a uniform distribution specified by the source size.

        Args:
            optics: The optics object (to pass to the parent class).
            source_size: The size of the source distribution (assumed square).

        Returns:
            params: A dictionary containing the initialised parameters for the model fit.
        """

        params = super().initialise_params(optics, coeffs, contrast)

        params["rotation"] = self.get_key("rotation"), np.array(0.0)

        return params

    def model_interferogram(
        self,
        psf,
        model,
        rotate: bool = None,
        source_id: str = None,
    ):

        psf = super().model_interferogram(
            psf,
            model,
            rotate=rotate,
            source_id=source_id,
        )
        rotation = model.get(self.map_param("rotation"))
        return psf.rotate(rotation)


# %%
load_dict = lambda x: np.load(f"{x}", allow_pickle=True).item()  # helper function

# just two science exposures and one calibrator for this demo
sci_exps = [PRRotFit(file) for file in sci_files]
# sci_exps = [dorito.model_fits.PointResolvedFit(file) for file in sci_files]
cal_exps = [amigo.model_fits.PointFit(file) for file in cal_files]
exps = sci_exps + cal_exps
# exps = cal_exps
# exps = sci_exps

# building the model
source_size = 101  # pixels
basis, window = dorito.bases.inscribed_annulus_basis(source_size, iterations=1)
init_dist = np.ones((source_size, source_size)) / window.sum()

model = dorito.models.TransformedResolvedModel(
    exposures=exps,
    optics=amigo.optical_models.AMIOptics(),
    detector=amigo.detector_models.LinearDetector(),
    ramp_model=amigo.ramp_models.NonLinearRamp(),
    read=amigo.read_models.ReadModel(),
    state=load_dict(cache + "calibration.npy"),
    basis=basis,
    window=window,
    param_initers={"contrast": 0.04, "distribution": init_dist},
)

# model.params["aberrations"] = load_dict("files/HD135344B_CALFIT.npy")["aberrations"]

# %%
import sys
import shutil

shutil.copy(__file__, output_path + "/script.py")

job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
job_name = os.environ.get("SLURM_JOB_NAME")
job_idx = "_".join(job_id, job_name)

batch_idx = int(sys.argv[1]) if len(sys.argv) > 1 else int(0)
output_path = os.path.join(output_path, job_idx) + f"/{batch_idx}/"

if not os.path.exists(output_path):
    os.makedirs(output_path)
print(f"Output path: {output_path}")

for exp in exps:
    exp.print_summary()
    amigo.plotting.summarise_fit(model, exp, residuals=False, save_path=output_path)


# %% [markdown]
# ## Optimisation Stage 1: Gradient Descent

# %%
pos_keys = []
spc_keys = []
flx_keys = []
for exp in exps:
    if not exp.calibrator:
        spc_keys.append(exp.map_param("spectra"))
        pos_keys.append(exp.map_param("positions"))
        flx_keys.append(exp.map_param("fluxes"))


def norm_fn(model_params, args):
    params = model_params.params

    # NOTE: This normalisation won't work for an arbitrary basis!
    if "log_dist" in params.keys():
        for k, log_dist in params["log_dist"].items():
            distribution = 10**log_dist
            log_dist = np.log10(distribution / distribution.sum())

            # basis = args["basis"]
            # window = args["window"]
            # distribution = 10 ** basis.from_basis(log_dist) * window
            # distribution = gaussian_filter(distribution, sigma=sig)
            # distribution += 1e-16 * ((window + 1) % 2)
            # log_dist = np.log10(distribution / distribution.sum())
            # log_dist = basis.to_basis(log_dist)
            params["log_dist"][k] = log_dist

    if "spectra" in params.keys():
        spectra = jax.tree.map(
            lambda x: np.clip(x, a_min=-0.8, a_max=0.8), params["spectra"]
        )
        params["spectra"] = spectra

    return model_params.set("params", params), args


pscale = lambda model: model.optics.psf_pixel_scale / model.optics.oversample


# %%
def logTV(model, exposure, source_id=None):
    return dorito.stats.TV_loss(
        model.get_distribution(exposure, source_id=source_id, exponentiate=False)
    )


# %%
n_epoch = 100

config = {
    # "positions": sgd(3e-1, 0),
    # "fluxes": sgd(2e-1, 0),
    # "aberrations": sgd(2e-2, 4),
    # "spectra": sgd(4e-1, 8),
    # "log_dist": adam(5e-2, 20),
    # "rotation": sgd(1e-9, 100),
    # "log_dist": adam(5e-1, 30),
    # "contrast": adam(2e-2, 15),
    # "log_dist": adam(5e-2, 0),
    # ALTOGETHER
    "positions": sgd(1e-1, 0),
    "fluxes": sgd(5e-2, 0),
    "aberrations": sgd(8e-1, 4),
    "spectra": sgd(3e-1, 8),
    "log_dist": adam(5e-2, 20),
    "rotation": sgd(1e-8, 50),
}


def grad_fn(model, grads, args):

    # Reduce spectra gradients for the science exposures
    # if "spectra" in config.keys():
    #     grads = grads.multiply(spc_keys, 0.3)

    if "fluxes" in config.keys():
        grads = grads.multiply(flx_keys, 5)
    if "positions" in config.keys():
        grads = grads.multiply(pos_keys, 5)
    return grads, args


args = {
    "reg_dict": {
        # "ME": (5e0, dorito.stats.ME),
        # "TV": (1e-2, dorito.stats.TV),
        "TV": (1e3, logTV),
        # "TV": (1e0, dorito.stats.TV),
    },
    "source_id": "TD",
    "basis": basis,
    "window": window,
}

trainer = amigo.fitting.Trainer(
    loss_fn=dorito.stats.ramp_regularised_loss_fn,
    norm_fn=norm_fn,
    grad_fn=grad_fn,
    cache=os.path.join(amigo_cache, "fishers/"),
)

print("Populating fishers...")
trainer = trainer.populate_fishers(
    model,
    exps,
    hessians=load_dict(cache + "jacobians.npy")["hessian"],
    parameters=[
        p for p in config.keys() if p not in ["log_dist", "contrast", "rotation"]
    ],
)

print("Number of exposures: ", len(exps))

# Train the model
result = trainer.train(
    model=model,
    optimisers=config,
    epochs=n_epoch,
    batches=exps,
    args=args,
)

# %%
np.save(output_path + "params.npy", result.model.params, allow_pickle=True)
result_model = result.model

# balance_dict = dorito.stats.ramp_posterior_balances(result_model, sci_fits, args)
# np.save(output_path + "balance.npy", balance_dict, allow_pickle=True)

amigo.plotting.plot_losses(
    result.losses[0], start=int(n_epoch * 0.75), save_path=output_path
)
amigo.plotting.plot(result.history, save_path=output_path)

for exp in exps:
    exp.print_summary()
    amigo.plotting.summarise_fit(result.model, exp, save_path=output_path)


# %%
from dLux import utils as dlu

# np.save("files/pds70_abs.npy", result.model.params["aberrations"], allow_pickle=True)


def eff_wavel(model, filt):
    wavels, weights = model.filters[filt]
    return np.dot(wavels, weights)


result_model = result.model
optics_diameter = 6.603464  # JWST aperture diameter in meters

for exp in exps:

    if exp.calibrator:
        continue
    dist = result_model.get_distribution(exp, rotate=False)
    fig, ax = plt.subplots(figsize=(6, 3))

    c0 = dorito.plotting.plot_result(
        ax,
        dist,
        pixel_scale=model.psf_pixel_scale / model.oversample,
        cmap="inferno",
        # roll_angle_degrees=-exp.parang,
        norm=mpl.colors.PowerNorm(0.5, vmin=None),
        # norm=mpl.colors.PowerNorm(0.3, vmax=0.3),
        diff_lim=0.5 * dlu.rad2arcsec(eff_wavel(model, exp.filter) / optics_diameter),
        # scale=1.5,
    )

    fig.colorbar(c0)

    ax.set(title=f"HD135344B - {exp.filter}")  #   xticks=ticks, yticks=ticks)
    ax.scatter([0], [0], marker="*", color="white", s=10)

    plt.tight_layout()
    # plt.show()
    plt.savefig(output_path + f"{exp.key}_dist.png", dpi=300)
