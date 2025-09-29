# %%
# jax ecosystem
import jax
from jax import numpy as np, tree as jtu

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
print(jax.local_devices()[0].device_kind)

import zodiax as zdx
from zodiax.optimisation import sgd, adam
import amigo
import dorito

# other helpful libraries
import os
import sys

# matplotlib ecosystem
import matplotlib.pyplot as plt
import matplotlib as mpl
import ehtplot
import scienceplots

# matplotlib parameters
plt.style.use(["science", "bright", "no-latex"])

plt.rcParams["image.cmap"] = "inferno"
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = "lower"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 8
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"

# %%
from socket import gethostname

if gethostname() == "glinton":
    path = "/media/morgana1/snert/max/"
elif gethostname() == "AJQ4YHQH9TX":
    path = "/Volumes/morgana1/snert/max/"
else:
    path = "/fred/oz440/max/"

source_name = "IO"
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
]


# Bind file path, type and exposure type
file_fn = lambda data_path, filters=FILTERS, **kwargs: amigo.files.get_files(
    data_path,
    "calslope",
    EXP_TYPE=EXP_TYPE,
    FILTER=filters,
    **kwargs,
)

from datetime import datetime

form = "%d-%m-%y_%H-%M-%S.%f"
now = datetime.now()
datetime_str = now.strftime(form)

# clear directory of empty folders
for folder in os.listdir(output_path):
    folder_dir = os.path.join(output_path, folder)

    # skip if not a directory
    if not os.path.isdir(folder_dir):
        continue

    # if the folder is empty
    if len(os.listdir(folder_dir)) == 0:

        try:
            then = datetime.strptime(folder, form)

            # remove empty folder if it is older than 1 hour
            if (now - then).seconds > 3600:  # 1 hour
                print(f"Removing empty folder: {folder_dir}")
                os.rmdir(folder_dir)
        except ValueError:
            # if the folder name is not in the correct format, skip it
            print(f"Deleting folder: {folder_dir} (not in correct format)")
            os.rmdir(folder_dir)

# datetime_str = f"{i}_groups"
print(datetime_str)

batch_idx = sys.argv[1] if len(sys.argv) > 1 else "0"

job_id = os.environ.get("SLURM_ARRAY_JOB_ID")

output_path = os.path.join(output_path, job_id) + f"/{batch_idx}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)
print(f"Output path: {output_path}")

# %% [markdown]
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

    file["BADPIX"].data[:, :10] = 1
    file["BADPIX"].data[:, -10:] = 1
    file["BADPIX"].data[:10, :] = 1
    file["BADPIX"].data[-10:, :] = 1

    if not bool(file[0].header["IS_PSF"]):
        file["BADPIX"].data[43, 45] = 1
        file["BADPIX"].data[40, 45] = 1
        sci_files.append(file)
    elif bool(file[0].header["IS_PSF"]):
        file[0].header["TARGPROP"] = "HD 2236"
        cal_files.append(file)
    else:
        print(f"Unkown target: {file[0].header['TARGPROP']}")

dorito.misc.truncate_files(sci_files, 18)

# %%
from astropy.time import Time

t0 = Time(0, format="mjd")
for file in files:
    h = file[0].header
    t = Time(h["EXPMID"], format="mjd")

    if t.ymdhms[2] != t0.ymdhms[2]:
        print("\n" + 30 * "-" + "\n")

    print(
        f'{h["TARGPROP"]} {h["FILTER"]}, Dither {h["PATT_NUM"]}/{h["NUMDTHPT"]}, Roll {h["ROLL_REF"]:.1f}deg, {h["XPOSURE"] / 60:.1f}min, {t.iso}, Groups: {h["NGROUPS"]}, Ints: {h["NINTS"]}'
    )
    t0 = t

# %% [markdown]
# ## Building the model


# %%
class DynamicResolvedFit(dorito.model_fits.ResolvedFit):
    """
    Model fit for resolved sources where each exposure has a different
    intensity distribution.
    """

    def get_key(self, param):
        match param:
            case "log_dist":
                return "_".join([self.key, self.filter])

        return super().get_key(param)


# %%
source_size = 131  # pixels
load_dict = lambda x: np.load(f"{x}", allow_pickle=True).item()

# sci_fits = [dorito.model_fits.ResolvedFit(file, use_cov=True) for file in sci_files]
sci_fits = [DynamicResolvedFit(file, use_cov=True) for file in sci_files]
cal_fits = [amigo.model_fits.PointFit(file, use_cov=False) for file in cal_files]

# I only want to use the calibrator in the same primary dither position
fits = sci_fits + cal_fits[0:1]

# building the model
model = dorito.models.ResolvedAmigoModel(
    exposures=fits,
    optics=amigo.optical_models.AMIOptics(),
    detector=amigo.detector_models.LinearDetector(),
    ramp_model=amigo.ramp_models.NonLinearRamp(),
    read=amigo.read_models.ReadModel(),
    state=load_dict(cache + "calibration.npy"),
    param_initers={
        "distribution": np.ones((source_size, source_size)) / source_size**2
    },
)

# %%
# for exp in fits:
#     exp.print_summary()
#     amigo.plotting.summarise_fit(model, exp, residuals=False, save_path=output_path)

import shutil

shutil.copy(__file__, output_path + "/script.py")

# %% [markdown]
# ## Optimisation Stage 1: Gradient Descent

# %%
pos_keys = []
spc_keys = []
for exp in fits:
    if not exp.calibrator:
        pos_keys.append(exp.map_param("positions"))
        spc_keys.append(exp.map_param("spectra"))

from jax_gaussian import gaussian_filter
# sigs = np.concat((np.array([1e-16,]), np.linspace(0.05, 0.7, 14)))
# sig = float(sigs[int(batch_idx)])
sig = 0.25
def norm_fn(model_params, args):
    params = model_params.params
    if "log_dist" in params.keys():
        for k, log_dist in params["log_dist"].items():
            distribution = 10**log_dist
            distribution = gaussian_filter(distribution, sigma=sig)
            params["log_dist"][k] = np.log10(distribution / distribution.sum())

    if "spectra" in params.keys():
        spectra = jtu.map(
            lambda x: np.clip(x, a_min=-0.8, a_max=0.8), params["spectra"]
        )
        params["spectra"] = spectra

    return model_params.set("params", params), args


pscale = lambda model: model.optics.psf_pixel_scale / model.optics.oversample

# %%
n_epoch = 15000

config = {
    "positions": sgd(4e-2, 0, (500, 0.)),
    "fluxes": sgd(5e-2, 0),
    # "aberrations": sgd(5e0, 4),
    "spectra": sgd(1e-1, 10),
    "log_dist": adam(1e-2, 20, (11000, 0.2)),  # , (10, 0.25), b1=0.7),
}


def grad_fn(model, grads, args):
    # Nuke the position gradients for the science exposures
    if "positions" in config.keys():
        grads = grads.multiply(pos_keys, 0.5)

    # Reduce spectra gradients for the science exposures
    if "spectra" in config.keys():
        grads = grads.multiply(spc_keys, 0.3)
    return grads, args

# tsvs = [1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3]
# tvs = [0., 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 8e-1, 1e0, 2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 5e2]
tvs = np.concat((np.array([0]), np.logspace(-2, 3, 29)))

# mes = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7]
args = {
    "reg_dict": {
        "TV": (tvs[int(batch_idx)], dorito.stats.TV),
        # "TSV": (tsvs[int(batch_idx)], dorito.stats.TSV),
        # "ME": (mes[int(batch_idx)], dorito.stats.ME),
    }
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
    fits,
    hessians=load_dict(cache + "jacobians.npy")["hessian"],
    parameters=[p for p in config.keys()],  # if p not in ["log_dist"]],
)

print("Number of exposures: ", len(fits))

# Train the model
result = trainer.train(
    model=model,
    optimisers=config,
    epochs=n_epoch,
    batches=fits,
    args=args,
)


# %%
np.save(output_path + "params.npy", result.model.params, allow_pickle=True)
result_model = result.model

balance_dict = dorito.stats.ramp_posterior_balances(result_model, sci_fits, args)
np.save(output_path + "balance.npy", balance_dict, allow_pickle=True)

# np.save(output_path + "history.npy", result.history.params, allow_pickle=True)

# %%
from dLux import utils as dlu

optics_diameter = 6.603464  # JWST aperture diameter in meters


def eff_wavel(model, filt):
    wavels, weights = model.filters[filt]
    return np.dot(wavels, weights)


for exp in fits:

    if exp.calibrator:
        continue
    dist = result_model.get_distribution(exp, rotate=False)
    fig, ax = plt.subplots(figsize=(6, 2.3))

    c0 = dorito.plotting.plot_result(
        ax,
        dist / dist.max(),
        pixel_scale=model.psf_pixel_scale / model.oversample,
        # cmap="inferno",
        # roll_angle_degrees=-exp.parang,
        # norm=mpl.colors.LogNorm(vmin=1e-5),
        # norm=mpl.colors.PowerNorm(0.3, vmax=0.3),
        diff_lim=dlu.rad2arcsec(eff_wavel(model, exp.filter) / optics_diameter),
        scale=1,
    )

    fig.colorbar(c0)

    ax.set(title=f"Io - {exp.filter}")  #   xticks=ticks, yticks=ticks)

    plt.tight_layout()
    # plt.show()
    plt.savefig(output_path + f"{exp.key}_dist.png", dpi=300)
    plt.close()

for exp in fits:
    exp.print_summary()
    amigo.plotting.summarise_fit(result.model, exp, save_path=output_path)

amigo.plotting.plot_losses(
    result.losses[0], start=int(n_epoch * 0.75), save_path=output_path
)
amigo.plotting.plot(result.history, save_path=output_path)
