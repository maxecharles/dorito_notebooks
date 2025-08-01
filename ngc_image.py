# %%
# jax ecosystem
import jax
from jax import numpy as np, tree as jtu
import zodiax as zdx
from zodiax.optimisation import sgd, adam
import amigo
import dorito

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
print("Default Backend:", jax.default_backend())
print(jax.local_devices()[0].device_kind)

# other helpful libraries
import numpy
import os
import sys
import astropy

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

source_name = "NGC1068"
data_path = os.path.join(path, f"data/JWST/{source_name}/calslope/")
uncal_path = os.path.join(path, f"data/JWST/{source_name}/uncal/")
amigo_cache = os.path.join(path, "data/amigo_files/")

cache = os.path.join(amigo_cache, "v_0.0.10/")
output_path = os.path.join(amigo_cache, f"outputs/{source_name}/")

EXP_TYPE = "NIS_AMI"
FILTERS = [
    "F480M",
    # "F430M",
    # "F380M",
    # "F277W",
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
    # file["BADPIX"].data[36:66, :25] = 1  # BACKGROUND STAR?
    # file["BADPIX"].data[40, 45] = 1  # MIDDLE PIXELS

    if not bool(file[0].header["IS_PSF"]):
        # badpix = np.array(file["BADPIX"].data, dtype=bool)
        # im = np.array(file["SLOPE"].data.sum(0))
        # im = np.where(badpix, np.nan, im)
        # mask = binary_dilation(im == np.nanmax(im), iterations=2)
        # file["BADPIX"].data += mask.astype(int)
        sci_files.append(file)
    elif bool(file[0].header["IS_PSF"]):
        file[0].header["TARGPROP"] = "HD 228337"
        cal_files.append(file)
    else:
        print(f"Unkown target: {file[0].header['TARGPROP']}")

dorito.misc.truncate_files(sci_files, 4)

# %%
from datetime import datetime

now = datetime.now().replace(second=0, microsecond=0)
datetime_str = now.strftime("%d-%m-%y_%H-%M")

# datetime_str = f"{i}_groups"
print(datetime_str)

output_path = os.path.join(output_path, datetime_str) + "/"
if not os.path.exists(output_path):
    os.makedirs(output_path)
print(f"Output path: {output_path}")

# Saving a copy of this script
import shutil
shutil.copy(__file__, os.path.join(output_path, "script.py"))


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
source_size = 121  # pixels
load_dict = lambda x: np.load(f"{x}", allow_pickle=True).item()

sci_fits = [
    dorito.model_fits.ResolvedFit(file, source_size, use_cov=True) for file in sci_files
]
cal_fits = [amigo.model_fits.PointFit(file, use_cov=True) for file in cal_files]

# I only want to use the calibrator in the same primary dither position
# fits = cal_fits[0:1]
fits = sci_fits[0:1] + cal_fits[0:1]
# fits = sci_fits + cal_fits

# building the model
model = dorito.models.ResolvedAmigoModel(
    # model = amigo.core_models.AmigoModel(
    exposures=fits,
    optics=amigo.optical_models.AMIOptics(),
    detector=amigo.detector_models.LinearDetector(),
    ramp_model=amigo.ramp_models.NonLinearRamp(),
    read=amigo.read_models.ReadModel(),
    state=load_dict(cache + "calibration.npy"),
)

# %% [markdown]
# ## Optimisation Stage 1: Gradient Descent

# %%
pos_keys = []
spc_keys = []
for exp in fits:
    if not exp.calibrator:
        pos_keys.append(exp.map_param("positions"))
        spc_keys.append(exp.map_param("spectra"))


def norm_fn(model_params, args):
    params = model_params.params
    # if "log_distribution" in params.keys():
    #     for k, log_dist in params["log_distribution"].items():
    #         distribution = 10**log_dist
    #         params["log_distribution"][k] = np.log10(distribution / distribution.sum())

    if "spectra" in params.keys():
        spectra = jtu.map(
            lambda x: np.clip(x, a_min=-0.8, a_max=0.8), params["spectra"]
        )
        params["spectra"] = spectra

    return model_params.set("params", params), args


pscale = lambda model: model.optics.psf_pixel_scale / model.optics.oversample

# %%
n_epoch = 100

config = {
    "positions": sgd(5e-2, 0),
    "fluxes": sgd(5e-2, 5),
    "aberrations": sgd(2e-1, 10),
    "log_dist": adam(1e-1, 10, (1000, 0.75)),
    "spectra": sgd(2e-1, 50),
    # "amplitudes": sgd(5e-1, 50),
    # "phases": sgd(5e-1, 50),
}


def grad_fn(model, grads, args):
    # Nuke the position gradients for the science exposures
    if "positions" in config.keys():
        grads = grads.multiply(pos_keys, 0.1)

    # Reduce spectra gradients for the science exposures
    if "spectra" in config.keys():
        grads = grads.multiply(spc_keys, 0.3)
    return grads, args


reg_dict = {
    # "L1": 2.0e3,
    # "L1": 5.0e-4,
    # "L2": None,
    # "TV": 1.0e6,
    # "TV": 1.0e6,
    # "QV": 1.0e6,
    # "QV": 1.0e-3,
    # "ME": 1.0e2,
    # "SF": 1e3,
}

args = {
    "reg_dict": reg_dict,
    # # "reg_func_dict": dorito.stats.reg_func_dict,
    "reg_func_dict": {
        # "L1": dorito.stats.L1_on_wavelets,
        # "L1": L1_REG,
        # "QV": dorito.stats.TSV,
        "TV": dorito.stats.TV,
        # "ME": dorito.stats.ME,
    },
    # "mask": mask,
}

trainer = amigo.fitting.Trainer(
    # loss_fn=regularised_loss_fn,
    norm_fn=norm_fn,
    grad_fn=grad_fn,
    cache=os.path.join(amigo_cache, "fishers/"),
)

print("Populating fishers...")
trainer = trainer.populate_fishers(
    # model.set("detector.ramp.bleed", False).set("params", params),
    model,
    fits,
    hessians=load_dict(cache + "jacobians.npy")["hessian"],
    parameters=[p for p in config.keys()],  # if p not in ["log_distribution"]],
)

print("Number of exposures: ", len(fits))

# Train the model
result = trainer.train(
    model=model,
    optimisers=config,
    epochs=n_epoch,
    batches=fits,
)

# %%
from dLux import utils as dlu

np.save(output_path + "final_params.npy", result.model.params, allow_pickle=True)

optics_diameter = 6.603464  # JWST aperture diameter in meters

wavel = 4.8e-6

for exp in fits:

    if exp.calibrator:
        continue
    dist = result.model.get_distribution(exp)
    fig, ax = plt.subplots(figsize=(6, 2.3))

    c0 = dorito.plotting.plot_result(
        ax,
        dist,
        pixel_scale=pscale(model),
        cmap="inferno",
        roll_angle_degrees=+exp.parang,
        norm=mpl.colors.PowerNorm(0.3, vmax=0.3),
        diff_lim=dlu.rad2arcsec(wavel / optics_diameter),
        # scale=1.0,
    )

    fig.colorbar(c0)

    ax.set(title=f"NGC1068 - {exp.key}")  #   xticks=ticks, yticks=ticks)

    plt.tight_layout()
    plt.show()

amigo.plotting.plot_losses(result.losses[0], start=int(n_epoch * 0.75))
amigo.plotting.plot(result.history)

for exp in fits:
    exp.print_summary()
    amigo.plotting.summarise_fit(result.model, exp, save_path=output_path)
