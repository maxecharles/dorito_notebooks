# %% [markdown]
# # Fitting Io in the image plane

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
print(jax.local_devices()[0].device_kind)

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
plt.rcParams["figure.dpi"] = 250
plt.rcParams["font.size"] = 12
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"

# %%
from socket import gethostname

if gethostname() == "glinton":
    path = "/media/morgana1/snert/max/"
elif gethostname() == "AJQ4YHQH9TX":
    path = "/Volumes/morgana1/snert/max/"
else:
    abers_dir = "/fred/oz440/max/code/dorito_notebooks/files/"
    path = "/fred/oz440/max/"

source_name = "NGC1068"
# data_path = os.path.join(path, f"data/JWST/{source_name}/calslope/")
data_path = os.path.join(path, f"data/JWST/{source_name}/new_calslope/new_calslope/")
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

from datetime import datetime

form = "%d-%m-%y_%H-%M-%S.%f"
now = datetime.now()
datetime_str = now.strftime(form)

# # clear directory of empty folders
# for folder in os.listdir(output_path):
#     folder_dir = os.path.join(output_path, folder)

#     # skip if not a directory
#     if not os.path.isdir(folder_dir):
#         continue

#     # if the folder is empty
#     if len(os.listdir(folder_dir)) == 0:

#         try:
#             then = datetime.strptime(folder, form)

#             # remove empty folder if it is older than 1 hour
#             if (now - then).seconds > 3600:  # 1 hour
#                 print(f"Removing empty folder: {folder_dir}")
#                 os.rmdir(folder_dir)
#         except ValueError:
#             # if the folder name is not in the correct format, skip it
#             print(f"Deleting folder: {folder_dir} (not in correct format)")
#             os.rmdir(folder_dir)

# datetime_str = f"{i}_groups"
print(datetime_str)

batch_idx = sys.argv[1] if len(sys.argv) > 1 else "0"

job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
if job_id is None:
    job_id = "local_test"

output_path = os.path.join(output_path, job_id) + f"/{batch_idx}/"
if not os.path.exists(output_path):
    os.makedirs(output_path)
print(f"Output path: {output_path}")

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

    if not bool(file[0].header["IS_PSF"]):
        sci_files.append(file)
    elif bool(file[0].header["IS_PSF"]):
        file[0].header["TARGPROP"] = "HD 228337"
        cal_files.append(file)
    else:
        print(f"Unkown target: {file[0].header['TARGPROP']}")

dorito.misc.truncate_files(sci_files, 4)

from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from photutils.psf.matching import TukeyWindow
import numpy as onp

source_size = 101  # pixels
npix = source_size

pixel_grid = onp.zeros((npix, npix))
basis = onp.zeros((npix * npix, npix * npix))

mask = TukeyWindow(alpha=0.0)((npix, npix))


for j in tqdm(range(npix)):
    for i in range(npix):
        pixel_grid = 0 * onp.ones_like(pixel_grid)
        pixel_grid[j, i] = 1
        convolved = gaussian_filter(pixel_grid, sigma=1.0)# * mask
        # convolved = pixel_grid
        basis[:, j * npix + i] = convolved.flatten()

eigvals, eigvecs = np.linalg.eigh(basis)
eigvals, eigvecs = eigvals.real[::-1], eigvecs.real[..., ::-1]
basis_dict = {"eigvals": eigvals, "eigvecs": eigvecs}

n_terms = [
    100,
    400,
    700,
    1000,
    1300,
    1600,
    1800,
    2000,
    2500,
    3000,
    3500,
    4000,
    4500,
    5000,
    len(eigvals),
][int(batch_idx)]
# n_terms = 1600
print(f"Using {n_terms} basis terms")


# %%
def gaussian_2d(shape, center=None, sigma=10):
    x = np.arange(0, shape[1])
    y = np.arange(0, shape[0])
    x, y = np.meshgrid(x, y)
    if center is None:
        center = (shape[1] // 2, shape[0] // 2)
    gauss = np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma**2))
    return gauss / gauss.sum()


def normalise_coeffs(basis, coeffs):
    dist = basis.from_eigenbasis(coeffs)
    dist = dist / dist.sum()
    return basis.to_eigenbasis(dist)


# source_size = 81  # pixels
load_dict = lambda x: np.load(f"{x}", allow_pickle=True).item()

# sci_fits = [DynamicResolvedFit(file, source_size) for file in sci_files]
# sci_fits = [dorito.model_fits.ResolvedFit(file) for file in sci_files]
sci_fits = [dorito.model_fits.TransformedResolvedFit(file) for file in sci_files]
cal_fits = [amigo.model_fits.PointFit(file) for file in cal_files]

# I only want to use the calibrator in the same primary dither position
fits = sci_fits[0:1] + cal_fits[0:1]
# fits = sci_fits + cal_fits[0:1]

# building the model
init_dist = np.ones((source_size, source_size)) / (source_size**2)
init_dist = gaussian_2d(init_dist.shape, sigma=15)
plt.imshow(init_dist)
plt.show()

abbs = np.load(abers_dir+"ngc_abbs.npy", allow_pickle=True).item()
state = load_dict(cache + "calibration.npy")
state["aberrations"]["F480M"] = abbs["aberrations"]["01260_F480M"]

basis = dorito.models.ImageBasis(basis_dict, n_basis=n_terms)


# building the model
model = dorito.models.TransformedResolvedModel(
    # model = dorito.models.ResolvedAmigoModel(
    exposures=fits,
    optics=amigo.optical_models.AMIOptics(),
    detector=amigo.detector_models.LinearDetector(),
    ramp_model=amigo.ramp_models.NonLinearRamp(),
    read=amigo.read_models.ReadModel(),
    basis=basis,
    state=load_dict(cache + "calibration.npy"),
    param_initers={
        "coeffs": normalise_coeffs(
            basis, basis.to_eigenbasis(init_dist / init_dist.sum())
        )
    },
    rotate=False,
)

# %%
# for exp in fits:
    # exp.print_summary()
    # amigo.plotting.summarise_fit(model, exp, residuals=False)

# %% [markdown]
# ## Optimisation Stage 1: Gradient Descent

# %%
pos_keys = []
spc_keys = []
for exp in fits:
    if not exp.calibrator:
        pos_keys.append(exp.map_param("positions"))
        spc_keys.append(exp.map_param("spectra"))


def loLU(x, lol=1e6):
    return np.log(1 + np.exp(lol * x))


def norm_fn(model_params, args):
    params = model_params.params

    if "log_dist" in params.keys():
        for k, coeffs in params["log_dist"].items():
            dist = basis.from_eigenbasis(coeffs)
            dist = dist / dist.sum()
            coeffs = basis.to_eigenbasis(dist)
            params["log_dist"][k] = coeffs

    if "spectra" in params.keys():
        spectra = jtu.map(
            lambda x: np.clip(x, a_min=-0.8, a_max=0.8), params["spectra"]
        )
        params["spectra"] = spectra

    return model_params.set("params", params), args


def loss_fn(model, exp, args={"reg_dict": {}}):
    # this is per exposure

    # regular likelihood term
    likelihood = -np.nanmean(exp.mv_zscore(model))

    # prior on the sum
    dist = model(exp)
    # total_sum = dist.sum()
    # prior = -jax.scipy.stats.norm.logpdf(total_sum, loc=1.0, scale=0.00001)
    prior = 0
    prior += loLU(-dist).sum()
    prior += (
        dorito.stats.apply_regularisers(model, exp, args) if not exp.calibrator else 0.0
    )
    # prior = 1e6 * jax.nn.relu(-dist).sum()
    return likelihood + prior, ()


pscale = lambda model: model.optics.psf_pixel_scale / model.optics.oversample

# %%
n_epoch = 12000

config = {
    "positions": sgd(5e-2, 0, (500, 0.)),
    "fluxes": sgd(1e-2, 0),
    # "aberrations": sgd(5e-1, 5),
    # "log_dist": adam(5e-2, 10),
    "log_dist": adam(1e-5, 15),
    "spectra": sgd(5e-2, 500),
}


def grad_fn(model, grads, args):
    # Nuke the position gradients for the science exposures
    if "positions" in config.keys():
        grads = grads.multiply(pos_keys, 0.3)

    # Reduce spectra gradients for the science exposures
    if "spectra" in config.keys():
        grads = grads.multiply(spc_keys, 1e-2)
    return grads, args


trainer = amigo.fitting.Trainer(
    loss_fn=loss_fn,
    norm_fn=norm_fn,
    # grad_fn=grad_fn,
    cache=os.path.join(amigo_cache, "fishers/"),
)

print("Populating fishers...")
trainer = trainer.populate_fishers(
    # model.set("detector.ramp.bleed", False).set("params", params),
    model,
    fits[0:1],
    hessians=load_dict(cache + "jacobians.npy")["hessian"],
    parameters=[p for p in config.keys()],  # if p not in ["log_distribution"]],
)

print("Number of exposures: ", len(fits))

# Train the model
result = trainer.train(
    model=model,
    optimisers=config,
    epochs=n_epoch,
    batches=fits[0:1],  # Just the science
    # batches=fits[-1:],  # Just the calibrator
    # batches=fits,
)

# %%

np.save(output_path + "params.npy", result.model.params, allow_pickle=True)
result_model = result.model

# balance_dict = dorito.stats.ramp_posterior_balances(result_model, sci_fits, args)
# np.save(output_path + "balance.npy", balance_dict, allow_pickle=True)

from dLux import utils as dlu

optics_diameter = 6.603464  # JWST aperture diameter in meters
def eff_wavel(model, filt):
    wavels, weights = model.filters[filt]
    return np.dot(wavels, weights)



for exp in fits[0:1]:

    if exp.calibrator:
        continue
    dist = result.model.get_distribution(exp, rotate=False)
    fig, ax = plt.subplots(figsize=(6, 2.3))
    print(dist.sum(), dist.min(), dist.max())
    c0 = dorito.plotting.plot_result(
        ax,
        dist,
        pixel_scale=model.psf_pixel_scale / model.oversample,
        norm=mpl.colors.LogNorm(vmin=1e-5),
        diff_lim=dlu.rad2arcsec(eff_wavel(model, exp.filter) / optics_diameter),
    )

    fig.colorbar(c0)

    ax.set(title=f"NGC1068 - {exp.key}")  #   xticks=ticks, yticks=ticks)

    plt.tight_layout()
    plt.savefig(output_path + f"{exp.key}_dist.png", dpi=300)
    plt.close()

for exp in fits[0:1]:
    exp.print_summary()
    amigo.plotting.summarise_fit(result.model, exp, save_path=output_path)

amigo.plotting.plot_losses(
    result.losses[0], start=int(n_epoch * 0.75), save_path=output_path
)
amigo.plotting.plot(result.history, save_path=output_path)

# # %%
# fft = np.fft.fftshift(np.fft.fft2(dist))

# plt.figure(figsize=(8, 3))
# plt.subplot(1, 2, 1)
# plt.title("np.abs(fft)")
# plt.imshow(np.abs(fft), cmap="viridis", norm=mpl.colors.LogNorm(vmin=None))
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.title("np.angle(fft)")
# plt.imshow(np.angle(fft), vmin=-np.pi, vmax=np.pi, cmap="twilight")
# plt.colorbar()
# plt.show()
