# %%
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
print(jax.local_devices()[0].device_kind)
from jax import numpy as np, random as jr, tree as jtu
import os


from zodiax.optimisation import sgd, adam

import amigo
import dorito


# visualisation
import matplotlib.pyplot as plt
import matplotlib as mpl
import ehtplot
import scienceplots
from scipy.ndimage import binary_dilation
# import cmasher as cmr

# import sys

# i = sys.argv[1]
# i = int(sys.argv[1])
# print(f"Running script with input {i}")
# # ... your logic here ...
# i = 5


# matplotlib parameters
plt.style.use(["science", "bright", "no-latex"])

plt.rcParams["image.cmap"] = "inferno"
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = "lower"
plt.rcParams["figure.dpi"] = 100
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

source_name = "NGC1068"
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
]
# FILTERS = [i]

# Bind file path, type and exposure type
file_fn = lambda data_path, filters=FILTERS, **kwargs: amigo.files.get_files(
    # "/Users/mcha5804/JWST/ERS1373/calgrps/",
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

output_path = os.path.join(output_path, datetime_str) + "/"
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
    # file["BADPIX"].data[36:66, :25] = 1  # BACKGROUND STAR?

    if not bool(file[0].header["IS_PSF"]):
        # file["BADPIX"].data[40, 45] = 1  # MIDDLE PIXELS
        badpix = np.array(file["BADPIX"].data, dtype=bool)
        im = np.array(file["SLOPE"].data.sum(0))
        im = np.where(badpix, np.nan, im)
        mask = binary_dilation(im == np.nanmax(im), iterations=2)
        file["BADPIX"].data += mask.astype(int)
        sci_files.append(file)
    elif bool(file[0].header["IS_PSF"]):
        file[0].header["TARGPROP"] = "HD 228337"
        cal_files.append(file)
    else:
        print(f"Unkown target: {file[0].header['TARGPROP']}")

dorito.misc.truncate_files(sci_files, 4)


# %%
from astropy.time import Time

for file in files:
    h = file[0].header
    t = Time(h["EXPMID"], format="mjd")
    print(
        f'{h["TARGPROP"]} {h["FILTER"]}, Dither {h["PATT_NUM"]}/{h["NUMDTHPT"]}, Roll {h["ROLL_REF"]:.1f}deg, {h["XPOSURE"] / 60:.1f}min, {t.iso}'
    )

# %%
# from scipy.ndimage import binary_dilation


# def get_depth(exp, threshold=30_000):
#     depth_cube = []
#     for group in exp.ramp[1:]:
#         group = np.where(exp.badpix, np.nan, group)
#         sat_badpix = binary_dilation(group > threshold)
#         depth_cube.append(sat_badpix)

#     return (exp.ngroups - 1) - np.array(depth_cube, dtype=int).sum(0)


# def fudge_cov_pp(cov, d, bajillion):
#     cov = cov.at[d:, :].set(0.0)
#     cov = cov.at[:, d:].set(0.0)
#     cov = cov.at[d:, d:].set(bajillion * np.eye((cov.shape[0] - d)))
#     return cov


# def fudge_cov(cov, depth, bajillion=1e6):
#     """Modifies a (48, 48, 80, 80) covariance array in-place,
#     using per-pixel depths (80, 80)"""
#     h, w = depth.shape
#     final_cov = np.empty_like(cov)

#     for i in range(h):
#         for j in range(w):
#             d = int(depth[i, j])
#             this_cov = fudge_cov_pp(cov[..., i, j], d, bajillion)
#             final_cov = final_cov.at[..., i, j].set(this_cov)

#     return final_cov


# %%
print("Model initialisation...")
load_dict = lambda x: np.load(f"{x}", allow_pickle=True).item()

cal_exposures = [
    amigo.model_fits.SplineVisFit(file, use_cov=True) for file in cal_files
]
sci_exposures = [
    amigo.model_fits.SplineVisFit(file, use_cov=True) for file in sci_files
]
exposures = cal_exposures[0:1] + sci_exposures[0:1]
# exposures = cal_exposures + sci_exposures
# exposures = [
#     exp.set("cov", fudge_cov(exp.cov, get_depth(exp, threshold=25_000)))
#     for exp in exposures
# ]
# exposures = exposures[0:1]

model = amigo.core_models.AmigoModel(
    exposures,
    optics=amigo.optical_models.AMIOptics(),
    detector=amigo.detector_models.LinearDetector(),
    ramp_model=amigo.ramp_models.NonLinearRamp(),
    read=amigo.read_models.ReadModel(),
    vis_model=amigo.vis_models.LogVisModel(
        load_dict(cache + "vis_basis.npy"), n_basis=420
    ),
    state=load_dict(cache + "calibration.npy"),
)


# # %%
# for exp in exposures:
#     exp.print_summary()
#     amigo.plotting.summarise_fit(model, exp, residuals=False)
import shutil
shutil.copy(__file__, output_path + '/script.py') 

# %%
print("Training...")
n_epoch = 20000

config = {
    "positions": sgd(5e-1, 0, (10, 0.1), (500, 0)),
    "fluxes": sgd(5e-2, 5),
    "aberrations": sgd(2e0, 10),
    "spectra": sgd(5e-1, 25),
    "amplitudes": sgd(5e-1, 50),
    "phases": sgd(5e-1, 50),
}

pos_keys = []
spc_keys = []
for exp in exposures:
    if not exp.calibrator:
        pos_keys.append(exp.map_param("positions"))
        spc_keys.append(exp.map_param("spectra"))


def grad_fn(model, grads, args):
    # Nuke the position gradients for the science exposures
    if "positions" in config.keys():
        grads = grads.multiply(pos_keys, 0.1)

    # Reduce spectra gradients for the science exposures
    if "spectra" in config.keys():
        grads = grads.multiply(spc_keys, 0.3)
    return grads, args


trainer = amigo.fitting.Trainer(
    grad_fn=grad_fn,
    cache=os.path.join(amigo_cache, "fishers/"),
)

trainer = trainer.populate_fishers(
    model,
    exposures,
    hessians=load_dict(cache + "jacobians.npy")["hessian"],
    parameters=list(config.keys()),
)

print("Number of exposures: ", len(exposures))

# Train the model
result = trainer.train(
    model=model,
    optimisers=config,
    epochs=n_epoch,
    # batches=exposures[0:1],
    batches=exposures,
    # batches=amigo.fitting.batch_exposures(exposures, n_batch=len(exposures) // 3),
)

# %%
losses = np.array([v for v in result.losses.values()]).mean(0)

amigo.plotting.plot_losses(losses, start=int(0.8 * n_epoch), save_path=output_path)
amigo.plotting.plot(result.history, save_path=output_path)

for exp in exposures:
    print(exp.star)
    amigo.plotting.summarise_fit(result.model, exp, save_path=output_path)

np.save(output_path + "params_postgd.npy", result.model.params, allow_pickle=True)
# %%
stars = list(set([exp.star for exp in exposures]))

for targ_star in stars:
    plt.figure(figsize=(10, 4))
    plt.title(f"{targ_star}")
    for key, spec in result.model.spectra.items():
        star, filt = key.split("_")

        if star != targ_star:
            continue

        wavels, filt_weights = model.filters[filt]
        xs = np.linspace(-1, 1, len(wavels), endpoint=True)
        spectra_slopes = 1 + spec * xs
        weights = filt_weights * spectra_slopes  # * wavels

        #
        filt_weights = filt_weights / filt_weights.sum()
        spectra_slopes = spectra_slopes / spectra_slopes.sum()
        weights = weights / weights.sum()

        plt.plot(wavels, weights, label="Filter Weights")
        plt.plot(wavels, spectra_slopes, label="Spectral Weights")
        plt.plot(wavels, filt_weights, label="Combined Weights")

    plt.legend()
    plt.savefig("spectra.png")
    plt.close()
    # plt.show()

# %%
import equinox as eqx
import optimistix as optx
from amigo.core_models import ModelParams
from tqdm import tqdm


def get_proj_mat(fmat):
    """Get the projection matrix for a Fisher matrix"""
    eig_vals, eig_vecs = np.linalg.eig(-fmat)
    return (eig_vals.real**-0.5)[:, None] * eig_vecs.real.T


proj_mats = jtu.map(get_proj_mat, trainer.fishers)


@eqx.filter_jit
def fun(params, args):
    model, exp, model_params, proj_mats, shapes = args

    # Project and add the parameters
    proj_params = jtu.map(lambda x, y: np.dot(x, y), proj_mats, params)
    model_params = jtu.map(lambda x, y: x + y, model_params, proj_params)
    model_params = jtu.map(lambda x, y: x.reshape(y), model_params, shapes)
    model = model_params.inject(model)

    # Return the loss
    return -np.nanmean(exp.mv_zscore(model))


params = ["amplitudes", "phases", "fluxes", "spectra"]
final_model = result.model

args_out = {}
sols_out = {}
for exp in tqdm(exposures):
    opt_params = {}
    fishers = {}
    shapes = {}
    for param in params:
        key = exp.map_param(param)
        shapes[key] = final_model.get(key).shape
        opt_params[key] = final_model.get(key).flatten()
        fishers[key] = trainer.fishers[f"{exp.key}.{param}"]

    shapes = ModelParams(shapes)
    model_params = ModelParams(opt_params)
    initial_params = jtu.map(lambda x: np.zeros_like(x), model_params)
    proj_mats = jtu.map(get_proj_mat, ModelParams(fishers))
    proj_mats = jtu.map(lambda x: np.where(np.isnan(x) | np.isinf(x), 0, x), proj_mats)

    args = (final_model, exp, model_params, proj_mats, shapes)
    args_out[exp.key] = args

    try:
        print("Initial loss:", fun(initial_params, args))
    except Exception as e:
        print(f"Skipped {exp.key}")
        continue
    solver = optx.BFGS(rtol=1e-8, atol=1e-8)
    sol = optx.minimise(fun, solver, initial_params, args, throw=False, max_steps=2048)
    sols_out[exp.key] = sol

    print("Final loss:", fun(sol.value, args))
    print(sol.stats["num_steps"], sol.state.num_accepted_steps)
    print(optx.RESULTS[sol.result])
    print()

# %%
spectra = {}
for key, args in args_out.items():
    proj_mats = args[3]
    proj_params = jtu.map(lambda x, y: np.dot(x, y), proj_mats, sols_out[key].value)
    model_params = jtu.map(lambda x, y: x + y, args[2], proj_params)
    model_params = jtu.map(lambda x, y: x.reshape(y), model_params, args[4])
    final_model = model_params.inject(final_model)

    # Spectra is joint-fit so we take the mean
    exp = args[1]
    spec = model_params.params[exp.map_param("spectra")]
    spec_key = exp.get_key("spectra")
    if spec_key not in spectra:
        spectra[spec_key] = [spec]
    else:
        spectra[spec_key].append(spec)

spectra = jtu.map(
    lambda x: np.array(x).mean(), spectra, is_leaf=lambda x: isinstance(x, list)
)
final_model = final_model.set("spectra", spectra)

final_state = {}
for key, values in result.state.items():
    final_state[key] = final_model.get(key)

# %%
# final_state = load_dict(output_path + "final_state.npy")

# params = {}
# for key, value in model.params.items():
#     if key in final_state.keys():
#         params[key] = final_state[key]
#     else:
#         params[key] = value
# final_model = model.set("params", params)

# %%
try:
    for filt in FILTERS:
        for exp in exposures:
            if exp.filter == filt:
                exp_type = "Calibrator" if exp.calibrator else "Science"
                print(exp.filter, exp_type, exp.star, exp.ngroups)
                amigo.plotting.summarise_fit(final_model, exp, save_path=output_path)
except Exception as e:
    print(f"Error during plotting: {e}")
    pass


# %%
import jax.tree as jtu
from tqdm import tqdm
from amigo.fisher import FIM


fmats = {}
for exp in tqdm(exposures):

    print("Fishing...", exp.key)
    loglike_fn = lambda model: -np.nansum(exp.loglike(model))
    params = [exp.map_param("amplitudes"), exp.map_param("phases")]
    fmat = FIM(final_model, params, loglike_fn, reduce_ram=True, batch_size=1)
    fmats[exp.get_key("phases")] = fmat
    print("Fished.")


# %%
full_covs = jtu.map(lambda x: np.linalg.inv(x), fmats)

n = model.vis_model.n_basis
amp_covs = jtu.map(lambda x: x[:n, :n], full_covs)
phase_covs = jtu.map(lambda x: x[n:, n:], full_covs)

covs = {
    "amplitudes": amp_covs,
    "phases": phase_covs,
}

plt.figure(figsize=(20, 10))
for i, key in enumerate(amp_covs):
    if i > 5:
        break

    if i < 3:
        param = "amplitude"
        cov = amp_covs[key]
    else:
        param = "phase"
        cov = amp_covs[key]
    plt.subplot(2, 3, i + 1)
    plt.title(f"{param} Covariance")
    v = np.nanmax(np.abs(cov))
    plt.imshow(cov, seismic, vmin=-v, vmax=v)
    # plt.imshow(np.log10(np.abs(cov)), inferno)
    plt.colorbar()
plt.show()

# %%
fit_outputs = {
    "n_basis": final_model.vis_model.n_basis,
}

for exp in exposures:
    get_fn = lambda param: final_model.get(exp.map_param(param))
    fit_outputs[exp.key] = {
        "star": exp.star,
        "calibrator": exp.calibrator,
        "filter": exp.filter,
        "parang": exp.parang,
        "defocus": final_model.get(exp.map_param("defocus")),
        "n_basis": final_model.vis_model.n_basis,
        "fishers": fmats[exp.get_key("phases")],
        "amp_cov": amp_covs[exp.get_key("phases")],
        "phase_cov": phase_covs[exp.get_key("phases")],
        **{param: get_fn(param) for param in final_state.keys()},
    }

np.save(output_path + f"visibilities", fit_outputs, allow_pickle=True)


# %%
# Load the cached states
load_dict = lambda x: np.load(x, allow_pickle=True).item()
cal_values = load_dict(cache + "calibration.npy")
vis_basis = load_dict(cache + "vis_basis.npy")
fit = load_dict(output_path + f"visibilities.npy")
n_basis = fit.pop("n_basis")

optics = amigo.optical_models.AMIOptics(psf_upsample=1)
vis_model = amigo.vis_models.LogVisModel(vis_basis, n_basis=n_basis)

# %% [markdown]
# ## Get the average aberrations per filter for the Kernel visibilities

# %%
# Sort the aberration values by their filter
filters = sorted(list(set([fit_values["filter"] for fit_values in fit.values()])))
aberrations = {k: [] for k in filters}
for exp_key, fit_values in fit.items():
    aberrations[fit_values["filter"]].append(fit_values["aberrations"])

# Get the mean aberration value per filter
leaf_fn = lambda x: isinstance(x, list)
list_mean = lambda x: np.array(x).mean(0)
aberrations = jtu.map(lambda x: list_mean(x), aberrations, is_leaf=leaf_fn)

# %% [markdown]
# ## Calculate the Kernel Visibilities

# %%
from amigo.stats import svd
from amigo.vis_calibration import vis_jac_fn
from amigo.core_models import ModelParams
from amigo.misc import tqdm
import dLux.utils as dlu


thresh = 1e-12
kernel_outputs = {}
for filt in tqdm(filters):
    optics = optics.set("defocus", cal_values["defocus"][filt])

    model_params = ModelParams(
        {
            # Note we dont need to marginalise over positions - its subsumed by abbs
            "defocus": cal_values["defocus"][filt],
            "spectra": np.zeros(1),
            "fluxes": np.zeros(1),
            "abb_coeffs": aberrations[filt],
        }
    )

    jac_fn = lambda X: vis_jac_fn(X, (optics, vis_model, filt))
    amp_fn, phase_fn = lambda X: jac_fn(X)[0], lambda X: jac_fn(X)[1]
    J_amp = model_params.jacfwd(amp_fn, n_batch=30)
    J_phase = model_params.jacfwd(phase_fn, n_batch=30)

    # Get the Jacobian decompositions
    u_amp, s_amp, vh_amp = svd(J_amp)
    u_phase, s_phase, vh_phase = svd(J_phase)

    n_amp = len(s_amp) - np.sum(s_amp < thresh)
    n_phase = len(s_phase) - np.sum(s_phase < thresh)

    # NOTE: After deleting the rows/cols, the matrix is NOT symmetric, so its inverse
    # is NOT the same as the transpose.
    kernel_outputs[filt] = {
        "kernel_mats": {
            "amplitudes": u_amp[:, n_amp:].T,
            "phases": u_phase[:, n_phase:].T,
        },
        "proj_mats": {
            "amplitudes": u_amp[:, : len(s_amp)].T,
            "phases": u_phase[:, : len(s_phase)].T,
        },
        "J_mats": {
            "amplitudes": J_amp,
            "phases": J_phase,
        },
        "singular_vals": {
            "amplitudes": s_amp,
            "phases": s_phase,
        },
    }

# %% [markdown]
# Plotting the singular values.

# %%
plt.figure(figsize=(12, 4))
for i, filt in enumerate(filters):
    s_amp = kernel_outputs[filt]["singular_vals"]["amplitudes"]
    s_phase = kernel_outputs[filt]["singular_vals"]["phases"]

    plt.title(filt)
    plt.plot(s_amp, label=f"{filt} amplitude", c=f"C{i}", alpha=0.5)
    plt.plot(s_phase, label=f"{filt} phase", ls="--", c=f"C{i}", alpha=0.5)
    plt.yscale("log")

plt.axhline(thresh, ls="--", c="k", alpha=0.5, label="Threshold")
plt.legend()
plt.tight_layout()
plt.savefig(output_path + "singular_values.png")
plt.close()

# %% [markdown]
# Plotting the Jacobian responses.

# %%
for filt in filters[:1]:
    lat_amp_Js = kernel_outputs[filt]["J_mats"]["amplitudes"]
    lat_phase_Js = kernel_outputs[filt]["J_mats"]["phases"]

    plt.figure(figsize=(25, 8))
    plt.suptitle(filt)
    for i in range(5):
        log_amp, phase = vis_model.latent_to_im(
            lat_amp_Js[:, i], lat_phase_Js[:, i], filt
        )

        v = np.nanmax(np.abs(log_amp))
        plt.subplot(2, 5, i + 1)
        plt.title(f"Jac amp {i}")
        plt.imshow(log_amp, seismic, vmin=-v, vmax=v)
        plt.colorbar()

        v = np.nanmax(np.abs(phase))
        plt.subplot(2, 5, i + 6)
        plt.title(f"Jac Phase {i}")
        plt.imshow(phase, seismic, vmin=-v, vmax=v)
        plt.colorbar()

    plt.tight_layout()
    plt.savefig(output_path + f"{filt}_J_mats.png")
    plt.close()

# %% [markdown]
# Plotting the projection matrices.

# %%
for filt in filters[:1]:
    amp_vecs = kernel_outputs[filt]["proj_mats"]["amplitudes"]
    phase_vecs = kernel_outputs[filt]["proj_mats"]["phases"]

    plt.figure(figsize=(25, 8))
    plt.suptitle(filt)
    for i in range(5):
        log_amp, phase = vis_model.latent_to_im(amp_vecs[i], phase_vecs[i], filt)

        v = np.nanmax(np.abs(log_amp))
        plt.subplot(2, 5, i + 1)
        plt.title(f"Projection (log amp) {i}")
        plt.imshow(log_amp, seismic, vmin=-v, vmax=v)
        plt.colorbar()

        v = np.nanmax(np.abs(phase))
        plt.subplot(2, 5, i + 6)
        plt.title(rf"Projection  Phase) {i}")
        plt.imshow(phase, seismic, vmin=-v, vmax=v)
        plt.colorbar()

    plt.tight_layout()
    plt.show()

# %% [markdown]
# Plotting the kernel matrices.

# %%
for filt in filters:

    amp_vecs = kernel_outputs[filt]["kernel_mats"]["amplitudes"]
    phase_vecs = kernel_outputs[filt]["kernel_mats"]["phases"]

    plt.figure(figsize=(25, 8))
    for i in range(5):
        log_amp, phase = vis_model.latent_to_im(amp_vecs[i], phase_vecs[i], filt)

        v = np.nanmax(np.abs(log_amp))
        plt.subplot(2, 5, i + 1)
        plt.title(f"K amp {i}")
        plt.imshow(log_amp, seismic, vmin=-v, vmax=v)
        plt.colorbar()

        v = np.nanmax(np.abs(phase))
        plt.subplot(2, 5, i + 6)
        plt.title(f"K Phase {i}")
        plt.imshow(phase, seismic, vmin=-v, vmax=v)
        plt.colorbar()

    plt.tight_layout()
    plt.savefig(output_path + f"{filt}_kernel_mats.png")
    plt.close()


# %%
from amigo.stats import orthogonalise, build_disco
from amigo.vis_calibration import get_mean_wavelength, average_vis_fits, calibrate_vis

# Calibrate all the science stars with all the calibrators
cal_stars = list(set([vals["star"] for vals in fit.values() if vals["calibrator"]]))
sci_stars = list(set([vals["star"] for vals in fit.values() if not vals["calibrator"]]))

# sci_stars = ["PDS-70"]
# sci_stars = ["HD-100546"]
# sci_stars = ["HD-135344B"]
stars = sci_stars + cal_stars


# Populate the fit dict with the extra values we need
for key, vals in fit.items():
    filt = vals["filter"]
    amp_K = kernel_outputs[filt]["kernel_mats"]["amplitudes"]
    phase_K = kernel_outputs[filt]["kernel_mats"]["phases"]

    full_cov = np.linalg.inv(vals["fishers"])
    amp_cov = full_cov[:n_basis, :n_basis]
    phase_cov = full_cov[n_basis:, n_basis:]

    vals["amp_cov"] = amp_cov
    vals["phase_cov"] = phase_cov
    vals["K_amp"] = np.dot(amp_K, vals["amplitudes"])
    vals["K_phase"] = np.dot(phase_K, vals["phases"])
    vals["K_amp_cov"] = np.dot(amp_K, np.dot(amp_cov, np.linalg.pinv(amp_K)))
    vals["K_phase_cov"] = np.dot(phase_K, np.dot(phase_cov, np.linalg.pinv(phase_K)))

    # Get the spectrally weighted wavelengths
    wavels, filt_weights = optics.filters[filt]
    vals["wavel"] = get_mean_wavelength(wavels, filt_weights, vals["spectra"])


# Average over the multiple exposures
vis_outputs = {}
for i, filt in enumerate(filters):
    for is_cal in [True, False]:
        # Get the list of fits to the right star and filter
        stars_in = cal_stars if is_cal else sci_stars
        star_type = "cal" if is_cal else "sci"
        vis_fits = [
            vals
            for vals in fit.values()
            if vals["star"] in stars_in and vals["filter"] == filt
        ]

        # Ensure we actually have fits to this star/filter
        if len(vis_fits) == 0:
            continue

        # Average the fits
        vis_outputs[f"{star_type}_{filt}"] = average_vis_fits(vis_fits)

# Calibrate the outputs
cal_vis_outputs = {}
for i, filt in enumerate(filters):
    sci_key = f"sci_{filt}"
    cal_key = f"cal_{filt}"
    keys = vis_outputs.keys()
    if sci_key not in keys or cal_key not in keys:
        continue

    # Make sure we aren't averaging over roll angles
    parang_std = vis_outputs[sci_key]["parangs"].std(0)
    assert parang_std < 0.1

    # Calibrate the visibilities
    k_cal_vis_dict = calibrate_vis(vis_outputs, filt, kernel=True)
    cal_vis_dict = calibrate_vis(vis_outputs, filt, kernel=False)

    # Projection matrices
    V_amp = vis_model.V_amp[filt]
    V_phase = vis_model.V_phase[filt]
    K_amp_op = kernel_outputs[filt]["kernel_mats"]["amplitudes"]
    K_phase_op = kernel_outputs[filt]["kernel_mats"]["phases"]

    # Orthogonalise the visibilities
    K_vis = k_cal_vis_dict["K_vis"]
    K_phi = k_cal_vis_dict["K_phi"]
    K_vis_cov = k_cal_vis_dict["K_vis_cov"]
    K_phi_cov = k_cal_vis_dict["K_phi_cov"]

    # Orthogonalise the kernel visibilities
    o_vis, o_vis_cov, o_vis_mat, o_vis_eigv = orthogonalise(K_vis, K_vis_cov)
    o_phi, o_phi_cov, o_phi_mat, o_phi_eigv = orthogonalise(K_phi, K_phi_cov)

    # Build the disco matrices
    disco_vis_mat = build_disco(V_amp, K_amp_op, o_vis_mat)
    disco_phi_mat = build_disco(V_phase, K_phase_op, o_phi_mat)

    # Save the Orthonormal calibrated Kernel Observables (Ockos)
    o_cal_vis_dict = {
        "O_vis": o_vis,
        "O_phi": o_phi,
        "O_vis_cov": o_vis_cov,
        "O_phi_cov": o_phi_cov,
        "O_vis_mat": o_vis_mat,
        "O_phi_mat": o_phi_mat,
        "O_vis_eigv": o_vis_eigv,
        "O_phi_eigv": o_phi_eigv,
        "disco_vis_mat": disco_vis_mat,
        "disco_phi_mat": disco_phi_mat,
    }

    # uv coordinates
    parang = vis_outputs[sci_key]["parangs"].mean(0)
    wavel = vis_outputs[sci_key]["wavels"].mean(0)

    # Get the coordinates
    n = vis_model.n_knots**2 // 2
    u, v = vis_model.otf_coords.reshape(2, -1)[:, :n]
    u, v = dlu.rotate_coords(np.array([u, -v]), -dlu.deg2rad(parang))

    sci_outputs = {
        "u": u,
        "v": v,
        "vis_mat": V_amp,
        "phi_mat": V_phase,
        "K_vis_mat": K_amp_op,
        "K_phi_mat": K_phase_op,
        "parang": parang,
        "wavel": wavel,
    }

    cal_vis_dict = {**sci_outputs, **o_cal_vis_dict, **k_cal_vis_dict, **cal_vis_dict}
    cal_vis_outputs[filt] = cal_vis_dict


# np.save(f"{file_path}/results/GTO1242/cal_vis", cal_vis_outputs, allow_pickle=True)
# np.save(f"{file_path}/results/GO1843/cal_vis", cal_vis_outputs, allow_pickle=True)
np.save(output_path + "discos.npy", cal_vis_outputs, allow_pickle=True)

# %% [markdown]
# ## Examine the outputs

# %%
import dLux.utils as dlu

for filt in filters:
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(f"{filt}")
    axes = fig.subplot_mosaic(
        [
            ["amp", "phase"],
            ["K_amp", "K_phase"],
        ],
    )

    for key, vals in fit.items():

        if vals["filter"] != filt:
            continue

        amp = vals["amplitudes"]
        phase = vals["phases"]
        amp_cov = vals["amp_cov"]
        phase_cov = vals["phase_cov"]

        K_amp = vals["K_amp"]
        K_phase = vals["K_phase"]
        K_amp_cov = vals["K_amp_cov"]
        K_phase_cov = vals["K_phase_cov"]

        amp_err = np.sqrt(np.diag(amp_cov))
        phase_err = np.sqrt(np.diag(phase_cov))
        K_amp_err = np.sqrt(np.diag(K_amp_cov))
        K_phase_err = np.sqrt(np.diag(K_phase_cov))

        #
        amp = 100 * (np.exp(amp) - 1)
        amp_err = 100 * (np.exp(amp_err) - 1)
        K_amp = 100 * (np.exp(K_amp) - 1)
        K_amp_err = 100 * (np.exp(K_amp_err) - 1)

        #
        phase = dlu.rad2deg(phase)
        phase_err = dlu.rad2deg(phase_err)
        K_phase = dlu.rad2deg(K_phase)
        K_phase_err = dlu.rad2deg(K_phase_err)

        inds = np.arange(len(amp))
        K_inds = np.arange(len(K_phase))

        #
        star = vals["star"]
        errorbar = lambda ax, x, y, yerr: ax.errorbar(
            x, y, yerr=yerr, alpha=0.25, label=star, marker="o", capsize=5, ms=5, ls=""
        )
        errorbar(axes["amp"], inds, amp, amp_err)
        errorbar(axes["phase"], inds, phase, phase_err)
        errorbar(axes["K_amp"], K_inds, K_amp, K_amp_err)
        errorbar(axes["K_phase"], K_inds, K_phase, K_phase_err)

    axes["amp"].axhline(0, color="k", ls="--", lw=0.5)
    axes["phase"].axhline(0, color="k", ls="--", lw=0.5)
    axes["K_amp"].axhline(0, color="k", ls="--", lw=0.5)
    axes["K_phase"].axhline(0, color="k", ls="--", lw=0.5)

    axes["amp"].set(ylabel="Amplitude (%)", xlabel="index", title="Amplitudes")
    axes["phase"].set(ylabel="Phase (deg)", xlabel="index", title="Phases")
    axes["K_amp"].set(ylabel="Amplitude (%)", xlabel="index", title="Kernel Amplitudes")
    axes["K_phase"].set(ylabel="Phase (deg)", xlabel="index", title="Kernel Phases")

    axes["amp"].legend()
    axes["phase"].legend()
    axes["K_amp"].legend()
    axes["K_phase"].legend()

    plt.tight_layout()
    plt.savefig(output_path + "mosaic1.png")
    plt.close()

# %%
for filt in filters:
    if filt not in cal_vis_outputs.keys():
        continue

    K_amp = cal_vis_outputs[filt]["K_vis"]
    K_phase = cal_vis_outputs[filt]["K_phi"]
    K_amp_cov = cal_vis_outputs[filt]["K_vis_cov"]
    K_phase_cov = cal_vis_outputs[filt]["K_phi_cov"]

    O_amp = cal_vis_outputs[filt]["O_vis"]
    O_phase = cal_vis_outputs[filt]["O_phi"]
    O_amp_cov = cal_vis_outputs[filt]["O_vis_cov"]
    O_phase_cov = cal_vis_outputs[filt]["O_phi_cov"]

    # Get the errors
    K_amp_err = np.sqrt(np.diag(K_amp_cov))
    K_phase_err = np.sqrt(np.diag(K_phase_cov))
    O_amp_err = np.sqrt(np.diag(O_amp_cov))
    O_phase_err = np.sqrt(np.diag(O_phase_cov))

    # Scale to appropriate units
    K_amp = 100 * (np.exp(K_amp) - 1)
    K_phase = dlu.rad2deg(K_phase)
    K_amp_err = 100 * (np.exp(K_amp_err) - 1)
    K_phase_err = dlu.rad2deg(K_phase_err)

    O_amp = 100 * (np.exp(O_amp) - 1)
    O_phase = dlu.rad2deg(O_phase)
    O_amp_err = 100 * (np.exp(O_amp_err) - 1)
    O_phase_err = dlu.rad2deg(O_phase_err)

    inds = np.arange(len(K_phase))

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"Calibrated Visibilitied: {filt}")

    axes = fig.subplot_mosaic(
        [
            ["K_amp", "K_phase"],
            ["O_amp", "O_phase"],
        ],
    )

    #
    axes["K_amp"].errorbar(
        inds,
        K_amp,
        yerr=K_amp_err,
        alpha=0.5,
        label=filt,
        marker="o",
        capsize=5,
        ms=5,
        ls="",
    )
    axes["K_phase"].errorbar(
        inds,
        K_phase,
        yerr=K_phase_err,
        alpha=0.5,
        label=filt,
        marker="o",
        capsize=5,
        ms=5,
        ls="",
    )
    axes["O_amp"].errorbar(
        inds,
        O_amp,
        yerr=O_amp_err,
        alpha=0.5,
        label=filt,
        marker="o",
        capsize=5,
        ms=5,
        ls="",
    )
    axes["O_phase"].errorbar(
        inds,
        O_phase,
        yerr=O_phase_err,
        alpha=0.5,
        label=filt,
        marker="o",
        capsize=5,
        ms=5,
        ls="",
    )

    axes["K_amp"].axhline(0, color="k", ls="--", lw=0.5)
    axes["K_phase"].axhline(0, color="k", ls="--", lw=0.5)
    axes["O_amp"].axhline(0, color="k", ls="--", lw=0.5)
    axes["O_phase"].axhline(0, color="k", ls="--", lw=0.5)

    axes["K_amp"].set(ylabel="Amplitude (%)", xlabel="index", title="Kernel Amplitudes")
    axes["K_phase"].set(ylabel="Phase (deg)", xlabel="index", title="Kernel Phases")
    axes["O_amp"].set(ylabel="Amplitude (%)", xlabel="index", title="Ocko Amplitudes")
    axes["O_phase"].set(ylabel="Phase (deg)", xlabel="index", title="Ocko Phases")

    fig.tight_layout()
    plt.savefig(output_path + "mosaic2.png")
    plt.close()

# %%
from amigo.misc import interp
from jax.scipy.signal import correlate
from amigo.vis_models import vis_to_im

# mask = optics.pupil_mask.calc_mask(optics.wf_npixels, optics.diameter)
mask = optics.pupil_mask.transmission
mask = dlu.downsample(mask, 8, mean=True)
splodges = correlate(mask, mask, method="fft")
splodges /= splodges.max()
splodges = dlu.downsample(splodges, 5, mean=False)
otf_mask = (splodges > 0.25).astype(float)

print(otf_mask.shape)


plt.figure(figsize=(15, 8))
plt.suptitle("Kernel Cleaned Visibilities")
for i, filt in enumerate(cal_vis_outputs.keys()):
    K_vis = cal_vis_outputs[filt]["K_vis"]
    K_phi = cal_vis_outputs[filt]["K_phi"]
    K_vis_mat = cal_vis_outputs[filt]["K_vis_mat"]
    K_phi_mat = cal_vis_outputs[filt]["K_phi_mat"]
    vis_mat = cal_vis_outputs[filt]["vis_mat"]
    phi_mat = cal_vis_outputs[filt]["phi_mat"]

    vis = np.dot(np.linalg.pinv(K_vis_mat), K_vis)
    phi = np.dot(np.linalg.pinv(K_phi_mat), K_phi)

    vis_pix = np.dot(vis, vis_mat)
    phi_pix = np.dot(phi, phi_mat)

    log_amps, phases = vis_to_im(vis_pix, phi_pix, (51, 51))

    knots = dlu.pixel_coords(51, 2)
    samples = dlu.pixel_coords(255, 2)
    log_amps = interp(log_amps, knots, samples)
    phases = interp(phases, knots, samples)

    full_otf = interp(otf_mask, knots, samples)
    log_amps = np.where(full_otf > 0.25, log_amps, np.nan)
    phases = np.where(full_otf > 0.25, phases, np.nan)

    amp = 100 * (np.exp(log_amps) - 1)
    phase = dlu.rad2deg(phases)

    v = np.nanmax(np.abs(amp))
    plt.subplot(2, 3, i + 1)
    plt.title(f"{filt}")
    plt.imshow(amp, coolwarm, vmin=-v, vmax=v)
    plt.colorbar(label="Amplitude (%)")

    v = np.nanmax(np.abs(phase))
    plt.subplot(2, 3, i + 4)
    plt.title(f"{filt}")
    plt.imshow(phase, coolwarm, vmin=-v, vmax=v)
    plt.colorbar(label="Phase (deg)")

plt.tight_layout()
plt.savefig(output_path + "kernelcleanedvis.png")
plt.close()

# %%
from amigo.misc import interp

from jax.scipy.signal import correlate

# mask = optics.pupil_mask.calc_mask(optics.wf_npixels, optics.diameter)
mask = optics.pupil_mask.transmission
mask = dlu.downsample(mask, 8, mean=True)
splodges = correlate(mask, mask, method="fft")
splodges /= splodges.max()
splodges = dlu.downsample(splodges, 5, mean=False)
otf_mask = (splodges > 0.25).astype(float)

print(otf_mask.shape)


plt.figure(figsize=(15, 8))
plt.suptitle("WFE Visibilities")
for i, filt in enumerate(cal_vis_outputs.keys()):
    raw_vis = cal_vis_outputs[filt]["vis"]
    raw_phi = cal_vis_outputs[filt]["phi"]

    #
    vis = cal_vis_outputs[filt]["vis"]
    phi = cal_vis_outputs[filt]["phi"]
    vis_mat = cal_vis_outputs[filt]["vis_mat"]
    phi_mat = cal_vis_outputs[filt]["phi_mat"]

    #
    vis_mat = cal_vis_outputs[filt]["vis_mat"]
    phi_mat = cal_vis_outputs[filt]["phi_mat"]

    vis = np.dot(np.linalg.pinv(K_vis_mat), K_vis)
    phi = np.dot(np.linalg.pinv(K_phi_mat), K_phi)

    vis_pix = np.dot(raw_vis, vis_mat) - np.dot(vis, vis_mat)
    phi_pix = np.dot(raw_phi, phi_mat) - np.dot(phi, phi_mat)

    log_amps, phases = vis_to_im(vis_pix, phi_pix, (51, 51))

    knots = dlu.pixel_coords(51, 2)
    samples = dlu.pixel_coords(255, 2)
    log_amps = interp(log_amps, knots, samples)
    phases = interp(phases, knots, samples)

    full_otf = interp(otf_mask, knots, samples)
    log_amps = np.where(full_otf > 0.25, log_amps, np.nan)
    phases = np.where(full_otf > 0.25, phases, np.nan)

    amp = 100 * (np.exp(log_amps) - 1)
    phase = dlu.rad2deg(phases)

    v = np.nanmax(np.abs(amp))
    plt.subplot(2, 3, i + 1)
    plt.title(f"{filt}")
    plt.imshow(amp, coolwarm, vmin=-v, vmax=v)
    plt.colorbar(label="Amplitude (%)")

    v = np.nanmax(np.abs(phase))
    plt.subplot(2, 3, i + 4)
    plt.title(f"{filt}")
    plt.imshow(phase, coolwarm, vmin=-v, vmax=v)
    plt.colorbar(label="Phase (deg)")

plt.tight_layout()
plt.savefig(output_path + "WFE_visibilities.png")
plt.close()

print("donezo")
