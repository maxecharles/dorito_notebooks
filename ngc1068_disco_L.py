# %%
import jax

# jax.config.update("jax_enable_x64", False)
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_debug_nans", True)
print(jax.local_devices()[0].device_kind)
from jax import numpy as np, random as jr, tree as jtu
import os


import zodiax as zdx
from zodiax.optimisation import sgd, adam
import dLux.utils as dlu

from amigo.fitting import Trainer
import dorito


# visualisation
import matplotlib.pyplot as plt
import matplotlib as mpl
import ehtplot
import scienceplots


# import cmasher as cmr

# matplotlib parameters
plt.style.use(["science", "bright", "no-latex"])

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

source_name = "NGC1068"
data_path = os.path.join(path, f"data/JWST/{source_name}/calslope/")
uncal_path = os.path.join(path, f"data/JWST/{source_name}/uncal/")
amigo_cache = os.path.join(path, "data/amigo_files/")

cache = os.path.join(amigo_cache, "v_0.0.10/")
output_path = os.path.join(amigo_cache, f"outputs/{source_name}/")

load_dict = lambda x: np.load(x, allow_pickle=True).item()
disco_path = os.path.join(output_path, f"5672157/")
discos = {}
for filt in os.listdir(disco_path):
    disco = np.load(
        os.path.join(disco_path, filt, "discos.npy"), allow_pickle=True
    ).item()
    for key, value in disco.items():
        discos[key] = value
print(discos.keys())


from datetime import datetime
import sys

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
# output_path = os.path.join(output_path, datetime_str) + "/"

job_id = os.environ.get("SLURM_ARRAY_JOB_ID")

batch_idx = sys.argv[1] if len(sys.argv) > 1 else "0"
output_path = os.path.join(output_path, job_id) + f"/{batch_idx}/"

if not os.path.exists(output_path):
    os.makedirs(output_path)
print(f"Output path: {output_path}")


# %%
def gaussian_2d(shape, center=None, sigma=10, order=2):
    y = np.arange(shape[0])
    x = np.arange(shape[1])
    x, y = np.meshgrid(x, y)
    if center is None:
        center = (shape[1] // 2, shape[0] // 2)

    r2 = (x - center[0]) ** 2 + (y - center[1]) ** 2
    r = np.sqrt(r2)

    supergauss = np.exp(-(r**order) / (2 * sigma**order))
    return supergauss / supergauss.max()


# %%
optics_diameter = 6.603464  # JWST aperture diameter in meters
otf_coords = dlu.pixel_coords(51, 2 * optics_diameter)

ois = [
    # dorito.model_fits.MCAOIFit(oi_data, key, filter=key)
    dorito.model_fits.ResolvedOIFit(oi_data, key, filter=key[:5])
    for key, oi_data in discos.items()
]
# ois = ois[0:2]
for oi in ois:
    print(oi.key)

s = 161
distribution = np.ones((s, s))
distribution *= gaussian_2d((s, s), sigma=5, order=2)


model = dorito.models.ResolvedDiscoModel(
    ois,
    distribution,
    uv_npixels=2 * otf_coords.shape[-1],
    uv_pscale=0.5 * np.diff(otf_coords[0, 0]).mean(),
    oversample=3.0,
)

# Initialise at the mean dirty image
otf = np.load(amigo_cache + "/otf_support.npy")

params = {"log_dist": {}, "base_uv": model.params["base_uv"]}
hypot = lambda x: np.sqrt(np.array([_x**2 for _x in x]).sum(0))
for filt in ["F380M", "F430M", "F480M"]:
    dirty_imgs = [
        oi.dirty_image(model, rotate=True, otf_support=otf)
        for oi in ois
        if oi.filter == filt
    ]

    if dirty_imgs == []:
        print(f"No dirty images for filter {filt}, skipping initialisation.")
        continue
    mean_dirty = hypot(dirty_imgs)
    # mean_dirty *= gaussian_2d((s, s), sigma=20, order=2)
    mean_dirty *= distribution
    init_dist = np.log10(mean_dirty / mean_dirty.sum() + 1e-16)

    for oi in ois:
        if oi.filter == filt:
            params["log_dist"][oi.get_key("log_dist")] = init_dist

model = model.set("params", params)


# %%
def normalise_distribution(model_params, args):
    params = model_params.params

    if "log_dist" in params.keys():
        for k, log_dist in params["log_dist"].items():
            dist = 10**log_dist

            # normalising the distribution
            params["log_dist"][k] = np.log10(dist / dist.sum())

    return model_params.set("params", params), args


def looper(looper, loss_dict):
    loss = np.array([v[-1] for v in loss_dict.values()]).mean(0)
    looper.set_description(f"Loss: {loss:.3e}")


def phase_centre(model, oi):

    distribution = model.get_distribution(oi)

    # Tip/Tilt vectors (ramp in u,v)
    mu = oi.u / np.linalg.norm(oi.u)
    mv = oi.v / np.linalg.norm(oi.v)

    # to log visibilities
    cvis = oi.to_cvis(model, distribution)
    phase = np.angle(cvis)

    # calculating the phase centre of the distribution
    return np.hypot(np.dot(mu, phase), np.dot(mv, phase))


def centered_loss_fn(model, exposure, args, width=1e-3):
    posterior = dorito.stats.disco_regularised_loss_fn(model, exposure, args)[0]

    # phase centre prior
    centre = phase_centre(model, exposure)
    posterior += -jax.scipy.stats.norm.logpdf(centre, loc=0.0, scale=width)

    return posterior, ()


import shutil

shutil.copy(__file__, output_path + "/script.py")

# %%
# n_epoch = len(x)
n_epoch = 30000
config = {
    # "contrast": sgd(1e-8, 10000),
    # "log_dist": adam(5e-2, 0, (1000, 0.1)),
    # "log_dist": adam(2e-3, 0, (1000, 0.1)),
    # "log_dist": adam(6e-4, 0),
    # "log_dist": adam(1e-3, 0, (3000, 0.1)),
    # "log_dist": adam(1e-3, 0),
    # "log_dist": adam(2e-2, 0, (1000, 0.1)),
    "log_dist": adam(5e-3, 0),
}
tvs = np.concat((np.array([0]), np.logspace(1, 4, 20)))

args = {
    "reg_dict": {
        "TV": (tvs[int(batch_idx)], dorito.stats.TV),
        # "ME": (1e5, dorito.stats.ME),
    }
}

trainer = Trainer(
    loss_fn=centered_loss_fn,
    norm_fn=normalise_distribution,
)

trainer = trainer.update_fishers(
    model=model,
    exposures=[],
    parameters=[],
)

# Train the model
result = trainer.train(
    model=model,
    optimisers=config,
    epochs=n_epoch,
    batches=ois,
    args=args,
)

np.save(output_path + "params.npy", result.model.params)

# %%
from amigo import plotting

for oi in ois:
    dist = result.model.get_distribution(oi, rotate=False)
    dist /= dist.max()
    disco_amp, disco_phase = np.split(oi(result.model), 2)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharey=False)

    ax[0].errorbar(
        disco_amp,
        oi.vis,
        yerr=oi.d_vis,
        label="Amplitude Correlations",
        fmt="x",
        alpha=0.3,
        color="midnightblue",
    )
    x_min, x_max = ax[0].get_xlim()
    y_min, y_max = ax[0].get_ylim()
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)
    ax[0].plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")
    ax[0].set_title("Amplitude Correlations")
    ax[0].set_xlabel("Reconstructed Disco Amplitude")
    ax[0].set_ylabel("Calibrated Disco Amplitude")
    ax[0].legend()

    ax[1].errorbar(
        disco_phase,
        oi.phi,
        yerr=oi.d_phi,
        label="Phase Correlations",
        fmt="x",
        alpha=0.3,
        color="indigo",
    )
    x_min, x_max = ax[1].get_xlim()
    y_min, y_max = ax[1].get_ylim()
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)
    ax[1].plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")
    ax[1].set_title("Phase Correlations")
    ax[1].set_xlabel("Reconstructed Disco Phase")
    ax[1].set_ylabel("Calibrated Disco Phase")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(output_path + f"{oi.key}_prebfgs_correlations.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 2.3))

    c0 = dorito.plotting.plot_result(
        ax,
        dist,
        pixel_scale=dlu.rad2arcsec(model.pscale_in),
        cmap=inferno,
        # cmap="cubehelix",
        # cmap="afmhot_10u",
        # norm=mpl.colors.PowerNorm(0.1, vmin=0, vmax=500),
        norm=mpl.colors.PowerNorm(0.5, vmin=0, vmax=0.2),
        # scale=2,
        ticks=[-1, 0, 1],
        diff_lim=0.5 * dlu.rad2arcsec(oi.wavel / optics_diameter),
    )

    fig.colorbar(c0)

    ax.set(title=f"NGC1068 - {oi.key}")  # , xticks=ticks, yticks=ticks)

    plt.tight_layout()
    plt.savefig(output_path + f"{oi.key}_prebfgs_dist.png", dpi=300)
    plt.close()

plotting.plot_losses(result.losses[0], start=int(n_epoch * 0.75), save_path=output_path)
plotting.plot(result.history, save_path=output_path)

# # %%
# import equinox as eqx
# import optimistix as optx


# def joint_solve(
#     model,
#     exposures,
#     reg_args,
#     Solver=optx.BFGS,
#     max_steps=2**16,
#     rtol=1e-4,
#     atol=1e-4,
# ):

#     @eqx.filter_jit
#     def fun(y, args):

#         # unwrapping the args
#         model, exps, reg_args = args

#         # input is the log distribution
#         log_dist = y

#         # setting the model to the new log distribution
#         params = model.params
#         params["log_dist"][exps[0].get_key("log_dist")] = log_dist
#         model = model.set("params", params)

#         # calculating loss
#         loss = [
#             dorito.stats.disco_regularised_loss_fn(model, exp, reg_args)[0]
#             for exp in exps
#         ]
#         loss = np.array(loss).mean()

#         sum_prior = -jax.scipy.stats.norm.logpdf(
#             model.get_distribution(exps[0]).sum(), loc=1.0, scale=1e-5
#         )

#         # phase centre prior
#         centre = phase_centre(model, exps[0])
#         centre_prior = -jax.scipy.stats.norm.logpdf(centre, loc=0.0, scale=1e-3)

#         return loss + sum_prior + centre_prior

#     args_out = {}
#     sols_out = {}

#     for filt in ["F380M", "F430M", "F480M"]:

#         exps = [oi for oi in exposures if oi.filter == filt]
#         for exp in exps:
#             print(exp.key)

#         if len(exps) == 0:
#             print(f"No exposures for filter {filt}, skipping initialisation.")
#             continue

#         X = model.params["log_dist"][filt]
#         _args = (model, exps, reg_args)
#         args_out[filt] = _args

#         print("Initial loss:", fun(X, _args))
#         solver = Solver(rtol=rtol, atol=atol)
#         sol = optx.minimise(fun, solver, X, _args, throw=False, max_steps=max_steps)
#         sols_out[filt] = sol

#         print("Final loss:", fun(sol.value, _args))
#         print(sol.stats["num_steps"], sol.state.num_accepted_steps)
#         print(optx.RESULTS[sol.result])
#         print()

#     params = {"log_dist": {}, "base_uv": model.params["base_uv"]}
#     for exp in ois:
#         log_dist = sols_out[exp.filter].value
#         params["log_dist"][exp.get_key("log_dist")] = log_dist

#     return model.set("params", params)


# # %%
# final_model = result.model

# bfgs_model = joint_solve(
#     # model,
#     final_model,
#     ois,
#     args,
#     Solver=optx.BFGS,
#     max_steps=2**16,
#     rtol=1e-6,
#     atol=1e-6,
# )


# # %%
# for oi in ois:

#     dist = bfgs_model(oi)
#     dist = dist / dist.max()

#     disco_amp, disco_phase = np.split(oi(bfgs_model), 2)

#     fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharey=False)

#     ax[0].errorbar(
#         disco_amp,
#         oi.vis,
#         yerr=oi.d_vis,
#         label="Amplitude Correlations",
#         fmt="x",
#         alpha=0.3,
#         color="midnightblue",
#     )
#     x_min, x_max = ax[0].get_xlim()
#     y_min, y_max = ax[0].get_ylim()
#     min_val = min(x_min, y_min)
#     max_val = max(x_max, y_max)
#     ax[0].plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")
#     ax[0].set_title(f"Amplitude Correlations: {oi.filter}, {oi.parang:.1f} deg")
#     ax[0].set_xlabel("Reconstructed Disco Amplitude")
#     ax[0].set_ylabel("Calibrated Disco Amplitude")
#     ax[0].legend()

#     ax[1].errorbar(
#         disco_phase,
#         oi.phi,
#         yerr=oi.d_phi,
#         label="Phase Correlations",
#         fmt="x",
#         alpha=0.3,
#         color="indigo",
#     )
#     x_min, x_max = ax[1].get_xlim()
#     y_min, y_max = ax[1].get_ylim()
#     min_val = min(x_min, y_min)
#     max_val = max(x_max, y_max)
#     ax[1].plot([min_val, max_val], [min_val, max_val], "r--", label="y=x")
#     ax[1].set_title(f"Phase Correlations: {oi.filter}, {oi.parang:.1f} deg")
#     ax[1].set_xlabel("Reconstructed Disco Phase")
#     ax[1].set_ylabel("Calibrated Disco Phase")
#     ax[1].legend()

#     plt.tight_layout()
#     plt.show()

#     fig, ax = plt.subplots(figsize=(6, 2.3))

#     c0 = dorito.plotting.plot_result(
#         ax,
#         dist,
#         pixel_scale=dlu.rad2arcsec(model.pscale_in),
#         cmap="inferno",
#         norm=mpl.colors.LogNorm(vmin=1e-4),
#         # norm=mpl.colors.PowerNorm(0.5),
#         diff_lim=dlu.rad2arcsec(oi.wavel / optics_diameter) / 2,
#         roll_angle_degrees=-oi.parang,
#         scale=1.2,
#     )

#     fig.colorbar(c0)

#     ax.set(title=f"WR137 - {oi.key}")  #   xticks=ticks, yticks=ticks)

#     plt.tight_layout()
#     plt.savefig(output_path + f"{oi.key}_postbfgs_dist.png", dpi=300)
#     plt.close()
