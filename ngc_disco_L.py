# %%
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")
print(jax.local_devices()[0].device_kind)
from jax import numpy as np
import os
import sys


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

source_name = "WR137"
data_path = os.path.join(path, f"data/JWST/{source_name}/calslope/")
uncal_path = os.path.join(path, f"data/JWST/{source_name}/uncal/")
amigo_cache = os.path.join(path, "data/amigo_files/")

cache = os.path.join(amigo_cache, "v_0.0.10/")
output_path = os.path.join(amigo_cache, f"outputs/{source_name}/")

load_dict = lambda x: np.load(x, allow_pickle=True).item()
cal_vis_outputs = load_dict(os.path.join(output_path, "all_cal_vis.npy"))

print(cal_vis_outputs.keys())

# %%
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
optics_diameter = 6.603464  # JWST aperture diameter in meters
otf_coords = dlu.pixel_coords(51, 2 * optics_diameter)

ois = [
    dorito.model_fits.ResolvedOIFit(oi_data, key, filter=key[:5])
    for key, oi_data in cal_vis_outputs.items()
]

ois = [oi for oi in ois if oi.filter in ["F480M"]]
ois = ois[1:]
# ois = ois[1:3] + ois[4:]

# model = Model(
size = 151
model = dorito.models.ResolvedDiscoModel(
    ois,
    distribution=np.ones((size, size)),
    uv_npixels=2 * otf_coords.shape[-1],
    uv_pscale=0.5 * np.diff(otf_coords[0, 0]).mean(),
    oversample=6.0,
    rotate=True,
)


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

# %%
import shutil

shutil.copy(__file__, output_path + "/script.py")

# %%
n_epoch = 20000
config = {
    # "log_dist": adam(1e-1, 0, (5000, 0.3)),
    "log_dist": adam(5e-3, 0),
}

mes = np.concat((np.array([0]), np.logspace(1, 7, 29)))

args = {
    "reg_dict": {
        "ME": (mes[int(batch_idx)], dorito.stats.ME),
        # "ME": (1e5, dorito.stats.ME),
    }
}

trainer = Trainer(
    loss_fn=centered_loss_fn,
    # loss_fn=dorito.stats.regularised_loss_fn,
    norm_fn=normalise_distribution,
    looper_fn=looper,
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
    # batches=batch_exposures(ois, 4),
    args=args,
)

# %%
def disco_posterior_balance(model, exp, args={"reg_dict": {}}):
    # this is per exposure
    # NOTE this might not work for multiple regularisers

    # regular likelihood term
    likelihood = dorito.stats.oi_log_likelihood(model, exp)


    # evaluating the regularisation term with each for each regulariser
    priors = [fun(model, exp) for _, fun in args["reg_dict"].values()]
    prior = np.array(priors).sum()

    return likelihood, prior


def disco_posterior_balances(model, exposures, args={"reg_dict": {}}):

    balances = np.array([disco_posterior_balance(model, exp, args) for exp in exposures]).T

    return {
        "likelihoods": balances[0],
        "priors": balances[1],
        "exp_keys": [exp.key for exp in exposures],
        "args": args,
    }


np.save(output_path + "prebfgs_params.npy", result.model.params, allow_pickle=True)
result_model = result.model

balance_dict = disco_posterior_balances(result_model, ois, args)
np.save(output_path + "prebfgs_balance.npy", balance_dict, allow_pickle=True)


# %%
from amigo import plotting

for oi in ois:
    dist = result.model.get_distribution(oi, rotate=False)

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
    # plt.show()
    plt.savefig(output_path + f"{oi.key}_prebfgs_correlations.png", dpi=300)
    plt.close()

    ticks = [-0.5, -0.25, 0, 0.25, 0.5]

    fig, ax = plt.subplots(figsize=(6, 2.3))

    c0 = dorito.plotting.plot_result(
        ax,
        dist,
        pixel_scale=dlu.rad2arcsec(model.pscale_in),
        cmap="afmhot_10u",
        # power=0.3,
        # vmin=0,
        diff_lim=dlu.rad2arcsec(oi.wavel / optics_diameter),
        # scale=1.0,
    )

    fig.colorbar(c0)

    ax.set(title=f"WR137 - {oi.key}")  #   xticks=ticks, yticks=ticks)

    plt.tight_layout()
    # plt.show()
    plt.savefig(output_path + f"{oi.key}_prebfgs_dist.png", dpi=300)
    plt.close()

plotting.plot_losses(result.losses[0], start=int(n_epoch * 0.75), save_path=output_path)
plotting.plot(result.history, save_path=output_path)

# %%
import equinox as eqx
import optimistix as optx


def joint_solve(
    model,
    exposures,
    reg_args,
    Solver=optx.BFGS,
    max_steps=2**16,
    rtol=1e-4,
    atol=1e-4,
):

    @eqx.filter_jit
    def fun(y, args):

        # unwrapping the args
        model, exps, reg_args = args

        # input is the log distribution
        log_dist = y

        # setting the model to the new log distribution
        params = model.params
        params["log_dist"][exps[0].get_key("log_dist")] = log_dist
        model = model.set("params", params)

        # calculating loss
        loss = [
            dorito.stats.disco_regularised_loss_fn(model, exp, reg_args)[0]
            for exp in exps
        ]
        loss = np.array(loss).mean()

        sum_prior = -jax.scipy.stats.norm.logpdf(
            model.get_distribution(exps[0]).sum(), loc=1.0, scale=1e-5
        )

        # phase centre prior
        centre = phase_centre(model, exps[0])
        centre_prior = -jax.scipy.stats.norm.logpdf(centre, loc=0.0, scale=1e-3)

        return loss + sum_prior + centre_prior

    args_out = {}
    sols_out = {}

    for filt in ["F380M", "F430M", "F480M"]:

        exps = [oi for oi in exposures if oi.filter == filt]
        for exp in exps:
            print(exp.key)

        if len(exps) == 0:
            print(f"No exposures for filter {filt}, skipping initialisation.")
            continue

        X = model.params["log_dist"][filt]
        _args = (model, exps, reg_args)
        args_out[filt] = _args

        print("Initial loss:", fun(X, _args))
        solver = Solver(rtol=rtol, atol=atol)
        sol = optx.minimise(fun, solver, X, _args, throw=False, max_steps=max_steps)
        sols_out[filt] = sol

        print("Final loss:", fun(sol.value, _args))
        print(sol.stats["num_steps"], sol.state.num_accepted_steps)
        print(optx.RESULTS[sol.result])
        print()

    params = {"log_dist": {}, "base_uv": model.params["base_uv"]}
    for exp in ois:
        log_dist = sols_out[exp.filter].value
        params["log_dist"][exp.get_key("log_dist")] = log_dist

    return model.set("params", params)


# %%
final_model = result.model


bfgs_model = joint_solve(
    final_model,
    ois,
    args,
    Solver=optx.BFGS,
    max_steps=2**16,
    rtol=1e-4,
    atol=1e-4,
)

np.save(output_path + "params.npy", bfgs_model.params, allow_pickle=True)
result_model = result.model

balance_dict = disco_posterior_balances(bfgs_model, ois, args)
np.save(output_path + "balance.npy", balance_dict, allow_pickle=True)
# %%
for oi in ois:
    dist = bfgs_model(oi)
    dist = dist / dist.max()

    disco_amp, disco_phase = np.split(oi(bfgs_model), 2)

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
    plt.savefig(output_path + f"{oi.key}_postbfgs_correlations.png", dpi=300)
    plt.close()

    ticks = [-0.5, -0.25, 0, 0.25, 0.5]

    fig, ax = plt.subplots(figsize=(6, 2.3))

    c0 = dorito.plotting.plot_result(
        ax,
        dist,
        pixel_scale=dlu.rad2arcsec(model.pscale_in),
        cmap="afmhot_u",
        norm=mpl.colors.LogNorm(vmin=1e-4),
        # norm=mpl.colors.PowerNorm(0.5),
        diff_lim=dlu.rad2arcsec(oi.wavel / optics_diameter) / 2,
        roll_angle_degrees=-oi.parang,
        scale=1.2,
    )

    fig.colorbar(c0)

    ax.set(title=f"WR137 - {oi.key}")  #   xticks=ticks, yticks=ticks)

    plt.tight_layout()
    plt.savefig(output_path + f"{oi.key}_postbfgs_dist.png", dpi=300)
    plt.close()

