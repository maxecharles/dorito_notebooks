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
import numpy
import os
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
    path = "/fred/oz440/max/"

source_name = "NGC1068"
# data_path = os.path.join(path, f"data/JWST/{source_name}/calslope/")
data_path = os.path.join(path, f"data/JWST/{source_name}/new_calslope/")
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

    if not bool(file[0].header["IS_PSF"]):
        sci_files.append(file)
    elif bool(file[0].header["IS_PSF"]):
        file[0].header["TARGPROP"] = "HD 228337"
        cal_files.append(file)
    else:
        print(f"Unkown target: {file[0].header['TARGPROP']}")

dorito.misc.truncate_files(sci_files, 4)

# %%
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from photutils.psf.matching import TukeyWindow
import numpy as onp

source_size = 81  # pixels
npix = source_size

pixel_grid = onp.zeros((npix, npix))
basis = onp.zeros((npix * npix, npix * npix))

mask = TukeyWindow(alpha=0.2)((npix, npix))

# plt.imshow(mask)
# plt.colorbar()
# plt.show()

for j in tqdm(range(npix)):
    for i in range(npix):
        pixel_grid = 0 * onp.ones_like(pixel_grid)
        pixel_grid[j, i] = 1
        convolved = gaussian_filter(pixel_grid, sigma=3.0)  # * mask
        # convolved = pixel_grid
        basis[:, j * npix + i] = convolved.flatten()

eigvals, eigvecs = np.linalg.eigh(basis)
eigvals, eigvecs = eigvals.real[::-1], eigvecs.real[..., ::-1]
basis_dict = {"eigvals": eigvals, "eigvecs": eigvecs}

# %%
# for i in range(10):
#     plt.imshow(
#         eigvecs[..., i].reshape((npix, npix)),
#         cmap="coolwarm",
#         norm=mpl.colors.CenteredNorm(),
#     )
#     plt.colorbar()
#     plt.show()

n1, n2 = 1201, 2500
plt.plot(np.log10(eigvals))
plt.axvline(n1, color="k", ls="--")
plt.axvline(n2, color="k", ls="--")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im = axes[0].imshow(eigvecs[:, n1].reshape((npix, npix)), cmap="coolwarm")
axes[0].set_title("Decent Example")
fig.colorbar(im, ax=axes[0])

im = axes[1].imshow(eigvecs[:, n2].reshape((npix, npix)), cmap="coolwarm")
axes[1].set_title("Bad Checquerboard")
fig.colorbar(im, ax=axes[1])
plt.show()

# %%
# def get_envelope(s, window=2000):
#     diff = -onp.diff(np.log10(s))
#     diff[window:] = 0
#     plt.plot(diff)
#     maxima = onp.maximum.accumulate(diff[::-1])[::-1]
#     plt.plot(maxima)
#     plt.xlim(0, window)
#     plt.show()
#     return maxima


# def get_knee(s, window=2000):
#     maxima = get_envelope(s, window=window)
#     return onp.min(onp.where(maxima < maxima[0] * 0.8))


# get_envelope(eigvals, window=3000)

# get_knee(eigvals, window=3000)

# %%
from jaxtyping import Array
import equinox as eqx
from amigo.model_fits import ModelFit
from dorito.model_fits import ResolvedFit


class ImageBasis(zdx.Base):

    M: Array
    M_inv: Array
    n_basis: int = eqx.field(static=True)
    size: int = eqx.field(static=True)

    def __init__(self, basis_dict: dict, n_basis: int):
        self.n_basis = n_basis
        self.M = basis_dict["eigvecs"][:, :n_basis]
        self.M_inv = np.linalg.pinv(self.M)
        self.size = int(np.sqrt(self.M.shape[0]))

    def to_eigenbasis(self, img: Array) -> Array:
        return np.dot(self.M_inv, img.flatten())

    def from_eigenbasis(self, coeffs: Array) -> Array:
        return np.dot(self.M, coeffs).reshape((self.size, self.size))


class TransformedResolvedModel(dorito.models.ResolvedAmigoModel):

    basis: None

    def __init__(
        self,
        exposures,
        optics,
        detector,
        ramp_model,
        read,
        basis: ImageBasis,
        state,
        source_oversample=1,
        param_initers: dict = None,
    ):

        # This seems to fix some recompile issues
        def fn(x):
            if isinstance(x, jax.Array):
                if "i" in x.dtype.str:
                    return x
                return np.array(x, dtype=float)
            return x

        self.basis = jtu.map(lambda x: fn(x), basis)

        super().__init__(
            exposures,
            optics,
            detector,
            ramp_model,
            read,
            state,
            source_oversample,
            param_initers,
        )

    def get_distribution(self, exposure, rotate=True):
        """
        Get the distribution from the exposure.

        Args:
            exposure: The exposure object containing the distribution key.
        Returns:
            Array: The intensity distribution of the source.
        """
        # distribution = 10 ** self.params["log_dist"][exposure.get_key("log_dist")]
        coeffs = self.params["log_dist"][exposure.get_key("log_dist")]
        # distribution = 10 ** self.basis.from_eigenbasis(coeffs)
        distribution = self.basis.from_eigenbasis(coeffs)
        # distribution += np.array([-distribution.min(), 0]).max()
        # distribution = distribution / distribution.sum()
        # if rotate:
        #     distribution = exposure.rotate(distribution)

        return distribution


class TransformedResolvedFit(ResolvedFit):
    """
    Model fit for resolved sources. This adds the log distribution parameter.
    """

    def initialise_params(self, optics, coeffs):
        """
        Initialise the parameters for the resolved source model fit.
        The log distribution is set to a uniform distribution specified by the source size.

        Args:
            optics: The optics object (to pass to the parent class).
            source_size: The size of the source distribution (assumed square).

        Returns:
            params: A dictionary containing the initialised parameters for the model fit.
        """

        params = ModelFit.initialise_params(self, optics)

        # log distribution
        params["log_dist"] = (self.get_key("log_dist"), coeffs)

        return params


# %% [markdown]
# ## Building the model
#
# We are going to have to build the exposures. DORITO has a `ResolvedFit` class build in, however this will jointly fit all exposures from the same filter. Since we want to capture Io's rotation in a time series, we instead want to uniquely fit all five epochs. To do this, we will write a child class of `ResolvedFit` and amend the `get_key` method. By adding the `self.key` to the `log_distribution` parameter key, this ensures each exposure will fit a unique distribution.

# %%
tukey_basis = ImageBasis(basis_dict, n_basis=1200)


# %%
def normalise_coeffs(basis, coeffs):
    dist = basis.from_eigenbasis(coeffs)
    # dist = 10 ** basis.from_eigenbasis(coeffs)
    dist = dist / dist.sum()
    return basis.to_eigenbasis(dist)
    # return basis.to_eigenbasis(np.log10(dist))


# coeffs = np.log10(onp.random.dirichlet(np.ones(tukey_basis.n_basis), size=1))

# dist = tukey_basis.from_eigenbasis(np.zeros_like(coeffs))

# # coeffs = onp.random.normal(size=(tukey_basis.n_basis,))
# coeffs = tukey_basis.to_eigenbasis(np.log10(mask / mask.sum() + 1e-6))

# print(coeffs.shape)
# dist = tukey_basis.from_eigenbasis(coeffs)
# dist = 10**dist
# dist /= dist.sum()
# new_coeffs = tukey_basis.to_eigenbasis(np.log10(dist))
# new_dist = 10 ** (tukey_basis.from_eigenbasis(new_coeffs))

# # norm_coeffs = normalise_coeffs(tukey_basis, coeffs)
# # round_dist = 10 ** tukey_basis.from_eigenbasis(norm_coeffs)

# plt.figure(figsize=(11, 2))
# plt.subplot(1, 3, 1)
# plt.imshow(dist, norm=mpl.colors.PowerNorm(1, vmin=None))
# plt.title(f"{dist.sum():.2f}")
# plt.colorbar()
# plt.subplot(1, 3, 2)
# plt.imshow(new_dist, norm=mpl.colors.PowerNorm(1, vmin=None))
# plt.title(new_dist.sum())
# plt.colorbar()
# plt.subplot(1, 3, 3)
# # plt.scatter(range(tukey_basis.n_basis), coeffs, marker="^")
# for idx, val in enumerate(coeffs):
#     plt.plot([idx, idx], [0, val], color="pink", alpha=0.8, linewidth=2)
# for idx, val in enumerate(new_coeffs):
#     plt.plot([idx, idx], [0, val], color="blue", alpha=0.5, linewidth=0.8)
# plt.ylim(-5, 5)
# plt.show()


# %%
def gaussian_2d(shape, center=None, sigma=10):
    x = np.arange(0, shape[1])
    y = np.arange(0, shape[0])
    x, y = np.meshgrid(x, y)
    if center is None:
        center = (shape[1] // 2, shape[0] // 2)
    gauss = np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma**2))
    return gauss / gauss.sum()


# source_size = 81  # pixels
load_dict = lambda x: np.load(f"{x}", allow_pickle=True).item()

# sci_fits = [DynamicResolvedFit(file, source_size) for file in sci_files]
# sci_fits = [dorito.model_fits.ResolvedFit(file) for file in sci_files]
sci_fits = [TransformedResolvedFit(file) for file in sci_files]
cal_fits = [amigo.model_fits.PointFit(file) for file in cal_files]

# I only want to use the calibrator in the same primary dither position
fits = sci_fits[0:1] + cal_fits[0:1]
# fits = sci_fits + cal_fits[0:1]

# building the model
init_dist = np.ones((source_size, source_size)) / (source_size**2)
init_dist = gaussian_2d(init_dist.shape, sigma=15)
plt.imshow(init_dist)
plt.show()

abbs = np.load("files/abers.npy", allow_pickle=True).item()
state = load_dict(cache + "calibration.npy")
state["aberrations"]["F430M"] = abbs["01373_F430M"]

model = TransformedResolvedModel(
    # model = dorito.models.ResolvedAmigoModel(
    exposures=fits,
    optics=amigo.optical_models.AMIOptics(),
    detector=amigo.detector_models.LinearDetector(),
    ramp_model=amigo.ramp_models.NonLinearRamp(),
    read=amigo.read_models.ReadModel(),
    basis=tukey_basis,
    state=state,
    # param_initers={"coeffs": np.array(new_coeffs)},
    # param_initers={"coeffs": tukey_basis.to_eigenbasis(init_dist)},
    param_initers={
        "coeffs": normalise_coeffs(
            tukey_basis,
            tukey_basis.to_eigenbasis(init_dist / init_dist.sum()),
            # tukey_basis.to_eigenbasis(np.log10(init_dist / init_dist.sum())),
        )
    },
    # param_initers={"distribution": init_dist},
    # param_initers={"coeffs": tukey_basis.to_eigenbasis(mask / mask.sum())},
    # param_initers={"coeffs": tukey_basis.to_eigenbasis(np.log10(mask / mask.sum()))},
)

# %%
plt.imshow(model.get_distribution(sci_fits[0]))
plt.colorbar()
plt.show()

# %%
for exp in fits:
    exp.print_summary()
    amigo.plotting.summarise_fit(model, exp, residuals=False)

# %% [markdown]
# ## Optimisation Stage 1: Gradient Descent

# %%
pos_keys = []
spc_keys = []
for exp in fits:
    if not exp.calibrator:
        pos_keys.append(exp.map_param("positions"))
        spc_keys.append(exp.map_param("spectra"))


def loLU(x, lol=1e3):
    return np.log(1 + np.exp(lol * x))


def norm_fn(model_params, args):
    params = model_params.params
    # if "log_dist" in params.keys():
    #     for k, coeffs in params["log_dist"].items():
    #         # distribution = tukey_basis.from_eigenbasis(coeffs)
    #         # distribution = np.clip(distribution, 0)
    #         distribution = 10 ** tukey_basis.from_eigenbasis(coeffs)
    #         distribution /= distribution.sum()
    #         distribution = np.log10(distribution)
    #         coeffs = tukey_basis.to_eigenbasis(distribution)
    #         params["log_dist"][k] = coeffs

    # if "log_dist" in params.keys():
    #     for k, log_dist in params["log_dist"].items():
    #         distribution = 10**log_dist
    #         distribution /= distribution.sum()
    #         distribution = np.log10(distribution)
    #         params["log_dist"][k] = distribution

    if "log_dist" in params.keys():
        for k, coeffs in params["log_dist"].items():
            dist = tukey_basis.from_eigenbasis(coeffs)
            # dist = np.clip(dist, 0)
            # if dist.min() < 0:
            # dist += np.array([-dist.min(), 0]).max()
            dist = dist / dist.sum()
            coeffs = tukey_basis.to_eigenbasis(dist)
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
    if not exp.calibrator:
        dist = model.get_distribution(exp)
        prior = 10000 * jax.nn.relu(-dist).sum()
        # prior = loLU(-dist).sum()
    else:
        prior = 0
    return likelihood + prior, ()


pscale = lambda model: model.optics.psf_pixel_scale / model.optics.oversample

# %%
n_epoch = 6000

config = {
    "positions": sgd(5e-2, 0),
    "fluxes": sgd(1e-2, 0),
    "aberrations": sgd(5e-1, 5),
    # "log_dist": adam(5e-2, 10),
    "log_dist": adam(1e-4, 15, (100, 2)),
    "spectra": sgd(2e-1, 50),
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
    # batches=fits[0:1],  # Just the science
    # batches=fits[-1:],  # Just the calibrator
    batches=fits,
)

# %%
from dLux import utils as dlu

optics_diameter = 6.603464  # JWST aperture diameter in meters
wavel = 4.3e-6

for exp in fits:

    if exp.calibrator:
        continue
    dist = result.model.get_distribution(exp, rotate=False)
    fig, ax = plt.subplots(figsize=(6, 2.3))
    print(dist.sum(), dist.min(), dist.max())
    c0 = dorito.plotting.plot_result(
        ax,
        dist,
        pixel_scale=model.psf_pixel_scale / model.oversample,
        # cmap="seismic",
        # roll_angle_degrees=-exp.parang,
        # norm=mpl.colors.LogNorm(vmin=1e-5),
        # norm=mpl.colors.CenteredNorm(),
        norm=None,
        # norm=mpl.colors.PowerNorm(1.0, vmin=0),
        diff_lim=dlu.rad2arcsec(wavel / optics_diameter),
        # scale=1.5,
    )

    fig.colorbar(c0)

    ax.set(title=f"Io - {exp.key}")  #   xticks=ticks, yticks=ticks)

    plt.tight_layout()
    plt.savefig(output_path + f"{exp.key}_dist.png", dpi=300)
    plt.close()

amigo.plotting.plot_losses(
    result.losses[0], start=int(n_epoch * 0.75), save_path=output_path
)
amigo.plotting.plot(result.history, save_path=output_path)

for exp in fits:
    exp.print_summary()
    amigo.plotting.summarise_fit(result.model, exp, save_path=output_path)

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
