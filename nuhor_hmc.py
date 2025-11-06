# %%
# jax ecosystem
import jax

jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", True)
print(jax.local_devices()[0].device_kind)

from jax import numpy as np, tree as jtu
import zodiax as zdx
from zodiax.optimisation import sgd, adam
import amigo
import dorito

# other helpful libraries
import numpy
import os
import astropy

# matplotlib ecosystem
import matplotlib.pyplot as plt
import matplotlib as mpl

# import ehtplot
# import scienceplots  # to use matplotlib style "science"

# matplotlib parameters
# plt.style.use(["science", "bright", "no-latex"])

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

source_name = "NUHOR"
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

    file["BADPIX"].data[20:23, 4] = 1
    file["BADPIX"].data[12:15, -5] = 1

    file["BADPIX"].data[:, :3] = 1
    file["BADPIX"].data[:, -3:] = 1
    file["BADPIX"].data[:3, :] = 1
    file["BADPIX"].data[-3:, :] = 1

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

# %%
from dorito.models import ResolvedAmigoModel
from dorito.model_fits import PointResolvedFit
from amigo.model_fits import ModelFit


class ExoZodiFit(PointResolvedFit):
    """
    Model fit for resolved sources. This adds the log distribution parameter.
    """

    def get_key(self, param):

        match param:
            case "stdev":
                return self.filter

        return super().get_key(param)

    def map_param(self, param):

        # Map the appropriate parameter to the correct key
        if param in ["stdev"]:
            return f"{param}.{self.get_key(param)}"

        # Else its global
        return super().map_param(param)

    def initialise_params(self, optics, stdev, contrast):
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
        params["stdev"] = (self.get_key("stdev"), stdev)
        params["contrast"] = self.get_key("contrast"), np.array(contrast)

        return params


class ExoZodiModel(ResolvedAmigoModel):

    size: int = None

    def __init__(
        self,
        exposures,
        optics,
        detector,
        ramp_model,
        read,
        state,
        size,
        rotate=True,
        source_oversample=1,
        param_initers=None,
    ):
        self.size = size
        super().__init__(
            exposures,
            optics,
            detector,
            ramp_model,
            read,
            state,
            rotate,
            source_oversample,
            param_initers,
        )

    @staticmethod
    def gaussian2d(npix, sigma):
        c = (npix - 1) / 2
        x = np.arange(npix)
        gx = jax.scipy.stats.norm.pdf(x, loc=c, scale=sigma)
        g = np.outer(gx, gx)
        return g / np.sum(g)

    def get_distribution(self, exposure, rotate=None, source_id=None):

        stdev = self.params["stdev"][exposure.get_key("stdev")]
        distribution = self.gaussian2d(self.size, stdev)

        return distribution


# %%
load_dict = lambda x: np.load(f"{x}", allow_pickle=True).item()  # helper function

# just two science exposures and one calibrator for this demo
sci_exps = [ExoZodiFit(file) for file in sci_files]
# cal_exps = [amigo.model_fits.PointFit(file) for file in cal_files]
exps = sci_exps  # + cal_exps

# building the model
source_size = 15  # pixels

model = ExoZodiModel(
    size=source_size,
    exposures=exps,
    optics=amigo.optical_models.AMIOptics(),
    detector=amigo.detector_models.LinearDetector(),
    ramp_model=amigo.ramp_models.NonLinearRamp(),
    read=amigo.read_models.ReadModel(),
    state=load_dict(cache + "calibration.npy"),
    param_initers={"contrast": np.array(1e-3), "stdev": np.array(1)},
)

model = model.set("params", load_dict(f"{cache}/aberrations/NUHOR_CAL.npy"))


# %%
import sys

job_name = os.environ.get("SLURM_JOB_NAME")
try:
    job_id = os.environ.get("SLURM_JOB_ID")
    job_idx = "_".join(job_id, job_name)
except:
    job_idx = job_name if job_name is not None else "local"

# batch_idx = int(sys.argv[1]) if len(sys.argv) > 1 else int(0)
output_path = os.path.join(output_path, job_idx) + f"/"#{batch_idx}/"

if not os.path.exists(output_path):
    os.makedirs(output_path)
print(f"Output path: {output_path}")

import shutil

shutil.copy(__file__, output_path + "/script.py")


for exp in exps:
    exp.print_summary()
    amigo.plotting.summarise_fit(model, exp, residuals=False, save_path=output_path)

# %%
import numpyro


def npy_model(model, exposure):
    # sampling params
    stdev = numpyro.sample("sig", numpyro.distributions.HalfNormal(1))  # prior on m
    contrast = numpyro.sample("cont", numpyro.distributions.HalfNormal(1))  # Prior on c

    # updating model with sampled params
    params = model.params
    params["stdev"][exposure.get_key("stdev")] = stdev
    params["contrast"][exposure.get_key("contrast")] = contrast
    model.set("params", params)

    # laying out the components
    data = exposure.slopes.flatten()
    X = exposure(model).flatten()
    err = np.einsum("iixy->ixy", exposure.cov).flatten()  # the diagonal of the cov mats

    with numpyro.plate("data", data.size):
        numpyro.sample("y", numpyro.distributions.Normal(X, err), obs=data)


# numpyro.render_model(npy_model, model_args=(model, exps[0]))

# %%
sampler = numpyro.infer.MCMC(
    numpyro.infer.NUTS(npy_model), num_chains=1, num_samples=1000, num_warmup=100
)
sampler.run(jax.random.PRNGKey(1), model, exps[0])
results = sampler.get_samples()  # Dictionary of MCMC samples

# %%
np.save(output_path + "results.npy", results, allow_pickle=True)

import chainconsumer as cc

C = cc.ChainConsumer()
C.add_chain(cc.Chain.from_numpyro(sampler, name="MCMC Results"))
fig = C.plotter.plot()
plt.savefig(output_path + "corner_plot.png", dpi=300)
plt.show()
