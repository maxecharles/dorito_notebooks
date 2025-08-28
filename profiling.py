# jax ecosystem
import jax

jax.config.update("jax_platform_name", "gpu")
print("Using device:", jax.devices()[0])
print(jax.local_devices()[0].device_kind)
jax.config.update("jax_explain_cache_misses", True)
jax.config.update("jax_enable_x64", True)

from jax import numpy as np
import equinox as eqx

# other helpful libraries
import amigo
import os
from time import time
from socket import gethostname

if gethostname() == "glinton":
    path = "/media/morgana1/snert/max/"
elif gethostname() == "AJQ4YHQH9TX":
    path = "/Volumes/morgana1/snert/max/"
else:
    path = "/fred/oz440/max/"

source_name = "PDS70"
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

load_dict = lambda x: np.load(f"{x}", allow_pickle=True).item()

sci_fits = [amigo.model_fits.PointFit(file, use_cov=True) for file in sci_files[0:1]]
cal_fits = [amigo.model_fits.PointFit(file, use_cov=True) for file in cal_files[0:1]]
fits = sci_fits + cal_fits

# building the model
model = amigo.core_models.AmigoModel(
    exposures=fits,
    optics=amigo.optical_models.AMIOptics(),
    detector=amigo.detector_models.LinearDetector(),
    ramp_model=amigo.ramp_models.NonLinearRamp(),
    read=amigo.read_models.ReadModel(),
    state=load_dict(cache + "calibration.npy"),
)

exp = cal_fits[0]


@eqx.filter_jit
# @eqx.filter_value_and_grad
# @eqx.filter_grad
def model_it(model, exposure):
    print("Compiling model...")
    # return amigo.fitting.loss_fn(model, exposure, {})[0]
    return exposure(model).sum()
    # return exposure.model_psf(model)


print(f"JAX version {jax.__version__}")

start = time()
model_it(model, exp).block_until_ready()
print(f"Model compiled in {time() - start:.4e} seconds")


with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    print("Timing model...")
    for i in range(5):
        start = time()
        model_it(model, exp).block_until_ready()
        end = time()
        print(f"Model {i} took {end - start:.4e} seconds")
