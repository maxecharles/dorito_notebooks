# %%
# jax ecosystem
import jax

jax.config.update("jax_platform_name", "cpu")
# print("Default Backend:", jax.default_backend())
print("Using device:", jax.devices()[0])
jax.config.update("jax_enable_x64", True)
print(jax.local_devices()[0].device_kind)

from jax import numpy as np, tree as jtu
import zodiax as zdx
from zodiax.optimisation import sgd, adam
import amigo
import dorito

# other helpful libraries
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
load_dict = lambda x: np.load(f"{x}", allow_pickle=True).item()

sci_fits = [amigo.model_fits.PointFit(file, use_cov=True) for file in sci_files]
cal_fits = [amigo.model_fits.PointFit(file, use_cov=True) for file in cal_files]

# I only want to use the calibrator in the same primary dither position
# fits = cal_fits[0:1]
fits = sci_fits[0:1] + cal_fits[0:1]
# fits = sci_fits + cal_fits

# building the model
model = amigo.core_models.AmigoModel(
    exposures=fits,
    optics=amigo.optical_models.AMIOptics(),
    detector=amigo.detector_models.LinearDetector(),
    ramp_model=amigo.ramp_models.NonLinearRamp(),
    read=amigo.read_models.ReadModel(),
    state=load_dict(cache + "calibration.npy"),
)

# %%
from time import time


@zdx.filter_jit
def model_it(model, exposure):
    return exposure(model)


# compile the model
print("Compiling model...")

start = time()
model_it(model, fits[0])
print(f"Model compiled in {time() - start:.2f} seconds")

# timing the model
print("Timing model...")
for i in range(10):
    start = time()
    model_it(model, fits[0]).block_until_ready()
    end = time()
    print(f"Model {i} took {end - start:.2f} seconds")

# %%
from amigo.fitting import loss_fn, get_val_grad_fn

loss = get_val_grad_fn(loss_fn)
model_params = amigo.core_models.ModelParams({p: model.get(p) for p in ["aberrations"]})


@zdx.filter_jit
def loss_it(model, exposures):
    print("Compiling loss...")
    x = loss(model_params, model, exposures, {})
    # print(x)
    return x[0][0]


start = time()
loss_it(model, fits)
print(f"Loss function compiled in {time() - start:.2f} seconds")

# timing the model
print("Timing model...")
for i in range(10):
    start = time()
    loss_it(model, fits).block_until_ready()
    end = time()
    print(f"Model {i} took {end - start:.2f} seconds")


# %%
print("Trying big matrix multiplication...")

n = 10000


@zdx.filter_jit
def big_matrix_mult():
    key = jax.random.PRNGKey(0)
    a = jax.random.uniform(key, (n, n))
    b = jax.random.uniform(jax.random.split(key)[1], (n, n))
    return np.dot(a, b)


print("Compiling big matrix multiplication...")
start = time()
big_matrix_mult()
end = time()
print(f"Big matrix multiplication compiled in {end - start:.2f} seconds")


for i in range(10):
    start = time()
    big_matrix_mult().block_until_ready()
    end = time()
    print(f"Big matrix multiplication {i} took {end - start:.2f} seconds")
