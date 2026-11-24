# %%
# jax ecosystem
import jax

# jax.config.update("jax_enable_x64", False)
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_debug_nans", False)

device = jax.local_devices()[0]
print(device.device_kind)

import jax.numpy as np
import jax.tree as jtu
import jax.random as jr

# amigo
import dorito
import amigo
import dLux
import zodiax

# matplotlib ecosystem
import matplotlib.pyplot as plt
import matplotlib as mpl
import ehtplot
import scienceplots

# other
import pandas as pd
import os
import sys

# matplotlib parameters
plt.style.use(["science", "bright", "no-latex"])
plt.rcParams["image.cmap"] = "inferno"
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = "lower"
# plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 8
# plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"


def get_cmap(cmap_name: str):
    cmap = mpl.colormaps[cmap_name]
    cmap.set_bad("k", 0.5)
    return cmap


inferno = get_cmap("inferno")
seismic = get_cmap("seismic")
coolwarm = get_cmap("coolwarm")

load_dict = lambda x: np.load(f"{x}", allow_pickle=True).item()  # helper function

if jax.config.read("jax_enable_x64"):
    print("64bit enabled")
else:
    print("32bit enabled")

import equinox
import lineax

print("jax version:", jax.__version__)
print("equinox version:", equinox.__version__)
print("lineax version", lineax.__version__)
print("zodiax version:", zodiax.__version__)
print("dLux version:", dLux.__version__)
print("amigo version:", amigo.__version__)
print("dorito version:", dorito.__version__)

# %% [markdown]
# # Loading in data

# %%
# Setting data path
from socket import gethostname
from retrain_fns import Tee

print(f"host name: {gethostname()}")

# compile caching
if device.device_kind != "cpu":
    # Persistent XLA compile cache, shared across jobs/nodes via /fred (NOT /tmp, which
    # is per-node and wiped between jobs). The ~16 per-batch loss-fn traces + 1
    # update-fn trace take ~9 min at the start of every run (visible as repeated
    # "Compiling Loss function..." lines); with the same exposures/architecture this
    # is identical work every time, so caching it saves that ~9 min on every run after
    # the first. Biggest relative win for short sweep/diagnostic runs (~7% of a 300
    # epoch run); negligible for a 5000 epoch run.
    cache_path = os.environ.get(
        "RETRAIN_JAX_CACHE", "/fred/oz440/max/data/jax_cache"
    )
    jax.config.update("jax_compilation_cache_dir", cache_path)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update(
        "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
    )

    stats = device.memory_stats()
    free = stats["bytes_limit"] - stats["bytes_in_use"]
    print(f"Total VRAM : {stats['bytes_limit'] / 1024**3:.2f} GB")
    print(f"In use     : {stats['bytes_in_use'] / 1024**3:.2f} GB")
    print(f"Free       : {free / 1024**3:.2f} GB")


if (
    gethostname() == "maxs-mbp-14.shared.sydney.edu.au"
    or gethostname() == "Maxs-MacBook-Pro-14.local"
):
    data_dir = "/Volumes/research-data/PRJ-PAT/max/data/JWST/"
    cache_dir = "/Volumes/research-data/PRJ-PAT/max/data/amigo_cache"
    amigo_files_path = "/Volumes/research-data/PRJ-PAT/max/data/amigo_files/v_0.0.11"
    output_path = "/Users/mc/nt_outputs/local_retrain"

elif gethostname().startswith("max-"):
    data_dir = "/home/dgxuser/max/data/JWST/"
    cache_dir = "/home/dgxuser/max/data/amigo_cache"
    amigo_files_path = "/home/dgxuser/max/data/amigo_files/v_0.0.11"
    output_path = "/home/dgxuser/max/outputs/retrain"

else:
    data_dir = "/fred/oz440/max/data/JWST/"
    cache_dir = "/fred/oz440/max/data/amigo_cache"
    amigo_files_path = "/fred/oz440/max/data/amigo_files/v_0.0.11"
    output_path = "/fred/oz440/max/outputs/retrain"

# cleaning directory of empty folders
for d in os.listdir(output_path):
    this_dir = os.path.join(output_path, d)
    fs = os.listdir(this_dir)
    if len(fs) == 0:
        os.rmdir(this_dir)


# dealing with saving figures when running in script
def check_script():
    if "__file__" in globals():
        print("Running as a script")
        return True
    else:
        print("Running in a notebook or interactive shell")
        return False


# Quick toggle flags
epochs = int(os.environ.get("RETRAIN_EPOCHS", 1000))
# Index into retrain_fns.nn_setup_options (0 = the paper's 16 wide x 3 hidden layers).
nn_config = int(os.environ.get("RETRAIN_NN_CONFIG", 0))
# nn_config = 2
# One intermediate summary (plots, states, residual statistics), half way through
intermediate_prints = [epochs // 2]
# intermediate_prints = [200, 5000]
# intermediate_prints = [25]
# intermediate_prints = list(range(1000, 19000, 2000))
n_files_per_filt = 10  # max out at 10
n_vals_per_filt = 5  # max out at 5
one_filt_flag = False
cal_flag = True
# RETRAIN_BINARY=0 leaves the GO 8330 binary out (calibrators = CAL 4481 only, as in the AMIGO paper).
# Its parameters in the loaded state are carried into the saved states (carry_params below).
binary_flag = os.environ.get("RETRAIN_BINARY", "1") == "1"
val_flag = os.environ.get("RETRAIN_VAL_FLAG", "1") == "1"
flat_flag = True
dark_flag = True
pos1_only = False
static_optics = False
plot_flag = check_script()
# Initial-fit plots for every exposure; RETRAIN_INIT_PLOTS=0 skips them (faster setup).
plot_flag = os.environ.get("RETRAIN_INIT_PLOTS", "1") == "1"
save_flag = check_script()
# RETRAIN_RECALCULATE=1 recomputes every Fisher the run asks for (frozen parameters are skipped) and
# overwrites them in the cache: point RETRAIN_FISHER_CACHE at a fresh folder (path containing
# linear_model_v2) and freeze nothing for a full periodic recalculation.
recalculate_flag = os.environ.get("RETRAIN_RECALCULATE", "0") == "1"
# Milo's pupil-wide eigenbasis (amigo `eigenbasis` branch) instead of per-hole Zernikes. Starts from the
# eigen_ab_fit.ipynb fit of final_statev46 (run that with SAVE = True first). RETRAIN_EIGEN=1 turns it on.
eigen_flag = os.environ.get("RETRAIN_EIGEN", "0") == "1"
eigen_path = os.environ.get(
    "RETRAIN_EIGEN_PATH",
    # 300 modes, matching final_statev48's aberrations shape (300,) -- eigenbasis_v1_windows.npy
    # only has 100 and will crash in eval_small_basis with a shape mismatch against that state.
    amigo_files_path + "/eigenbasis_300_windows.npy",
)
# RETRAIN_N_EIGEN=<n> fits only the first n modes of the eigenbasis file (default: all of them).
n_eigen = int(os.environ["RETRAIN_N_EIGEN"]) if "RETRAIN_N_EIGEN" in os.environ else None
# RETRAIN_FOURIER_BASIS=<file.npz> (needs amigo's fourier-wf branch, AMIGO_SRC=.../amigo-fourierwf/src)
# uses the information-ranked per-hole Fourier modes of build_fourier_basis.py instead of the
# eigenbasis, its first RETRAIN_N_EIGEN modes. Like the eigenbasis, one coefficient vector per key;
# unlike it, the modes fill each hole's whole window, so distortion and primary_beam can move the
# apertures inside them. convert_state_fourier.py converts an eigenbasis state.
fourier_basis_path = os.environ.get("RETRAIN_FOURIER_BASIS")
fourier_flag = fourier_basis_path is not None
if fourier_flag and eigen_flag:
    raise ValueError("RETRAIN_FOURIER_BASIS replaces the eigenbasis: unset RETRAIN_EIGEN")
if fourier_flag:
    print(f"Fourier wavefront basis: {fourier_basis_path}, {n_eigen or 'all'} modes")
# amigo's pipeline-branch optics (Louis's aperture and opd.py: worktree amigo-optics, branch optics-0.1, with the
# zodiax-0.6 venv) are used whenever that amigo is the one imported. Its aperture replaces the global
# `distortion` of the hole centres and the per-hole `primary_beam` warp with relative hole positions
# ("hole_positions") and a per-hole polynomial warp ("hole_distortion", Louis's affine one at order 1), the
# Fourier basis becomes one of its OPD components, and the wavefront is applied with the opposite sign.
# convert_state_metrology.py converts a state (and sets the warp's order).
import importlib.util

pipeline_optics = importlib.util.find_spec("amigo.opd") is not None
aperture_params = ["hole_positions", "hole_distortion"] if pipeline_optics else ["distortion", "primary_beam"]
if pipeline_optics:
    print("amigo pipeline-branch optics: hole positions and per-hole warps")
# RETRAIN_WF=hierarchical (needs amigo's hier-wf branch, e.g. PYTHONPATH=.../amigo-hierwf/src, and the
# eigenbasis) replaces one eigenbasis wavefront per key with a hierarchy, as in Louis's v0.10 work
# (2026-10-19): one shared set of eigenmodes for all data, fitted by the calibrators; per-hole
# piston/tip/tilt per filter (static, the calibrators); and per visit (each visit's own exposures,
# validators too), 17 free terms each once global piston, tip, tilt and defocus are removed.
# convert_state_hierarchical.py turns a per-key state into one.
hier_wf = os.environ.get("RETRAIN_WF", "") == "hierarchical"
if hier_wf:
    print("Hierarchical wavefronts: shared eigenmodes, per-filter and per-visit low-order terms")
# RETRAIN_BASE_OPDS=<file.npz> (pipeline-branch optics; convert_state_base_opds.py builds the file and converts a
# state) replaces the per-key Fourier basis with Louis's three wavefront parts (amigo opd.py): a Fourier set
# ("fourier_coeffs") and a Zernike set per filter ("zernike_coeffs"), both fitted by the calibrators, and every hole's
# piston, tip and tilt per key ("ptt_coeffs", 17 terms, the only part validators fit). States saved before amigo's
# 2026-11 rename (fourier_opd, filter_opd, aberrations) are renamed when loaded. RETRAIN_FOURIER_KEY:
#   "shared" (default)  one Fourier set for everything
#   "program"           one per calibrator program; any other program uses the calibrator program observed
#                       closest in time
#   "program_position"  one per calibrator program and primary dither position (2026-11-11: 8330 POS2 fits
#                       0.2-0.3 worse than POS1 with one set per program); other programs use the closest-dated
#                       calibrator program that has their position
#   "program_filter"    one per calibrator program and filter (2026-11-13: does the high-order wavefront want to
#                       differ by filter as well as program?); other programs use the closest-dated calibrator
#                       program's set for their filter
# RETRAIN_FILTER_OPD_KEY: "filter" (default) one per-filter Zernike set; "position" one per filter and primary
# dither position (Louis, 2026-11-11); "program" one per filter and calibrator program (2026-11-12: with a per-program
# Fourier set, the wavefront then shares nothing between calibrator programs), other programs using the closest-dated
# calibrator program's, as for the Fourier set. Missing keys start from the state's per-filter / per-program values.
base_opds_path = os.environ.get("RETRAIN_BASE_OPDS")
base_opds_flag = base_opds_path is not None
# The piston/tip/tilt parameter: "ptt_coeffs" with the base-model parts (amigo's name since the 2026-11 rename),
# "aberrations" with the per-hole Zernike, eigen and Fourier bases
PTT = "ptt_coeffs" if base_opds_flag else "aberrations"
fourier_key = os.environ.get("RETRAIN_FOURIER_KEY", "shared")
if base_opds_flag and not pipeline_optics:
    raise ValueError("RETRAIN_BASE_OPDS needs the pipeline-branch optics (amigo-optics)")
if base_opds_flag and (fourier_flag or eigen_flag or hier_wf):
    raise ValueError("RETRAIN_BASE_OPDS replaces the other wavefront bases: unset RETRAIN_FOURIER_BASIS / EIGEN / WF")
if fourier_key not in ("shared", "program", "program_position", "program_filter"):
    raise ValueError(f"RETRAIN_FOURIER_KEY={fourier_key}: use shared, program, program_position or program_filter")
filter_opd_key = os.environ.get("RETRAIN_FILTER_OPD_KEY", "filter")
if filter_opd_key not in ("filter", "position", "program"):
    raise ValueError(f"RETRAIN_FILTER_OPD_KEY={filter_opd_key}: use filter, position or program")
# How the piston/tip/tilt ("aberrations") is shared, base-model parts only (2026-11-13). ptt_decomposition.py found
# the calibrators' PTT to be one part per visit (program x POS) plus one per filter, the same in every visit, with
# 1.1 nm left over per key. RETRAIN_PTT_KEY, the calibrators': "filter" (default) one per program, filter and POS;
# "visit" one per program and POS, shared across filters. RETRAIN_VAL_PTT_KEY, the validators': "exposure"
# (default) one each; "visit" one per program, observation and POS, shared across filters (as a science calibrator
# would be fitted downstream). RETRAIN_FILTER_PTT=1 adds a per-filter PTT term ("filter_ptt", a second copy of the
# PTT modes) that the calibrators fit and every exposure in that filter reads; RETRAIN_FILTER_PTT_REF's filter
# (default F430M) is held at zero, because a part common to all filters is the visit term's.
#   2026-11-17: RETRAIN_VAL_PTT_KEY=visit_filter, one per program, observation, filter and POS; RETRAIN_FILTER_PTT=fixed
#   reads the term from the state and never fits it (the plan for the pipeline: the base model's per-filter term held,
#   PTT per visit fitted), so it needs no particular calibrator key.
# RETRAIN_HOLE_ZERNIKE=1 (2026-11-17, "shouldn't the per-filter Zernikes be per hole?"): a per-filter per-hole term
# ("hole_zernike") on the parts file's hole_zernike_planes / hole_zernike_M (make_hole_zernike_parts.py: every hole's Noll
# 1-10 with hole PTT, global defocus and the filter shapes projected out, 42 modes). It overlaps the Fourier set, so
# RETRAIN_HOLE_ZERNIKE_REF's filter (default F430M) is held at zero: that filter's per-hole shape is the Fourier set's.
ptt_key = os.environ.get("RETRAIN_PTT_KEY", "filter")
val_ptt_key = os.environ.get("RETRAIN_VAL_PTT_KEY", "exposure")
filter_ptt_mode = os.environ.get("RETRAIN_FILTER_PTT", "0")
if filter_ptt_mode not in ("0", "1", "fixed"):
    raise ValueError("RETRAIN_FILTER_PTT is 0, 1 (fitted by the calibrators) or fixed (read from the state)")
filter_ptt_flag = filter_ptt_mode != "0"
filter_ptt_ref = os.environ.get("RETRAIN_FILTER_PTT_REF", "F430M")
hole_zernike_flag = os.environ.get("RETRAIN_HOLE_ZERNIKE", "0") == "1"
hole_zernike_ref = os.environ.get("RETRAIN_HOLE_ZERNIKE_REF", "F430M")
if filter_ptt_flag or hole_zernike_flag:
    raise NotImplementedError("RETRAIN_FILTER_PTT / RETRAIN_HOLE_ZERNIKE need the filter_ptt / hole_zernike hooks of "
                              "amigo-optics-mgpu's model_fits, which retrain's amigo does not have")
if ptt_key not in ("filter", "visit") or val_ptt_key not in ("exposure", "visit", "visit_filter"):
    raise ValueError("RETRAIN_PTT_KEY is filter or visit; RETRAIN_VAL_PTT_KEY is exposure, visit or visit_filter")
if (filter_ptt_flag or hole_zernike_flag or ptt_key != "filter" or val_ptt_key != "exposure") and not base_opds_flag:
    raise ValueError("RETRAIN_PTT_KEY / RETRAIN_VAL_PTT_KEY / RETRAIN_FILTER_PTT / RETRAIN_HOLE_ZERNIKE need RETRAIN_BASE_OPDS")
if filter_ptt_mode == "1" and ptt_key != "visit":
    raise ValueError("RETRAIN_FILTER_PTT=1 needs RETRAIN_PTT_KEY=visit: a per-filter calibrator PTT would absorb it")
# fitted wavefront parts; wf_fixed: read from the state and held
wf_params = (["fourier_coeffs", "zernike_coeffs"] + (["filter_ptt"] if filter_ptt_mode == "1" else [])
             + (["hole_zernike"] if hole_zernike_flag else [])) if base_opds_flag else []
wf_fixed = ["filter_ptt"] if filter_ptt_mode == "fixed" else []
if base_opds_flag:
    print(f"Base-model wavefront parts: {base_opds_path}, Fourier set {fourier_key}")
    print(f"PTT: calibrators per {'visit' if ptt_key == 'visit' else 'program, filter and POS'}, validators per"
          f" {val_ptt_key}" + (f", plus a per-filter term ({filter_ptt_ref} held at zero)" if filter_ptt_mode == "1" else "")
          + (", plus the state's per-filter term, held" if filter_ptt_mode == "fixed" else ""))
    if hole_zernike_flag:
        print(f"Per-filter per-hole Zernike term ({hole_zernike_ref} held at zero)")
# RETRAIN_AMP_ORDERS=<n> (2026-11-17, is the per-filter edge wavefront a misfit of the holes themselves?): each hole's
# transmission gets Zernikes of radial orders 0..n-1 ("hole_amplitude", the aperture's amp_coeffs: 1 + sum), one set
# for all data and filters, fitted by the calibrators like the hole positions and warps.
amp_orders = int(os.environ["RETRAIN_AMP_ORDERS"]) if os.environ.get("RETRAIN_AMP_ORDERS") else None
if amp_orders is not None:
    if not pipeline_optics:
        raise ValueError("RETRAIN_AMP_ORDERS needs the pipeline-branch optics")
    aperture_params = aperture_params + ["hole_amplitude"]
    print(f"Hole amplitude: Zernike radial orders 0-{amp_orders - 1} per hole, shared by all data")
best_state = load_dict(
    os.environ.get(
        "RETRAIN_BEST_STATE",
        # final_statev48 = 26-09-28_22-01-02_042277_eigen300_v47_5000's final_state (not
        # best_state -- that run predates the best_batch epoch-averaging fix, so its
        # best_state.npy isn't trustworthy). It's an EIGENBASIS state (300 eigenmode
        # coefficients), so RETRAIN_EIGEN=1 is required or the aberrations shape won't
        # match the model's (crashes with a broadcast error in eval_small_basis).
        # RETRAIN_RESET_FF_NL resets FF/non_linearity back to calibration.npy's values
        # (see the "Populate from state" section) since that run converged on a
        # checkerboard artifact in those two.
        amigo_files_path + "/final_statev48.npy",
    )
)
reset_ff_nl_flag = os.environ.get("RETRAIN_RESET_FF_NL", "0") == "1"
# States saved before amigo's 2026-11 rename name the base-model parts fourier_opd / filter_opd and the PTT aberrations
OLD_WF_NAMES = {"fourier_opd": "fourier_coeffs", "filter_opd": "zernike_coeffs", "aberrations": "ptt_coeffs"}


def migrate_wf_names(state):
    if base_opds_flag and "fourier_opd" in state:
        for old, new in OLD_WF_NAMES.items():
            if old in state:
                state[new] = state.pop(old)
    return state


# every value as a jax array (with its own dtype), however the state was saved (amigo_training's are plain numpy)
best_state = migrate_wf_names(jax.tree.map(np.asarray, best_state))

# RETRAIN_AB_STATE takes only the aberrations from another state file (e.g. the same
# wavefront projected onto a different basis by project_aberrations.py), so a basis
# comparison shares every other parameter with best_state. The basis flags (RETRAIN_EIGEN,
# RETRAIN_EIGEN_PATH) must match that file's aberration shapes.
ab_state_path = os.environ.get("RETRAIN_AB_STATE")
if ab_state_path is not None:
    best_state[PTT] = migrate_wf_names(load_dict(ab_state_path))[PTT]
    print(f"Aberrations taken from {ab_state_path}")

# RETRAIN_FLAT_AB=1 ignores best_state's aberrations, so every wavefront starts flat (zero
# coefficients). Needed for a basis the state was never fitted in (its coefficients would
# mean nothing, or not even have the right shape).
flat_ab_flag = os.environ.get("RETRAIN_FLAT_AB", "0") == "1"
if flat_ab_flag:
    best_state.pop(PTT, None)
    print("Aberrations start flat")
if pipeline_optics and "hole_positions" not in best_state:
    raise ValueError("the pipeline-branch optics need a state converted by convert_state_metrology.py")

# Staged training: RETRAIN_FREEZE_NN=1 holds nn_weights fixed (optics-only phase, e.g.
# comparing OPD bases without the NN co-adapting differently to each one),
# RETRAIN_FREEZE_DETECTOR=1 holds the whole detector fixed (nn_weights and the pixel maps), and
# RETRAIN_FREEZE_OPTICS=1 holds the shared optics fixed (NN-only phase). Frozen params are
# left out of the optimisers and the Fisher calculations, so they get no updates at all.
freeze_detector = os.environ.get("RETRAIN_FREEZE_DETECTOR", "0") == "1"
freeze_nn = os.environ.get("RETRAIN_FREEZE_NN", "0") == "1" or freeze_detector
freeze_optics = os.environ.get("RETRAIN_FREEZE_OPTICS", "0") == "1"
optics_params = [PTT, *wf_params, "defocus", *aperture_params, "sigma"]
detector_params = ["FF", "non_linearity", "dark_current"]
# RETRAIN_FREEZE=<list> holds any other parameters fixed as well, e.g. "distortion:primary_beam"
# (64 of the 65 combinations the calibrators cannot pin down) or "FF:non_linearity:dark_current"
# (the pixel maps, which only the flats and darks train). Colons or commas both separate names
# (sbatch --export splits on commas, so use colons there).
frozen_params = (
    (["nn_weights"] if freeze_nn else [])
    + (detector_params if freeze_detector else [])
    + (optics_params if freeze_optics else [])
)
frozen_params += [
    p for p in os.environ.get("RETRAIN_FREEZE", "").replace(":", ",").split(",") if p and p not in frozen_params
]
if frozen_params:
    print(f"Frozen params: {frozen_params}")


def unfrozen(params):
    return [p for p in params if p not in frozen_params]


# RETRAIN_NN_OPT=adam trains nn_weights with Adam (as in the AMIGO paper) instead of SGD. Adam
# divides out gradient scale, so the SGD path's warm-up and temperature (both applied to the
# raw gradient in grads_fn) would be cancelled; in Adam mode both live in the optimiser
# instead (retrain_fns.tempered_adam), and the temperature is in units of Adam's normalised
# step, so T0/TF default to 0 there rather than to the SGD values.
nn_adam = os.environ.get("RETRAIN_NN_OPT", "sgd") == "adam"

# Diagnostic: let calibrator/binary exposures' likelihood contribute real gradients
# to FF/non_linearity too, instead of only ever getting them from flats (the
# current default). See amigo.model_fits.ALLOW_CAL_PIXEL_GRADS's docstring for the
# entanglement-with-nn_weights concern this is specifically testing for.
if os.environ.get("RETRAIN_ALLOW_CAL_PIXEL_GRADS") == "1":
    import amigo.model_fits

    amigo.model_fits.ALLOW_CAL_PIXEL_GRADS = True
    print("Allowing calibrator/binary exposures to update FF/non_linearity")
# best_state = load_dict(amigo_files_path + "/scratch_bfgs_final_state.npy")
# secondary_state = load_dict(amigo_files_path + "/scratch_bfgs_final_state.npy")
# best_state = load_dict(amigo_files_path + "/final_statev27.npy")
# best_state = load_dict(amigo_files_path + "/best_statev26.npy")
best_weights = None
# RETRAIN_INIT=distill starts the network from the distilled weights instead of the
# nn_weights stored in best_state (final_statev45).
if os.environ.get("RETRAIN_INIT") == "distill":
    import numpy as onp_

    best_weights = np.array(
        onp_.load(amigo_files_path + f"/scratch_distill_{nn_config}.npy"), dtype=float
    )
    print(f"Initialising nn_weights from distilled weights, config {nn_config}")
# RETRAIN_INIT_WEIGHTS=<path.npy> starts the network from an explicit weights file, e.g. a
# distill.py student (set RETRAIN_NN_CONFIG to that student's architecture).
if os.environ.get("RETRAIN_INIT_WEIGHTS") is not None:
    import numpy as onp_

    best_weights = np.array(onp_.load(os.environ["RETRAIN_INIT_WEIGHTS"]), dtype=float)
    print(f"Initialising nn_weights from {os.environ['RETRAIN_INIT_WEIGHTS']}")


if save_flag:
    try:
        job_index = int(sys.argv[1])
    except:
        pass

    job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if job_id is not None and task_id is not None:
        job_id = f"{job_id}_{task_id}"
    elif job_id is None:
        from datetime import datetime

        form = "%y-%m-%d_%H-%M-%S_%f"
        now = datetime.now()
        job_id = now.strftime(form)

    job_name = os.environ.get("SLURM_JOB_NAME")
    if job_name is not None:
        job_id = "_".join([job_id, job_name])

    print(f"Job ID: {job_id}")
    save_path = os.path.join(output_path, job_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    import shutil

    shutil.copy(__file__, save_path + "/script.py")

    log_path = os.path.join(save_path, "output.txt")

    sys.stdout = Tee(log_path)
    sys.stderr = sys.stdout  # capture errors too

else:
    save_path = None


print(f"Save path: {save_path}")
print(f"amigo_files_path: {amigo_files_path}")

# %%
from retrain_fns import summarise_files
import importlib

resource_path = importlib.resources.files(amigo).joinpath("data/badpix.npy")

with importlib.resources.as_file(resource_path) as file_path:
    badpix = np.array(np.load(file_path), dtype=int)
badpix_bool = badpix.astype(bool)  # use for boolean masking; badpix (int) is kept for FITS writing

# # Add extra bad pixels
# badpix = np.load(f"{cache_dir}/full_badpix.npy")
# # badpix = np.load(f"{cache_dir}/full_badpix.npy")
# badpix = np.array(badpix, dtype=int)


def add_badpix(file):
    file["BADPIX"].data = badpix.copy()
    # [25, 74] was flagged for the darks only; it also flares in GO 8330 (2026-10-23), so masked everywhere
    file["BADPIX"].data[25, 74] = 1
    # if file[0].header["PROGRAM"] == "04481" and file[0].header["FILTER"] == "F430M":
    #     file["BADPIX"].data[30, 40] = 1
    return file

# %% [markdown]
# ## Validation files
# Calibrators from other science programs

# %%
from amigo.misc import get_pos


def sort_files(files, n, use_pos=True):
    """
    use_pos=True (default): partition each filter's matches by AMI dither
    position (POS1-4, from get_pos's source centroid), keeping only POS1/POS2
    -- correct for calibrator/validator/binary exposures, which really were
    taken at one of those on-sky dither positions.

    use_pos=False: skip the position split entirely and keep whatever's
    matched, up to n. Flats and darks are lab exposures taken with no science
    target and no on-sky dither at all, so get_pos's "centroid the brightest
    pixel and match to the nearest POS" is meaningless for them -- a flat's
    lamp-illumination centroid or a dark's read-noise peak can land anywhere,
    including POS3/POS4, which use_pos=True's pos1/pos2-only branches would
    then silently drop. Confirmed via debug prints on a real job: get_files
    found our one dark file every time; sort_files was discarding it because
    get_pos(dark) happened to return "POS4". Flats have only ever "worked" by
    the same lucky coincidence of centroiding near POS1/POS2, not because the
    position concept legitimately applies to them.
    """
    files.sort(
        key=lambda x: (
            x[0].header["PROGRAM"],
            x[0].header["FILTER"],
            x[0].header["PATT_NUM"],
        ),
        reverse=True,
    )
    filts = ["F480M", "F430M", "F380M"]
    these_files = {}

    for filt in filts:
        if one_filt_flag and filt != "F480M":
            these_files[filt] = []
            continue

        matched = [f for f in files if f[0].header["FILTER"] == filt]

        if not use_pos:
            these_files[filt] = matched[0:n]
            continue

        if pos1_only:
            x = [f for f in matched if get_pos(f) == "POS1"]
        else:
            pos1 = [f for f in matched if get_pos(f) == "POS1"]
            pos2 = [f for f in matched if get_pos(f) == "POS2"]
            x = []
            for a, b in zip(pos1, pos2):
                x.extend([a, b])
            x.extend(pos1[len(pos2) :])
            x.extend(pos2[len(pos1) :])

        these_files[filt] = x[0:n]

    files = [f for sub_f in these_files.values() for f in sub_f]
    return files

# %%
# LOADING IN DATA
from astropy.io import fits
from amigo.files import get_files

val_files = []

# Collated validator subset
program_path = os.path.join(data_dir, "VAL_SUBSET/calslope/")
files = get_files(program_path, "nis_calslope", IS_PSF=True)
files = sort_files(files, n_vals_per_filt)
val_files += files

# Adding bad pixels and summarising metadata
val_files = [add_badpix(file) for file in val_files]
summarise_files(val_files)

# %% [markdown]
# ## Calibration files
# The main training data, including flats.

# %%
calpsf_files = []
calbin_files = []

print("CAL4481 - Anand Sivaramakrishnan")
program_path = os.path.join(data_dir, "4481/calslope/")
files = get_files(program_path, "nis_calslope", IS_PSF=True)
calpsf_files += files
calpsf_files = [add_badpix(file) for file in calpsf_files]
calpsf_files = sort_files(calpsf_files, n_files_per_filt)
summarise_files(calpsf_files)

print()
print("CAL8330 - Benjamin Pope")
program_path = os.path.join(data_dir, "8330/calslope/")
files = get_files(program_path, "nis_calslope")
calbin_files += files
calbin_files = [file for file in calbin_files if file[0].header["NGROUPS"] > 3]
calbin_files = [add_badpix(file) for file in calbin_files]
calbin_files = sort_files(calbin_files, n_files_per_filt)
summarise_files(calbin_files)

# RETRAIN_EXTRA_CAL=<program>[:<obs>,<obs>...] (2026-11-20; ':' also separates observations, for sbatch --export) adds another program's PSF exposures as calibrators,
# e.g. 1509:002,003 (CAL 1509 "NRM Pixel Linearity": F480M, 50-115 groups, dither positions 1-4). Every dither
# position is kept (sort_files would keep POS1/POS2 only). Pixels with a non-finite slope or covariance, or a
# non-positive variance, are masked as in validate_state.py (HR-7258 has saturated pixels with NaN covariance,
# which add_badpix leaves in and which turn the gradients NaN); with RETRAIN_USE_COV=1 also pixels whose
# covariance is not positive definite. The program gets its own Fourier set like the other calibrators.
extra_cal = os.environ.get("RETRAIN_EXTRA_CAL")
if extra_cal:
    import numpy as onp_

    _prog, _, _obs = extra_cal.partition(":")
    files = get_files(os.path.join(data_dir, f"{int(_prog)}/calslope/"), "nis_calslope", IS_PSF=True)
    if _obs:
        files = [file for file in files if file[0].header["OBSERVTN"] in _obs.replace(":", ",").split(",")]
    if not files:
        raise ValueError(f"RETRAIN_EXTRA_CAL={extra_cal}: no PSF calslope files")
    for file in files:
        _cov = onp_.asarray(file["SLOPE_COV"].data, float)
        _bad = ~(onp_.isfinite(file["SLOPE"].data).all(0) & onp_.isfinite(_cov).all((0, 1))
                 & (onp_.diagonal(_cov, axis1=0, axis2=1) > 0).all(-1))
        if os.environ.get("RETRAIN_USE_COV", "0") == "1" and _cov.shape[0] > 1:
            _C = onp_.transpose(onp_.where(onp_.isfinite(_cov), _cov, 0.0), (2, 3, 0, 1))
            _bad |= onp_.linalg.eigvalsh(_C + onp_.eye(_cov.shape[0]) * _bad[..., None, None])[..., 0] <= 0
        file = add_badpix(file)
        file["BADPIX"].data[_bad] = 1
        calpsf_files.append(file)
    print()
    print(f"Extra calibrators (RETRAIN_EXTRA_CAL={extra_cal})")
    summarise_files(files)

# RETRAIN_NG3_TEST=1 (a test of whether the hole-edge wavefront structure depends on well depth, i.e.
# gain / non-linearity): the 8330 exposures with NGROUPS <= 3 (left out of training above) and, as the
# control, the ones with more groups in the same filters, same visit and same dither positions, are
# added as validators with a wavefront of their own each (UniqueAbBinaryFit), started from their dither
# position's trained wavefront. Only those wavefronts are fitted (grads_fn); freeze everything shared.
ng3_test = os.environ.get("RETRAIN_NG3_TEST", "0") == "1"
ng3_files = []
if ng3_test:
    files = get_files(program_path, "nis_calslope")
    short = [file for file in files if file[0].header["NGROUPS"] <= 3]
    short_filts = {file[0].header["FILTER"] for file in short}
    long = [file for file in files if file[0].header["NGROUPS"] > 3 and file[0].header["FILTER"] in short_filts]
    ng3_files = [add_badpix(file) for file in sort_files(short, 10) + sort_files(long, 10)]
    print()
    print("NGROUPS TEST (8330, as validators)")
    summarise_files(ng3_files)

# %% [markdown]
# Flats.

# %%
flat_files = []
print()
print("FLATS")
program_path = os.path.join(data_dir, "FLATS/calslope/")
files = get_files(program_path, "nis_calslope", EXP_TYPE="NIS_LAMP")
flat_files += files
# flat_files = [file for file in flat_files if file[0].header["NGROUPS"] != 45]
flat_files = sort_files(flat_files, n_files_per_filt, use_pos=False)

flat_files = [add_badpix(file) for file in flat_files]
summarise_files(flat_files)

# %%
dark_files = []
print()
print("DARKS")
program_path = os.path.join(data_dir, "DARKS/calslope/")
files = get_files(program_path, "nis_calslope", EXP_TYPE="NIS_DARK")
dark_files += files
dark_files = [file for file in dark_files if file[0].header["NGROUPS"] != 800]
dark_files = sort_files(dark_files, n_files_per_filt, use_pos=False)

dark_files = [add_badpix(file) for file in dark_files]
summarise_files(dark_files)

# %% [markdown]
# # Exposures

# %%
from amigo.model_fits import PointFit, FlatFit, DarkFit, BinaryFit

# The focus (Fresnel defocus) key, RETRAIN_DEFOCUS_KEY:
#   "position" (default)  one per program, filter and dither position, the same as the calibrators'
#                         wavefront key (program_gradients.py, 2026-10-15: 8330's two dither
#                         positions want focus 8-18 um apart)
#   "program"             one per program and filter
#   "filter"              one per filter, shared by all programs (amigo's own key)
# With "position" or "program" every program fits its own focus, the validators too. The
# eigenbasis is orthogonal to defocus by construction, so with the shared key a program-to-program
# focus change has no parameter of its own: the wavefront modes can only imitate about 96% of it
# (measured at v50: validators 12.95 -> 12.34 with their own focus, split_check.py).
defocus_key = os.environ.get("RETRAIN_DEFOCUS_KEY", "position")
if defocus_key not in ("position", "program", "filter"):
    raise ValueError(f"RETRAIN_DEFOCUS_KEY={defocus_key}: use position, program or filter")
own_defocus = defocus_key in ("position", "program")
print("Defocus keyed by " + {"position": "program, filter and dither position", "program": "program and filter",
                             "filter": "filter"}[defocus_key])


def defocus_group(exp):
    if defocus_key == "position":
        return "_".join((exp.program, exp.filter, exp.POS))
    return "_".join((exp.program, exp.filter))


# RETRAIN_ROTATION_KEY=program fits a small rotation of the PSF for every program, on top of the
# detector's fixed 0.56126717 deg: the NRM sits in the NIRISS pupil wheel, which does not come
# back to exactly the same angle (header PWCPOS: 0.00244 deg steps, 0.3 deg spread over visits).
# RETRAIN_ROTATION_KEY=observation gives every observation its own; RETRAIN_ROTATION_SPLIT lists
# programs (":"-separated) that are keyed by observation while the rest stay per program, e.g.
# 01093, whose validators come from two visits at different wheel positions. amigo applies the
# rotations (ModelFit, about each star so they do not pull on the fitted positions); without
# RETRAIN_ROTATION_KEY they are not fitted and stay at 0.
rotation_key = os.environ.get("RETRAIN_ROTATION_KEY", "")
rotation_split = [p for p in os.environ.get("RETRAIN_ROTATION_SPLIT", "").replace(":", ",").split(",") if p]
fit_rotation = rotation_key in ("program", "observation")
if fit_rotation:
    print(f"Rotation fitted per {rotation_key}" + (f", per observation for {rotation_split}" if rotation_split else ""))


def rotation_group(exp):
    if rotation_key == "observation" or exp.program in rotation_split:
        return "_".join((exp.program, exp.observation))
    return exp.program


# The detector blur (the Gaussian sigma, arcsec: mostly charge diffusion) is keyed per exposure by
# amigo (ModelFit, one per filter by default), which always overrides the detector's own `sigma`,
# so that is frozen. Diffusion depends on how deep in the absorber the photons are absorbed, so on
# wavelength, and with one blur for every filter the network bends to make up the difference
# (program_gradients.py, 2026-10-15: exposures in the same filter pull the network the same way,
# different filters opposite ways). Only the calibrators fit the blurs; validators use the value of
# their key. RETRAIN_SIGMA_KEY:
#   "filter" (default)  one blur per filter
#   "global"            one blur for everything (one shared key, "all")
#   "program"           one per program and filter (RETRAIN_SIGMA_SPLIT: per observation as well for
#                       the listed programs); validators' programs are not calibrators, so their
#                       keys keep the loaded state's global sigma
# Each blur is round; RETRAIN_SIGMA_PAIR=1 makes it a pair, one width along x and one along y
# (sigma_scan.py: wider in y than x in 23 of 24 keys). The blurs start from the loaded state's
# (a pair as its mean when round) or else from its global sigma.
sigma_key = os.environ.get("RETRAIN_SIGMA_KEY", "filter")
if sigma_key not in ("filter", "global", "program"):
    raise ValueError(f"RETRAIN_SIGMA_KEY={sigma_key}: use filter, global or program")
sigma_split = [p for p in os.environ.get("RETRAIN_SIGMA_SPLIT", "").replace(":", ",").split(",") if p]
sigma_pair = os.environ.get("RETRAIN_SIGMA_PAIR", "0") == "1"
print({"filter": "Blur fitted per filter", "global": "One blur for everything",
       "program": "Blur fitted per program and filter"}[sigma_key]
      + (f" (per observation for {sigma_split})" if sigma_split and sigma_key == "program" else "")
      + (", along x and y" if sigma_pair else ", round") + ", by the calibrators")
if "sigma" not in frozen_params:
    frozen_params.append("sigma")


def sigma_group(exp):
    if sigma_key == "global":
        return "all"
    if sigma_key == "filter":
        return exp.filter
    group = "_".join((exp.program, exp.observation)) if exp.program in sigma_split else exp.program
    return "_".join((group, exp.filter))


def fourier_group(exp):
    if fourier_key == "shared":
        return "shared"
    if fourier_key == "program":
        return exp.program if exp.program in fourier_programs else nearest_fourier_program[exp.program]
    if fourier_key == "program_filter":
        program = exp.program if exp.program in fourier_programs else nearest_fourier_program[exp.program]
        return f"{program}_{exp.filter}"
    # program_position: the exposure's own program if it trains one at this position, else the closest-dated
    # calibrator program that has this position
    if (exp.program, exp.POS) in fourier_program_pos:
        return f"{exp.program}_{exp.POS}"
    progs = [p for p, pos in fourier_program_pos if pos == exp.POS]
    return f"{min(progs, key=lambda p: abs(program_dates[p] - program_dates[exp.program]))}_{exp.POS}"


def filter_opd_group(exp):
    if filter_opd_key == "position":
        return f"{exp.filter}_{exp.POS}"
    if filter_opd_key == "program":
        program = exp.program if exp.program in fourier_programs else nearest_fourier_program[exp.program]
        return f"{program}_{exp.filter}"
    return exp.filter


def ptt_group(exp):
    """The calibrators' PTT key (RETRAIN_PTT_KEY)."""
    if ptt_key == "visit":
        return "_".join((exp.program, exp.POS))
    return "_".join((exp.program, exp.filter, exp.POS))


def val_ptt_group(exp):
    """The validators' PTT key (RETRAIN_VAL_PTT_KEY)."""
    if val_ptt_key == "visit":
        return "_".join((exp.program, exp.observation, exp.POS))
    if val_ptt_key == "visit_filter":
        return "_".join((exp.program, exp.observation, exp.filter, exp.POS))
    return "_".join((exp.filter, exp.key))


class UniqueAbPointFit(PointFit):
    def get_key(self, param):
        match param:
            case "fourier_coeffs":
                return fourier_group(self)
            case "zernike_coeffs":
                return filter_opd_group(self)
            case "aberrations" | "ptt_coeffs" if hier_wf:
                return "shared"
            case "aberrations" | "ptt_coeffs":
                return val_ptt_group(self)
            case "defocus" if own_defocus:
                return defocus_group(self)
            case "rotations":
                return rotation_group(self)
            case "sigmas":
                return sigma_group(self)
        return super().get_key(param)


# One wavefront per program, filter and dither position. 04481 is all POS1, so for the
# calibrators this only renames the key, matching the _POS1 keys trained in final_statev48.
class PosPointFit(PointFit):
    def get_key(self, param):
        match param:
            case "fourier_coeffs":
                return fourier_group(self)
            case "zernike_coeffs":
                return filter_opd_group(self)
            case "aberrations" | "ptt_coeffs" if hier_wf:
                return "shared"
            case "aberrations" | "ptt_coeffs":
                return ptt_group(self)
            case "defocus" if own_defocus:
                return defocus_group(self)
            case "rotations":
                return rotation_group(self)
            case "sigmas":
                return sigma_group(self)
        return super().get_key(param)


class PosBinaryFit(BinaryFit):
    def get_key(self, param):
        match param:
            case "fourier_coeffs":
                return fourier_group(self)
            case "zernike_coeffs":
                return filter_opd_group(self)
            case "aberrations" | "ptt_coeffs" if hier_wf:
                return "shared"
            case "aberrations" | "ptt_coeffs":
                return ptt_group(self)
            case "defocus" if own_defocus:
                return defocus_group(self)
            case "rotations":
                return rotation_group(self)
            case "sigmas":
                return sigma_group(self)
        return super().get_key(param)


# A binary with a wavefront of its own per exposure (keyed like the validators'), everything else keyed
# like PosBinaryFit: RETRAIN_NG3_TEST.
class UniqueAbBinaryFit(PosBinaryFit):
    def get_key(self, param):
        if param == PTT and not hier_wf:
            return val_ptt_group(self)
        return super().get_key(param)


# The point-source exposures (calibrators, binaries, validators) use each pixel's full slope covariance by default
# (use_cov=True, 2026-11-12): adjacent slopes share a read, so their read noise is anti-correlated, and the
# diagonal-only likelihood (amigo's default, used for every run before this date) under-weights the mean slope
# (its noise variance is 0.06-0.16x the diagonal's overall, 0.44-0.82x in the PSF core; see
# outputs/checks/whitening). RETRAIN_USE_COV=0 goes back to the diagonal, e.g. to compare with earlier runs; losses
# and Fishers on the two likelihoods are not comparable, so a run that switches needs a fresh RETRAIN_FISHER_CACHE.
# Flats and darks always stay diagonal (the first full-covariance attempt crashed on a flat's near-singular
# SLOPE_COV; the calibrators' covariances are all positive definite).
use_cov = os.environ.get("RETRAIN_USE_COV", "0") == "1"

# RETRAIN_ONE_ON_FS=1 fits amigo's 1/f model on every point-source exposure (calibrators, binaries,
# validators): in every group, a polynomial along each column (offset and slope), the readout's
# column-correlated noise (NIRISS reads along columns). It has no Fisher, so Adam with
# RETRAIN_ONE_ON_FS_LR (DN per step).
one_on_fs_flag = os.environ.get("RETRAIN_ONE_ON_FS", "0") == "1"

# Construct the exposures
psf_kwargs = {"covariance": "full" if use_cov else "diagonal", "fit_one_on_fs": one_on_fs_flag}
# RETRAIN_FOURIER_KEY=program: each calibrator program has a Fourier set; every other program uses the set of
# the calibrator program observed closest in time (median DATE-OBS of each program's files)
import numpy as onp_

_dates = {}
for _file in calpsf_files + calbin_files + val_files + ng3_files:
    _dates.setdefault(_file[0].header["PROGRAM"], []).append(onp_.datetime64(_file[0].header["DATE-OBS"]).astype(float))
program_dates = {prog: onp_.median(days) for prog, days in _dates.items()}
fourier_programs = sorted({f[0].header["PROGRAM"] for f in (calpsf_files if cal_flag else []) + (calbin_files if binary_flag else [])})
from amigo.misc import get_pos as _get_pos

fourier_program_pos = sorted({(f[0].header["PROGRAM"], _get_pos(f)) for f in (calpsf_files if cal_flag else [])
                              + (calbin_files if binary_flag else [])})
nearest_fourier_program = {
    prog: min(fourier_programs, key=lambda cal: abs(program_dates[cal] - date)) for prog, date in program_dates.items()
} if fourier_programs else {}
if base_opds_flag and fourier_key != "shared":
    print("Fourier set per calibrator program", fourier_programs, "; others use",
          {p: c for p, c in nearest_fourier_program.items() if p not in fourier_programs})

calpsf_exposures = [PosPointFit(file, **psf_kwargs) for file in calpsf_files] if cal_flag else []
calbin_exposures = [PosBinaryFit(file, **psf_kwargs) for file in calbin_files] if binary_flag else []
val_exposures = [UniqueAbPointFit(file, **psf_kwargs) for file in val_files] if val_flag else []
ng3_exposures = [UniqueAbBinaryFit(file, **psf_kwargs) for file in ng3_files] if val_flag else []
val_exposures += ng3_exposures
# RETRAIN_FLAT_NN=0 models the flats without the NN's charge migration (as the darks already
# are). The flats only see FF times whatever pattern the NN makes, so with the NN in them FF is
# whatever cancels the NN (see flat_diag.py); without it, FF and non_linearity are set by the
# flats alone and no longer depend on the NN weights.
flat_nn_flag = os.environ.get("RETRAIN_FLAT_NN", "1") == "1"


class NoMigrationFlatFit(FlatFit):
    def model_ramp(self, illuminance, model):
        from amigo.ramp_models import Ramp

        bias = self.calc_bias(illuminance, model)
        no_bleed = model.ramp_model.set("bleed", False)
        ramp = no_bleed.evolve_illuminance(illuminance.data, bias, self.ngroups)
        return Ramp(ramp, illuminance.pixel_scale)


if not flat_nn_flag:
    print("Flats modelled without the NN")
flat_class = FlatFit if flat_nn_flag else NoMigrationFlatFit
flat_exposures = [flat_class(file, covariance="diagonal") for file in flat_files] if flat_flag else []
dark_exposures = [DarkFit(file, covariance="diagonal") for file in dark_files] if dark_flag else []

cal_exposures = calpsf_exposures + calbin_exposures
exposures = cal_exposures + val_exposures + flat_exposures + dark_exposures
print(f"Total number of exposures: {len(exposures)}")

# just closing the files
files = calpsf_files + calbin_files + val_files + ng3_files + flat_files + dark_files
_ = [f.close() for f in files]

# %% [markdown]
# Finally, let's batch the exposures for training.

# %%
from amigo.legacy.fitting import batch_exposures

bs = 3
# bs = 1 if one_filt_flag else 3

# Get the batches: Calibrator, flats, and validators
cal_batch = batch_exposures(cal_exposures, batch_size=bs, key="cal")
flat_batch = batch_exposures(flat_exposures + dark_exposures, batch_size=bs, key="flat")
val_batch = batch_exposures(val_exposures, batch_size=bs, key="val")
nn_batches = {**cal_batch, **flat_batch}

# %% [markdown]
# # Building model

# %%
from amigo.core_models import AmigoModel
from amigo.optical_models import AMIOptics
from amigo.detector_models import LinearDetector
from amigo.ramp_models import NonLinearRamp
from amigo.read_models import ReadModel
from retrain_fns import nn_setup_options

nn_arch = nn_setup_options[nn_config]
# nn_arch = nn_setup_options[2]
# nn_arch = nn_setup_options[job_index]

# fourier=True is another basis option: an SVD-orthogonalised Fourier dictionary
# instead of plain Zernikes (only orthogonal over a circle, not over the hexagonal holes) --
# exact for whatever aperture shape the mask describes, at the cost of the modes
# no longer being simple named aberrations. n_modes_keep=28 matches
# radial_orders=7's Noll mode count, for a fair comparison against the same amount
# of aberration freedom. See amigo.optical_models.orthogonalise_fourier_basis's
# docstring for the full rationale, including a real ~140x scale-matching fix
# against Zernike's RMS~1 convention that's easy to miss if reimplementing this.
if pipeline_optics:
    import numpy as onp_
    from amigo.optical_models import AffineDistortion, DynamicApertureMask, PolynomialDistortion

    if not (fourier_flag or base_opds_flag) or static_optics:
        raise ValueError("the pipeline-branch optics take RETRAIN_FOURIER_BASIS or RETRAIN_BASE_OPDS, on a dynamic aperture")
    # the warp's order follows the state's: 2 terms per axis at order 1 (affine), 5 at order 2, 9 at order 3
    hole_order = {2: 1, 5: 2, 9: 3}[np.shape(best_state["hole_distortion"])[-1]]
    if base_opds_flag:
        from amigo.opd import load_opds

        # each part's coefficients are fitted under its alias, named as opd.standard_opds names them
        opds = load_opds(base_opds_path)
        opd_alias = {f"{name}_coeffs": f"opds.{name}.x.x" for name in opds}
        if filter_ptt_flag:
            # the per-filter PTT term, summed with the visit PTT: a second copy of the PTT component, or, when the
            # parts file carries filter_ptt_M (make_ptt_states.py's B), only those modes of the holes' planes
            with onp_.load(base_opds_path, allow_pickle=False) as _parts:
                _filter_ptt_M = onp_.asarray(_parts["filter_ptt_M"]) if "filter_ptt_M" in _parts.files else None
            if _filter_ptt_M is None:
                opds["filter_ptt"] = opds["ptt"]
            else:
                from amigo.opd import BasisOPD

                opds["filter_ptt"] = BasisOPD(opds["ptt"].basis, x=zodiax.Map(
                    x=np.zeros(len(_filter_ptt_M)), M=np.asarray(_filter_ptt_M), s=np.array(1e-9)))
            opd_alias["filter_ptt_coeffs"] = "opds.filter_ptt.x.x"
            print(f"Per-filter PTT term: {len(opds['filter_ptt'].x.M)} modes")
        if hole_zernike_flag:
            from amigo.opd import BasisOPD

            with onp_.load(base_opds_path, allow_pickle=False) as _parts:
                if "hole_zernike_M" not in _parts.files:
                    raise ValueError(f"RETRAIN_HOLE_ZERNIKE: {base_opds_path} has no hole_zernike_planes / hole_zernike_M")
                _fh_planes, _fh_M = onp_.asarray(_parts["hole_zernike_planes"]), onp_.asarray(_parts["hole_zernike_M"])
            opds["hole_zernike"] = BasisOPD(np.asarray(_fh_planes), x=zodiax.Map(
                x=np.zeros(len(_fh_M)), M=np.asarray(_fh_M), s=np.array(1e-9)))
            opd_alias["hole_zernike_coeffs"] = "opds.hole_zernike.x.x"
            print(f"Per-filter per-hole Zernike term: {len(_fh_M)} modes")
    else:
        from amigo.fourier_basis import FourierBasis, FourierBasisOPD

        fourier_basis = FourierBasis.load(fourier_basis_path, n_eigen)
        opds, opd_alias = {"wavefront": FourierBasisOPD(fourier_basis)}, {PTT: "opds.wavefront.x"}
    pupil_mask = DynamicApertureMask(
        distortion=AffineDistortion() if hole_order == 1 else PolynomialDistortion(order=hole_order),
        opds=opds,
        alias={**opd_alias, "hole_positions": "holes.x", "hole_distortion": "distortion.coefficients",
               **({"hole_amplitude": "amp_coeffs"} if amp_orders is not None else {})},
        **({"amplitude_orders": amp_orders} if amp_orders is not None else {}),
    )
    if not base_opds_flag and not onp_.array_equal(onp_.asarray(fourier_basis.corners), onp_.asarray(pupil_mask.corners)):
        raise ValueError(f"{fourier_basis_path} was built on other hole windows than this aperture's")
    # the holes' common displacement is held fixed; the fitted positions are relative to it
    pupil_mask = pupil_mask.add("holes.b", np.asarray(best_state["hole_shift"]))
    # every wavelength propagated at once (amigo's default of 1 runs them one after another)
    optics = AMIOptics(pupil_mask=pupil_mask, sparse=True, psf_interpolation="intensity", wavelength_batch_size=1024)
    print(f"Hole warp of order {hole_order}")
else:
    optics = AMIOptics(
        # radial_orders=4,
        radial_orders=7,
        static=static_optics,
        sparse=True,
        fourier=os.environ.get("RETRAIN_FOURIER", "0") == "1",
        n_modes_keep=int(os.environ.get("RETRAIN_N_MODES_KEEP", 28)),
        **({"eigen_basis": eigen_path, "n_eigen": n_eigen} if eigen_flag else {}),  # replaces the Zernike aberration basis (only passed when on)
        **({"fourier_basis": fourier_basis_path, "n_eigen": n_eigen} if fourier_flag else {}),
        **({"low_order": True} if hier_wf else {}),  # per-hole low-order terms after the eigenmodes (RETRAIN_WF)
    )

# RETRAIN_NN_INPUT picks what the NN is shown as "charge" when it predicts charge migration:
#   "full" (default)  the first read (x gain) plus the charge accumulated since, as amigo does.
#                     The first read is mostly electronics: a +-690 DN (flats) / +-940 DN (science)
#                     odd/even-row offset and ~800 DN rms of other fixed pattern, so the NN reacts
#                     to that pattern as if it were charge (see flat_diag.py).
#   "no_structure"    the same with the first read's pixel-to-pixel structure removed (its median
#                     level is kept), so the NN only sees structure in the accumulated charge.
#   "signal"          only the charge accumulated since the start of the ramp.
# Only the NN's input changes: the charge that is evolved, and what the non-linearity sees, do not.
nn_input = os.environ.get("RETRAIN_NN_INPUT", "full")


class BlindRamp(NonLinearRamp):
    def build_ramp(self, illuminance, charge):
        if not self.bleed or nn_input == "full":
            return super().build_ramp(illuminance, charge)
        from jax.lax import scan
        from amigo.ramp_models import apply_kernels_stride
        import dLux.utils as dlu

        illuminance = illuminance / self.time_steps
        hidden = charge if nn_input == "signal" else charge - np.median(charge)

        def step(carry, _):
            new_charge = carry + apply_kernels_stride(illuminance, self.kernel_model(carry - hidden))
            return new_charge, new_charge

        _, steps = scan(step, charge, xs=None, length=self.time_steps)
        charges = np.concatenate([charge[None], steps], axis=0)
        return charges * dlu.downsample(self.ff_model.sensitivity, 3)


if nn_input != "full":
    print(f"NN input: {nn_input}")

# RETRAIN_NL_INPUT does the same for the non-linearity polynomial, which amigo applies to the whole
# ramp, first read included:
#   "full" (default)  as amigo does. The first read's row offset then makes an odd/even-row
#                     pattern in the slopes that scales with the offset (+0.28% in the full-frame
#                     flats, +0.35% in SUB80 science frames).
#   "no_structure"    the first read's pixel-to-pixel structure is removed before the polynomial
#                     (its median level is kept), so that pattern can only come from FF.
# The first read's pattern is taken from the model ramp itself, extrapolated back one group.
nl_input = os.environ.get("RETRAIN_NL_INPUT", "full")
read_model = ReadModel(per_pixel_dark=True)
if nl_input != "full":
    from amigo.read_models import PixelNonLinearity

    class BlindNonLinearity(PixelNonLinearity):
        def apply(self, ramp):
            first = 2 * ramp.data[0] - ramp.data[1]
            hidden = jax.lax.stop_gradient(first - np.median(first))
            return super().apply(ramp.add("data", -hidden)).add("data", hidden / self.gain)

    read_model = equinox.tree_at(
        lambda r: r.layers["pixel_non_linearity"], read_model, BlindNonLinearity()
    )
    print(f"Non-linearity input: {nl_input}")

# Get the model
model = AmigoModel(
    exposures=exposures,
    # optics=AMIOptics(),
    optics=optics,
    detector=LinearDetector(),
    ramp_model=(NonLinearRamp if nn_input == "full" else BlindRamp)(**nn_arch),
    read=read_model,
    # state=load_dict(amigo_files_path + "/calibration.npy"),
)
print(nn_arch, model.nn_weights.shape)
print(model.dark_current.shape)

# %% [markdown]
# Populate from state.

# %%
state = load_dict(amigo_files_path + "/calibration.npy")

if reset_ff_nl_flag:
    # best_state's FF/non_linearity converged on a checkerboard pattern (see
    # 26-09-28_22-01-02_042277_eigen300_v47_5000/flats.png) -- reset just those two
    # back to calibration.npy's fitted values before the merge loop below runs, so
    # everything else (aberrations, positions, nn_weights, etc.) still comes from
    # best_state/final_state as normal.
    best_state["FF"] = state["FF"]
    best_state["non_linearity"] = state["non_linearity"]
    print("Reset FF/non_linearity to calibration.npy's values")

# Calibrators and binaries have one wavefront per dither position (PosPointFit/PosBinaryFit).
# A state saved without that split only has the shared "prog_filt" key, so start any missing
# position from it; keys the state already has (e.g. final_statev48's) are left alone.
for exp in [] if flat_ab_flag else cal_exposures:
    key, shared = exp.get_key(PTT), f"{exp.program}_{exp.filter}"
    if key not in best_state[PTT] and shared in best_state[PTT]:
        best_state[PTT][key] = best_state[PTT][shared]
        print(f"Aberrations {key} started from {shared}")

# RETRAIN_NG3_TEST: each exposure's own wavefront starts from its dither position's trained one
for exp in ng3_exposures:
    key, source = exp.get_key(PTT), "_".join((exp.program, exp.filter, exp.POS))
    best_state[PTT][key] = best_state[PTT][source]
    print(f"Aberrations {key} (NGROUPS {exp.ngroups}) started from {source}")

# Focus keys the state does not have start from its program's value in the filter (a state keyed
# by program and filter), else from the filter's (a state keyed by filter); keys the state already
# has are left alone.
defocus_keys = {exp.get_key("defocus"): (exp.filter, "_".join((exp.program, exp.filter))) for exp in cal_exposures + val_exposures}
if own_defocus:
    loaded_defocus = sorted(best_state["defocus"])  # sources come from the state as loaded, not keys added below
    for key, (filt, prog_filt) in defocus_keys.items():
        if key not in best_state["defocus"]:
            if prog_filt in best_state["defocus"] or filt in best_state["defocus"]:
                source = prog_filt if prog_filt in best_state["defocus"] else filt
            else:
                # A state keyed by program, filter and position (e.g. allfree_distill32_1k) given a program it never
                # saw (RETRAIN_EXTRA_CAL): start every position from the closest-dated program's value in the filter
                # at its first position. Position labels are not the same place across programs (01093's POS2 is
                # its far off-centre pointing, -18 um against POS1's -57 um in allfree_distill32_1k)
                prog = key.split("_")[0]
                same_filter = [k for k in loaded_defocus if k.split("_")[1:2] == [filt]]
                nearest = min({k.split("_")[0] for k in same_filter},
                              key=lambda p: abs(program_dates.get(p, onp_.inf) - program_dates[prog]))
                source = next(k for k in same_filter if k.startswith(nearest + "_"))
            best_state["defocus"][key] = best_state["defocus"][source]
            print(f"Defocus {key} started from {source}")

# RETRAIN_FOURIER_KEY=program_position / RETRAIN_FILTER_OPD_KEY=position: keys the state does not have start from
# its per-program / per-filter values (the closest-dated trained program for programs it lacks); afterwards only
# the keys an exposure reads are kept (below), so no unused set gets an optimiser or a Fisher.
wf_keys = {"fourier_coeffs": set(), "zernike_coeffs": set()}
if base_opds_flag:
    for exp in cal_exposures + val_exposures:
        for param, plain in (("fourier_coeffs", exp.program), ("zernike_coeffs", exp.filter)):
            key = exp.get_key(param)
            wf_keys[param].add(key)
            have = best_state.get(param, {})
            if key in have or not have:
                continue
            if plain in have:
                source = plain
            elif param == "fourier_coeffs":
                source = min(have, key=lambda k: abs(program_dates.get(k.split("_")[0], 0) - program_dates[exp.program]))
            else:
                source = next(k for k in have if k.startswith(exp.filter))
            have[key] = have[source]
            print(f"{param} {key} started from {source}")


def update_abs(model, ab_dict):
    ab_shape = list(model.get(PTT).values())[0].shape
    state_ab_shape = list(ab_dict.values())[0].shape

    if ab_shape[1] != state_ab_shape[1]:

        if ab_shape[1] > state_ab_shape[1]:
            y = np
            expand_abs = lambda x: np.concatenate(
                (x, np.zeros((ab_shape[0], ab_shape[1] - state_ab_shape[1]))), axis=1
            )
            ab_dict = jax.tree.map(expand_abs, ab_dict)
        else:
            raise NotImplementedError()
    return model.set(PTT, ab_dict)


if "jitter" in state.keys():
    state["sigma"] = state["jitter"]
    del state["jitter"]

for key, value in state.items():
    if key == "transmission" and not static_optics:
        continue
    # calibration.npy's metrology is retrain's (the pipeline-branch aperture takes the state's instead)
    if key in ["primary_beam", "distortion"] and (static_optics or pipeline_optics):
        continue
    if key in [
        "aberrations",
        # 'primary_beam',
        # 'distortion',
        # 'transmission',
        "badpix",
        "dark_current",
    ]:
        continue
    model = model.set(key, value)

# calibration.npy holds Zernike aberrations, which don't apply to the eigenbasis (its starting
# coefficients come from best_state instead)
if (cal_flag or binary_flag) and not (eigen_flag or fourier_flag or base_opds_flag):
    ab_dict = {}
    for key, _ in model.get(PTT).items():
        # Find the filter token wherever it sits in the key: "prog_filt" (2 parts),
        # "prog_filt_POS1" (3 parts, PosPointFit/PosBinaryFit), or a validator's
        # "filt_program_..." (filter first). Matching against the known filter names
        # directly is robust to all three shapes.
        parts = key.split("_")
        matches = [p for p in parts if p in ("F380M", "F430M", "F480M")]
        filt = matches[0] if matches else "F430M"
        try:
            ab_dict[key] = state["aberrations"][filt]
        except KeyError:
            ab_dict[key] = state["aberrations"]["F430M"]

    model = update_abs(model, ab_dict)

print()
for key, value in best_state.items():
    if key in [
        "positions",
        "flat_coeffs",
        "fluxes",
        "pas",
        "separations",
        "contrasts",
        PTT,
        "nn_weights",
        *aperture_params,
        "FF",
        "non_linearity",
        "sigma",
        "defocus",
        'dark_current',
        "spectra",
        "sigmas",
        "low_filter",
        "low_visit",
        *wf_params,
        *wf_fixed,  # RETRAIN_FILTER_PTT=fixed: the per-filter term, held
    ] + (["rotations"] if fit_rotation else []):  # carried by states fitted with RETRAIN_ROTATION_KEY

        if key == "flat_coeffs" and not flat_flag:
            continue

        if key in ["pas", "separations", "contrasts"] and not binary_flag:
            continue

        if key in model.params.keys():
            # model = model.set(key, value)
            default = model.get(key)
            model = model.set(key, default | value)
        else:
            # pass
            model = model.set(key, value)

# blurs not in the loaded state start from its global sigma (the same along x and y for a pair); a
# loaded pair becomes its mean when round, a loaded round blur both widths of a pair
_s = float(np.mean(np.asarray(model.get("sigma"), float)))
_loaded = best_state.get("sigmas", {})


def _blur(value):
    value = np.asarray(value, float)
    return np.broadcast_to(value, (2,)) if sigma_pair else np.mean(value)


model = model.set("sigmas", {k: _blur(_loaded[k] if k in _loaded else _s) for k in model.sigmas})
print(f"Blurs: {sum(k not in _loaded for k in model.sigmas)} of {len(model.sigmas)} start from the global sigma"
      f" {_s:.5f} arcsec, the others from the state: " + ", ".join(f"{k} {np.round(np.asarray(v), 5)}" for k, v in model.sigmas.items()))

    # if key == "aberrations":
    #     # pass
    #     model = update_abs(model, value)

# for key, val in secondary_state.items():
#     if key == "aberrations":
#         val["08330_F380M"] = val.pop('F380M_08330_002_04_03_5')
#         val["08330_F430M"] = val.pop('F430M_08330_002_06_03_5')
#         val["08330_F480M"] = val.pop('F480M_08330_002_02_03_5')

#     model_param = model.get(key)

#     this_param = {k: val.get(k, v) for k, v in model_param.items()}
#     model = model.set(key, this_param)

for param, keys in wf_keys.items():
    if keys and param in model.params.keys():
        model = model.set(param, {k: v for k, v in model.get(param).items() if k in keys})

if own_defocus:
    # Only the keys an exposure reads: calibration.npy and older states also carry per-filter ones
    model = model.set("defocus", {key: best_state["defocus"][key] for key in defocus_keys})

# initial normalisations
ff = model.FF
ff -= np.median(ff) - 1.0
ff = ff.at[badpix_bool].set(1.0)
model = model.set("FF", ff)

non_lin = model.non_linearity.at[:, badpix_bool].set(0)
model = model.set("non_linearity", non_lin)

# Badpix dark_current pixels used to be reset to the current median every epoch in
# norm_fn (post-update) -- same issue as the validator aberrations: doesn't stop the
# gradient, so a real step was computed+applied every epoch and thrown away, with
# momentum for those pixels never reset by the discard. One-time init here, then
# grads_fn zeroes their gradient every step (below), same fix as aberrations.
dc = model.dark_current.at[badpix_bool].set(np.median(model.dark_current))
model = model.set("dark_current", dc)

# RETRAIN_FREE_CAL_SPECTRA=1 fits the calibrators' spectral slopes like the validators' (default:
# held at the values below). The values then come from the loaded state and are not overwritten.
free_cal_spectra = os.environ.get("RETRAIN_FREE_CAL_SPECTRA", "0") == "1"

if cal_flag and not free_cal_spectra:
    spec_dic = model.spectra
    spec_dic["HD-41094_F380M"] = np.array(-0.12454516)
    spec_dic["HD-41094_F430M"] = np.array(-0.14548536)
    spec_dic["HD-41094_F480M"] = np.array(-0.12436274)
    model = model.set("spectra", spec_dic)

if binary_flag:
    # Orbit prediction for 2025-11-11 (astrometry_ezaqr.ipynb): B at PA 66.31 deg from A (= A+C),
    # 0.1259"; A carries ~0.62 of the flux (Ks). `contrasts` is A's share, so it is 0.624, not
    # 0.376 (B's share).
    # for exp in calbin_exposures:
    # model = model.set(
    #     exp.map_param("pas"),
    #     np.array(66.31),
    # )
    # model = model.set(
    #     exp.map_param("separations"),
    #     np.array(0.1259),
    # )
    # model = model.set(
    #     exp.map_param("contrasts"),
    #     np.array(0.624),
    # )
    if not free_cal_spectra:
        spec_dic = model.spectra
        # [A (= A+C), B], from expecto.ipynb
        spec_dic["V-EZ-Aqr_F380M"] = np.array([-0.07376706, -0.07606107])
        spec_dic["V-EZ-Aqr_F430M"] = np.array([-0.14906244, -0.14530317])
        spec_dic["V-EZ-Aqr_F480M"] = np.array([-0.12971859, -0.12976371])
        model = model.set("spectra", spec_dic)

    # A must be the brighter star: `contrasts` is A's share of the flux, and the A-B swap is
    # nearly invisible to the data (it takes PA + 180 and 1 - contrast), so a state in the other
    # labelling would put A's spectrum on B. States from before amigo's BinaryFit sign fix
    # (2026-10-13) also hold the position angle in the old convention: convert_state_binary.py
    # converts both. Once A starts brighter, a fit cannot cross 0.5 (the pair is symmetric there).
    faint_A = {k: round(float(v), 4) for k, v in model.contrasts.items() if float(v) < 0.5}
    if faint_A:
        raise ValueError(f"binary A is the fainter star (contrast < 0.5) in {faint_A}: run convert_state_binary.py on the state")
    import inspect as _inspect
    from amigo.model_fits import BinaryFit as _BinaryFit
    if "-np.sin(phi)" not in _inspect.getsource(_BinaryFit.model_interferogram):
        raise RuntimeError(f"the amigo imported from {_inspect.getfile(_BinaryFit)} has the old BinaryFit sign "
                           "(position angles mirrored): states converted by convert_state_binary.py need the fixed one")

if best_weights is not None:
    model = model.set("nn_weights", best_weights)

# %% [markdown]
# ## Checking initial fits
# Just looking at one per filter. Starting with the calibrator exposures.

# %%
from amigo.plotting import summarise_fit

if plot_flag and (cal_flag or binary_flag):
    if save_flag:
        this_save_path = os.path.join(save_path, "init_cal")
        os.mkdir(this_save_path)
    else:
        this_save_path = None
    for exp in cal_exposures:
        exp.print_summary()
        # try:
        summarise_fit(model, exp, save_path=this_save_path, top_group=False)
        # except Exception as e:
        # print(f"Plotting failed: {e}")
        # pass

# %% [markdown]
# Now the validator exposures.

# %%
if plot_flag:
    if val_flag:
        if save_flag:
            this_save_path = os.path.join(save_path, "init_val")
            os.mkdir(this_save_path)
        else:
            this_save_path = None
        for exp in val_exposures:
            print(
                exp.program, exp.filter, exp.star, exp.ngroups, exp.nints, exp.filename
            )
            summarise_fit(model, exp, save_path=this_save_path)

# %% [markdown]
# Finally, the flats and darks.

# %%
if plot_flag:
    if flat_flag:
        if save_flag:
            this_save_path = os.path.join(save_path, "init_flat")
            os.mkdir(this_save_path)
        else:
            this_save_path = None
        for exp in flat_exposures:
            print(exp.program, exp.filter, exp.star, exp.ngroups, exp.nints)
            summarise_fit(model, exp, save_path=this_save_path)

# %%
if plot_flag:
    if dark_flag:
        if save_flag:
            this_save_path = os.path.join(save_path, "init_dark")
            os.mkdir(this_save_path)
        else:
            this_save_path = None
        for exp in dark_exposures:
            print(exp.program, exp.filter, exp.star, exp.ngroups, exp.nints)
            summarise_fit(model, exp, save_path=this_save_path)

# Residual statistics of the starting state: core / background rms of the per-pixel mean z-score of
# every calibrator and validator, against the noise floor and (CAL 4481) the AMIGO paper's fit
# (retrain_fns.residual_stats; the same numbers come with every summary). RETRAIN_RESIDUAL_STATS=0
# skips them here.
if save_flag and os.environ.get("RETRAIN_RESIDUAL_STATS", "1") == "1":
    from retrain_fns import residual_stats

    residual_stats(
        model,
        [("cal", e) for e in cal_exposures] + [("val", e) for e in val_exposures],
        save_path=save_path,
        name="init_residual_stats",
        title="starting state",
        epoch=0,
    )

# What the run does not use but a later run may need, kept in the saved states: without the
# binary, its own parameters and focus keys.
carry_params = {}
if not binary_flag:
    carry_params = {key: best_state[key] for key in ("pas", "separations", "contrasts") if key in best_state}
    carry_params["defocus"] = {k: v for k, v in best_state.get("defocus", {}).items() if k not in model.defocus}
    print(f"Without the binary; carried into the saved states: {sorted(carry_params)}"
          f" (focus keys {sorted(carry_params['defocus'])})")
# the holes' common displacement is fixed (the aperture's holes.b), so it is carried rather than saved
if pipeline_optics:
    carry_params["hole_shift"] = best_state["hole_shift"]

# %% [markdown]
# # Optimisation

# %% [markdown]
# ## Setup
# For the neural network training we want to use a custom learning rate warmup and a temperature decay, so we will have to modify a lot of the statistical functions of amigo to be able to handle this

# %%
from retrain_fns import (
    loss_fn,
    args_fn,
    looper_fn,
    batch_looper_fn,
    # grads_fn,
    aux_fn,
    ff_reg,
    nl_reg,
    sep_reg,
    pa_reg,
    linear_reg,
    temp_decay,
    cosine_warmup,
)

from retrain_fns import get_warmup, get_temperature

binary_ab_keys = [exp.get_key(PTT) for exp in calbin_exposures]
val_ab_keys = [exp.get_key(PTT) for exp in val_exposures]
ng3_ab_keys = [exp.get_key(PTT) for exp in ng3_exposures]
cal_stars = ["HD-41094", "V-EZ-Aqr"]
# Eigenbasis: validators fit the same modes as the calibrators, so nothing is truncated.
# RETRAIN_VAL_FULL_AB=1 does the same thing for the Zernike basis, as a fair-comparison
# control: isolates "validators get all 28 modes/hole, same as calibrators" from the
# eigenbasis's own basis-conditioning effect.
val_full_ab = os.environ.get("RETRAIN_VAL_FULL_AB", "0") == "1"
truncate_abs = (lambda x: x) if (eigen_flag or fourier_flag or base_opds_flag or val_full_ab) else (lambda x: x.at[:, 10:].set(0.0))
val_ab_lr_mult = float(os.environ.get("RETRAIN_VAL_AB_LR_MULT", 1.0))
# RETRAIN_FIT_MODES=<k> (one coefficient vector per key: eigen / Fourier bases): each key fits only its
# first k modes, the rest are held at their starting values (e.g. a whole-span conversion of an
# eigenbasis state, so the starting loss is kept while the freedom stays at k modes per key).
# RETRAIN_VAL_FIT_MODES does the same for the validators (default: as the calibrators). Applied to
# the gradient after the Fisher scaling, so the held modes get no update at all.
fit_modes = int(os.environ["RETRAIN_FIT_MODES"]) if "RETRAIN_FIT_MODES" in os.environ else None
val_fit_modes = int(os.environ.get("RETRAIN_VAL_FIT_MODES", 0)) or fit_modes
if fit_modes is not None or val_fit_modes is not None:
    print(f"Wavefront modes fitted per key: calibrators {fit_modes or 'all'}, validators {val_fit_modes or 'all'}")

# Validators' higher-order aberrations are poorly constrained and drift/destabilise if
# fit freely, so we keep them at zero. This used to be done by resetting the *value*
# back to zero every epoch in norm_fn (after the optimiser step) -- but that doesn't
# stop the gradient (same gotcha as #82: a post-hoc value reset on model_params isn't
# lax.stop_gradient), so a fully Fisher-normalised update was still computed and
# applied every epoch and only then thrown away, and optax's momentum state for those
# coefficients was never reset, letting a "phantom" one-step excursion grow over the
# whole run even though the saved value always read exactly zero. Fixed properly now:
# zero the value once up front, then zero the *gradient* every step in grads_fn
# (mirroring the existing "spectra" cal_stars block below), so momentum never gets fed
# a real signal for these coefficients in the first place.
model = model.set(
    PTT,
    {
        key: (truncate_abs(val) if key in val_ab_keys else val)
        for key, val in model.get(PTT).items()
    },
)


def norm_fn(model_params, args):
    """
    Normalisation function.
    """
    # flat field
    if hasattr(model_params, "FF"):
        FF = model_params.FF
        FF -= np.median(FF) - 1.0
        FF = FF.at[badpix_bool].set(1.0)
        model_params = model_params.set("FF", FF)

    # NOTE: badpix dark_current pixels used to be reset to the median here, post
    # optimiser-update. Moved to a one-time init reset + a grads_fn gradient-zero
    # instead -- see the comment near the init reset, and in grads_fn.

    # NOTE: validators' higher-order aberrations used to be reset to zero here, post
    # optimiser-update. Moved to a one-time init zero + a grads_fn gradient-zero
    # instead (see comments near val_ab_keys/truncate_abs and in grads_fn) -- a
    # post-hoc value reset on model_params doesn't stop the gradient, so this was
    # computing+applying a full Fisher-normalised step every epoch and discarding it,
    # while optax's momentum for those coefficients was never reset by the discard.

    # non linearity
    if hasattr(model_params, "non_linearity"):
        non_lin = model_params.non_linearity.at[:, badpix_bool].set(0)
        model_params = model_params.set("non_linearity", non_lin)

    # spectra clipping
    if hasattr(model_params, "spectra"):
        spectra = model_params.spectra
        clip_fn = lambda x: np.clip(x, a_min=-0.8, a_max=0.8)
        spectra = jax.tree.map(clip_fn, spectra)
        model_params = model_params.set("spectra", spectra)

    return model_params, args


def grads_fn(model, grads, args):

    # Get the parameters
    grad_params = grads.params

    if "spectra" in grad_params.keys() and not free_cal_spectra:
        for key, val in grad_params["spectra"].items():
            for star in cal_stars:
                if star in key:
                    val = jax.tree.map(lambda x: 0 * x, val)
                    grad_params["spectra"][key] = val

    # Zero the *gradient* for validators' higher-order aberrations (rather than
    # resetting the value post-update in norm_fn) -- see the comment where they're
    # zeroed at init, above. Keeps momentum from ever picking up a signal for these
    # coefficients, instead of computing+applying a full step every epoch only to
    # discard it.
    if PTT in grad_params.keys():
        for key, val in grad_params[PTT].items():
            if key in val_ab_keys:
                # Validators' (first 10) aberrations are far from converged and see
                # only their own exposure, so they can take a larger step than the
                # shared calibrator ones. Scaling the gradient here is a per-key lr
                # multiplier (post Fisher normalisation).
                grad_params[PTT][key] = truncate_abs(val) * val_ab_lr_mult
            n_fit = val_fit_modes if key in val_ab_keys else fit_modes
            if n_fit is not None and grad_params[PTT][key].ndim == 1:
                grad_params[PTT][key] = grad_params[PTT][key].at[n_fit:].set(0.0)
            # RETRAIN_NG3_TEST fits only the test exposures' own wavefronts
            if ng3_test and key not in ng3_ab_keys:
                grad_params[PTT][key] = 0.0 * grad_params[PTT][key]

    # RETRAIN_FILTER_PTT: the reference filter's term stays at zero (what all filters share is the visit PTT's)
    if "filter_ptt" in grad_params.keys() and filter_ptt_ref in grad_params["filter_ptt"]:
        grad_params["filter_ptt"][filter_ptt_ref] = 0.0 * grad_params["filter_ptt"][filter_ptt_ref]
    # RETRAIN_HOLE_ZERNIKE: likewise (the reference filter's per-hole shape is the Fourier set's)
    if "hole_zernike" in grad_params.keys() and hole_zernike_ref in grad_params["hole_zernike"]:
        grad_params["hole_zernike"][hole_zernike_ref] = 0.0 * grad_params["hole_zernike"][hole_zernike_ref]

    # Same fix, for badpix dark_current pixels (see the init-reset comment above).
    if "dark_current" in grad_params.keys():
        grad_params["dark_current"] = grad_params["dark_current"].at[badpix_bool].set(0)

    # if "aberrations" in grad_params.keys():
    #     grads = grads.multiply(binary_ab_keys, 1.0e2)

    # In Adam mode the warm-up and temperature are applied inside the optimiser instead
    # (see nn_adam), so the raw gradient passes through untouched.
    if "nn_weights" not in grad_params.keys() or nn_adam:
        grads = grads.set("params", grad_params)
        return grads, args

    # Get the key and update args with new key
    key, subkey = jr.split(args["key"], 2)
    args["key"] = subkey

    # Adds a temperature to the NN gradients
    values = grad_params["nn_weights"]
    temp = get_temperature(args)
    rand_vals = temp * jr.normal(key, values.shape)
    values += rand_vals

    # Add the learning rate warm-up (we also warm up the temperature here)
    values *= args["max_lr"] * get_warmup(args)

    # Increment the t parameter
    args["t"] += 1.0 / args["n_batch"]

    grad_params["nn_weights"] = values
    grads = grads.set("params", grad_params)

    return grads, args


# REGULARISATION
reg_dict = {
    "FF": (1.0, ff_reg),
    "NL": (1.0, nl_reg),
    # "PA": (1.0, sep_reg),
    # "SEP": (1.0, pa_reg),
    # "LIN": (10.0, linear_reg),
}

# RETRAIN_RIM_PRIOR=<sigma, nm>: a Gaussian prior on each wavefront's hole-edge structure. The rim is,
# in every hole, the OPD in the outer 8 px band that the cubic fitted to the 8-24 px ring just inside
# does not explain; its transmission-weighted mean square is c^T P c (build_rim_prior.py writes P for
# a basis and an aperture, RETRAIN_RIM_PRIOR_MATRIX), and each exposure adds (rim rms / sigma)^2 for its
# wavefront key. Fourier / eigen bases (one coefficient vector per key) only.
rim_sigma = float(os.environ["RETRAIN_RIM_PRIOR"]) if "RETRAIN_RIM_PRIOR" in os.environ else None
if rim_sigma is not None and base_opds_flag:
    raise ValueError("the rim prior is a matrix on the Fourier basis' coefficients, not the base-model wavefront parts")
if rim_sigma is not None:
    import numpy as onp_

    rim_P = np.asarray(onp_.load(os.environ.get(
        "RETRAIN_RIM_PRIOR_MATRIX", "/fred/oz440/max/outputs/fourierwf/rim_prior_full2k.npz"))["P"], float)

    def rim_reg(model, exposure, args={}):
        try:
            key = exposure.get_key(PTT)
        except ValueError:  # flats and darks have no wavefront
            return np.array(0.0)
        if key not in model.get(PTT):
            return np.array(0.0)
        c = model.get(PTT)[key]
        return c @ rim_P[: c.size, : c.size] @ c / rim_sigma**2

    reg_dict["RIM"] = (1.0, rim_reg)
    print(f"Hole-edge (rim) prior: sigma {rim_sigma} nm")

# %%
n_batch = len(nn_batches)

# WARM UP
max_lr = 1.  # Learning rate after warm up
t0 = 0  # Warm-up start
n_max = 100  # Warm-up end

# TEMPERATURE DECAY
# In Adam mode these are in units of Adam's normalised step (see nn_adam), hence the 0 defaults.
T0 = float(os.environ.get("RETRAIN_T0", 0.0 if nn_adam else 2e1))  # Initial temperature
TF = float(os.environ.get("RETRAIN_TF", 0.0 if nn_adam else 2e0))  # Final temperature
# T0 = 1e-6  # Initial temperature
# k=1e-10 keeps temperature ~constant at T0 for the whole run (intentional:
# other params are still optimising, so it's not time to anneal nn_weights
# noise down yet).
decay = float(os.environ.get("RETRAIN_DECAY", 1e-10))  # Decay rate of the temperature
# decay = 2e-3  # Decay rate of the temperature

args = {
    "n_batch": np.array(n_batch, float),  # Number of batches
    "key": jr.key(int(os.environ.get("RETRAIN_SEED", 0))),  # RNG Key
    "t": np.array(0.0),  # Initial time step
    "T0": np.array(T0),  # Initial temperature
    "TF": np.array(TF),  # Final temperature
    "k": np.array(decay),  # Decay rate
    "t0": np.array(t0),  # Warm-up start
    "n_max": np.array(n_max),  # Warm-up end point
    "l2": np.array(0.0),
    "max_lr": np.array(max_lr),  # Learning rate after warm up
    "reg_dict": reg_dict,
}

ts = np.arange(epochs)
temps = temp_decay(t=ts, T0=args["T0"], k=args["k"], TF=args["TF"])
lrs = cosine_warmup(ts, args["t0"], args["n_max"])
l2s = 0 * cosine_warmup(ts, 0, 1000)
args = {**args, **{"l2_schedule": l2s}}


fig, ax = plt.subplots(1, 2, figsize=(8, 2.5))
ax[0].set(title=f"Temperature schedule, T0: {args["T0"]}, k: {args["k"]}, TF: {args["TF"]}", xlabel="Epochs", ylim=(0, 1.05*T0), xlim=(0, epochs))
ax[0].axhline(TF, linestyle='--', color="red")
ax[0].plot(ts, temps, label="Temperature")
ax[1].set(title="Learning rate schedule", xlabel="Epochs", xlim=(0, epochs))
ax[1].plot(ts, lrs, label="Learning rate")

plt.tight_layout()
if save_flag:
    plt.savefig(os.path.join(save_path, "schedules.png"), dpi=200)
    plt.close()
else:
    plt.show()

# %% [markdown]
# Instantiating trainer class and updating fishers.

# %%
from amigo.legacy.calibration import (
    ValBatchedTrainer,
    Trainer,
)  # , looper_fn, aux_fn

# from dorito.stats import apply_regularisers
from retrain_fns import summarise_fn

# NOTE: This seems to nuke the jitter/SRF gradient??
linear_model = model.set(["psf_upsample", "bleed"], [1, False])

# Trainer class
trainer_class = ValBatchedTrainer if val_flag else Trainer

# The Fisher cache is keyed by exposure and parameter name only, not by aberration basis, so
# each basis gets its own cache directory (alternating bases used to recompute and overwrite
# the aberration Fishers every run). RETRAIN_FISHER_CACHE overrides the location.
# "linear_model_v2": Fishers computed after the bleed=False ramp fix (amigo ramp_models). The
# older caches (amigo_cache/fishers, amigo_cache/by_basis) came from the broken linear model
# (flux 1/9, FF applied twice), so they must never be mixed with new ones.
if eigen_flag or fourier_flag:
    basis_path = fourier_basis_path if fourier_flag else eigen_path
    basis_tag = ("fourier_" if fourier_flag else "eigen_") + os.path.splitext(os.path.basename(basis_path))[0]
    if n_eigen is not None:
        basis_tag += f"_n{n_eigen}"
else:
    basis_tag = "zernike_r7"
    if os.environ.get("RETRAIN_FOURIER", "0") == "1":
        basis_tag += f"_fourier{os.environ.get('RETRAIN_N_MODES_KEEP', 28)}"
fisher_cache = os.environ.get(
    "RETRAIN_FISHER_CACHE", os.path.join(cache_dir, "linear_model_v2", basis_tag)
)
print(f"Fisher cache: {fisher_cache}")
# A cache filled by a run with blur pairs holds 2x2 Fishers under the same name as the round
# blurs' 1x1 ones (and the other way round): refuse rather than mix them up.
from glob import glob as _glob

for _f in _glob(os.path.join(fisher_cache, "fishers", "*", "sigmas.npy"))[:1]:
    if np.load(_f).shape != ((2, 2) if sigma_pair else (1, 1)):
        raise ValueError(f"{fisher_cache} holds {'round' if sigma_pair else 'pair'} blur Fishers ({_f});"
                         " use another RETRAIN_FISHER_CACHE for " + ("pairs" if sigma_pair else "round blurs"))

# Fishers are cached per parameter name ({cache}/fishers/<file>/<param>.npy): a cache filled before amigo's 2026-11
# rename holds the base-model parts as fourier_opd / filter_opd / aberrations. Hard-link them to the new names (the
# same parameters; a link adds no file) so they are not recomputed.
if base_opds_flag:
    n_linked = 0
    for _dir in _glob(os.path.join(fisher_cache, "fishers", "*")):
        for old, new in OLD_WF_NAMES.items():
            _old, _new = os.path.join(_dir, f"{old}.npy"), os.path.join(_dir, f"{new}.npy")
            if os.path.exists(_old) and not os.path.exists(_new):
                os.link(_old, _new)
                n_linked += 1
    if n_linked:
        print(f"Fisher cache: {n_linked} Fishers linked to the renamed parameters")

# Only the aberration Fishers depend on the basis, so a new basis starts from another basis's
# Fishers for every other parameter instead of recomputing them all.
if "RETRAIN_FISHER_CACHE" not in os.environ:
    import shutil
    from glob import glob

    n_seeded = 0
    for other in sorted(glob(os.path.join(cache_dir, "linear_model_v2", "*", "fishers"))):
        if os.path.dirname(other) == fisher_cache:
            continue
        for src in glob(os.path.join(other, "*", "*.npy")):
            dst = os.path.join(fisher_cache, "fishers", os.path.relpath(src, other))
            if os.path.basename(src) != f"{PTT}.npy" and not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                n_seeded += 1
    if n_seeded:
        print(f"Fisher cache: {n_seeded} basis-independent Fishers copied from other bases")

trainer_class_kwargs = {
    "loss_fn": loss_fn,
    "args_fn": args_fn,
    "grad_fn": grads_fn,
    "norm_fn": norm_fn,
    "summarise_fn": summarise_fn,
    "intermediate_prints": intermediate_prints,
    "save_path": save_path,
    "cache": fisher_cache,
    # Record param history every Nth epoch instead of every epoch (nn_weights is
    # exempt, always recorded every epoch -- see Trainer's docstring). Cuts the
    # per-epoch device sync + O(epochs^2) list growth that was measurably slowing
    # down long runs (confirmed ~17.0s/epoch -> ~19.4s/epoch over 3500 epochs on
    # the init_v44 overnight run), with zero effect on any saved/reported number:
    # only the parameter-trajectory plots (params_*.png) get coarser resolution.
    # Only applies to the plain Trainer's history (val_flag=False); ValBatchedTrainer
    # has a second, faster-growing one (batch_history, per batch not per epoch) --
    # see batch_history_max_len below for that.
    "history_stride": int(os.environ.get("RETRAIN_HISTORY_STRIDE", 5)),
    # ValBatchedTrainer-only (batched_params, e.g. nn_weights): caps batch_history to
    # its last N entries rather than letting it grow for the whole run. This OOM-killed
    # a 5000-epoch val_flag=True run at epoch 3000 (nn_weights alone appended 16x/epoch,
    # unbounded, ~3.3GB and climbing by then). n_batch*200 keeps ~200 epochs of full
    # per-batch resolution (retrain_fns.py's final-state averaging only ever needs the
    # last n_batch of it) -- plenty for a trajectory plot, nowhere near unbounded.
    "batch_history_max_len": int(os.environ.get("RETRAIN_BATCH_HISTORY_MAX_LEN", n_batch * 200)),
}

# RETRAIN_MULTI_GPU=1 splits each epoch's batches across every visible GPU by
# estimated cost, data-parallel (see Trainer's devices docstring in amigo.fitting).
# Works with both trainer classes now, but ValBatchedTrainer's version changes
# nn_weights' update cadence from ~n_batch small steps/epoch to 1 larger step/epoch
# (needed to make cal/flat batches embarrassingly parallel too, not just
# validator batches) -- see the multigpu-validator-required note. Not yet
# re-validated for convergence against the single-GPU val_flag=True baseline.
if os.environ.get("RETRAIN_MULTI_GPU") == "1":
    trainer_class_kwargs["devices"] = jax.local_devices()
    print(f"Multi-GPU: splitting batches across {len(jax.local_devices())} devices")

if val_flag:
    trainer_class_kwargs["aux_fn"] = aux_fn
    trainer_class_kwargs["looper_fn"] = looper_fn
else:
    trainer_class_kwargs["looper_fn"] = batch_looper_fn

trainer = trainer_class(**trainer_class_kwargs)

# TODO should there be all the detector parameters here?
if flat_flag:
    print("Flat exposures")
    trainer = trainer.update_fishers(
        linear_model,
        flat_exposures,
        parameters=["flat_coeffs", "fluxes"],
        recalculate=recalculate_flag,
        overwrite=recalculate_flag,
    )

if val_flag:
    print("Val exposures")
    trainer = trainer.update_fishers(
        linear_model,
        val_exposures,
        batch_size=10,
        parameters=unfrozen(
            ["positions", "low_visit" if hier_wf else PTT, "fluxes", "spectra"]
            + (["defocus"] if own_defocus else [])
            + (["rotations"] if fit_rotation else [])
        ),
        recalculate=recalculate_flag,
        overwrite=recalculate_flag,
    )

if cal_flag or binary_flag:
    print("Batched cal")
    ps = [PTT, "defocus"] + (["low_filter", "low_visit"] if hier_wf else []) + wf_params
    if not static_optics:
        ps += aperture_params
        # ps += ["distortions", "primary_beams"]
    if unfrozen(ps):
        trainer = trainer.update_fishers(
            linear_model,
            cal_exposures,
            batch_size=10,
            parameters=unfrozen(ps),
            recalculate=recalculate_flag,
            overwrite=recalculate_flag,
        )

    print("Unbatched cal")
    trainer = trainer.update_fishers(
        linear_model,
        cal_exposures,
        parameters=unfrozen([
            "positions",
            "fluxes",
            "spectra",
            "sigma",  # formerly jitter
            # "dark_current",
        ] + (["rotations"] if fit_rotation else []) + ["sigmas"]),
        recalculate=recalculate_flag,
        overwrite=recalculate_flag,
    )

    if binary_flag:
        print("Binary Exposures")
        trainer = trainer.update_fishers(
            linear_model,
            calbin_exposures,
            parameters=[
                "pas",
                "separations",
                "contrasts",
            ],
            recalculate=recalculate_flag,
            overwrite=recalculate_flag,
        )

# The rim prior's curvature goes into every exposure's aberration Fisher (amigo computes them without
# the priors, and stores minus the loss Hessian), so the Fisher-normalised steps stay stable along
# the weakly constrained edge modes the prior acts on.
if rim_sigma is not None:
    fishers = dict(trainer.fishers)
    n_added = 0
    for exp in cal_exposures + val_exposures:
        name = f"{exp.key}.{PTT}"
        if name in fishers:
            n = fishers[name].shape[0]
            fishers[name] = fishers[name] - 2 * rim_P[:n, :n] / rim_sigma**2
            n_added += 1
    trainer = trainer.set("fishers", fishers)
    print(f"Rim prior curvature added to {n_added} aberration Fishers")

# %% [markdown]
# Training.

# %%
from amigo.legacy.fitting import sgd, adam
import optax
from optax.schedules import linear_schedule

# Learning-rate groups: RETRAIN_LR_GROUPS is a comma list of groups that use the
# new (raised) values, the rest keep their old ones. Default "optics,nn": the raises we
# keep by default; "all" = every group new (binary/ff raises were not kept, see sweeps).
lr_groups = os.environ.get("RETRAIN_LR_GROUPS", "optics,nn").split(",")


# Optional overrides for sweeps: RETRAIN_NN_LR sets the nn_weights lr,
# RETRAIN_BINARY_MULT scales the *old* binary lrs (instead of the fixed new values), and
# RETRAIN_NN_MOMENTUM sets the nesterov momentum used for nn_weights (default 0.0, other
# params stay at 0.6). Momentum=0.6 on the raised nn_weights lr caused a reproducible
# loss spike (~epoch 80-100, confirmed across 2 seeds); dropping momentum to 0 removed
# it in every test run with no cost to the final loss, so that's now the default. 0.2
# still left a small residual wobble, so we didn't split the difference.
nn_lr_override = os.environ.get("RETRAIN_NN_LR")
binary_mult = os.environ.get("RETRAIN_BINARY_MULT")
nn_momentum = float(os.environ.get("RETRAIN_NN_MOMENTUM", 0.0))
# Short single-GPU A/B test (2026-09-30): defocus/primary_beam/sigma/distortion were
# all still drifting smoothly with no oscillation at epoch 5000 of the
# eigen300_v47_5000 run (params_4/params_10/params_12.png there) -- doubling each to
# see if that's actually safe before committing to a long run. FF/non_linearity/
# aberrations deliberately untouched, see the retrain.ipynb optimisers cell comment.
lr_boost_flag = os.environ.get("RETRAIN_LR_BOOST", "0") == "1"


def pick(group, old, new, momentum=0.6):
    """(lr, start_epoch) tuples -> optimiser, using `new` if the group is enabled."""
    if group == "nn" and nn_lr_override is not None:
        new = (float(nn_lr_override), new[1])
    if group == "binary" and binary_mult is not None:
        new = (old[0] * float(binary_mult), old[1])
    return sgd(*(new if ("all" in lr_groups or group in lr_groups) else old), momentum=momentum)


# Define the optimisers
optimisers = {
    "fluxes": sgd(1e-3, 0),
    "flat_coeffs": sgd(1e-3, 0),
    "positions": sgd(1e-3, 2),
    PTT: pick("optics", (2e-5, 5), (5e-5, 5)),
    "fourier_coeffs": pick("optics", (2e-5, 5), (5e-5, 5)),
    "zernike_coeffs": pick("optics", (2e-5, 5), (5e-5, 5)),
    "filter_ptt": pick("optics", (2e-5, 5), (5e-5, 5)),  # as the PTT it copies
    "hole_zernike": pick("optics", (2e-5, 5), (5e-5, 5)),  # as the wavefront parts
    "hole_amplitude": pick("optics", (2e-5, 5), (5e-5, 5)),  # likewise (Fisher-normalised, ratio as theirs)
    # "aberrations": optax.sgd(linear_schedule(0, 1e-4, 50, 5)),
    "pas": pick("binary", (1e-4, 5), (4e-4, 5)),
    "separations": pick("binary", (5e-5, 7), (5e-4, 7)),
    "contrasts": pick("binary", (2e-4, 8), (2e-3, 8)),
    "nn_weights": pick("nn", (2e-6, 0), (5e-6, 0), momentum=nn_momentum),  # lr applied with warm up
    "defocus": pick("optics", (1e-4, 25), (2.5e-4 * (2.0 if lr_boost_flag else 1.0), 25)),
    # the pipeline-branch aperture's hole positions and warps take distortion's and primary_beam's settings
    aperture_params[0]: sgd(5e-5 * (2.0 if lr_boost_flag else 1.0), 30),
    "dark_current": sgd(3e-2, 35),
    aperture_params[1]: pick("optics", (1e-5, 40), (4e-5 * (2.0 if lr_boost_flag else 1.0), 40)),
    "sigma": pick("optics", (1e-4, 50), (2e-4 * (2.0 if lr_boost_flag else 1.0), 50)),  # start at 50th?
    # RETRAIN_ALLOW_CAL_PIXEL_GRADS adds ~45 calibrator/binary exposures into the
    # sum that drives these two (previously just the 3 in flat_0 -- 2 flats + 1
    # dark), a ~16x jump in raw gradient magnitude at a fixed lr never calibrated
    # by a Fisher/curvature estimate for either param (no update_fishers(...) call
    # anywhere requests "FF" or "non_linearity", from any exposure type). Testing
    # whether just scaling the lr down to compensate stops the blowup seen at the
    # exact epoch non_linearity's lr activates, or whether it's a deeper problem.
    "non_linearity": sgd(1e-4 * float(os.environ.get("RETRAIN_FF_NL_LR_SCALE", 1.0)), 100),
    "FF": sgd(
        (2e-4 if ("ff" in lr_groups or "all" in lr_groups) else 2e-5)
        * float(os.environ.get("RETRAIN_FF_NL_LR_SCALE", 1.0)),
        100 if ("ff" in lr_groups or "all" in lr_groups) else 150,
    ),
    "spectra": sgd(1e-3, 20),
    # Fisher-normalised like the rest, but its Fisher is new (never the pre-fix one), so it is
    # not in fisher_ratio below: this is the step as a fraction of a Newton step, directly.
    "rotations": sgd(2e-2, 2),
    "sigmas": sgd(2e-2, 2),  # likewise
    # hierarchical wavefronts (RETRAIN_WF): new Fishers too, so a fraction of a Newton step directly
    "low_filter": sgd(5e-3, 5),
    "low_visit": sgd(5e-3, 5),
}
if one_on_fs_flag:
    optimisers["one_on_fs"] = adam(float(os.environ.get("RETRAIN_ONE_ON_FS_LR", 1e-2)), 5)
if not fit_rotation:
    optimisers.pop("rotations")
if not hier_wf:
    optimisers.pop("low_filter")
    optimisers.pop("low_visit")
if not base_opds_flag:
    optimisers.pop("fourier_coeffs")
    optimisers.pop("zernike_coeffs")
if filter_ptt_mode != "1":  # off, or fixed (read from the state, never fitted)
    optimisers.pop("filter_ptt")
if not hole_zernike_flag:
    optimisers.pop("hole_zernike")
if amp_orders is None:
    optimisers.pop("hole_amplitude")

if not (cal_flag or binary_flag):
    cal_ps = ["nn_weights", "defocus", *aperture_params, "sigma", *wf_params]
    optimisers = {key: val for key, val in optimisers.items() if key not in cal_ps}

if not binary_flag:
    bin_ps = ["pas", "separations", "contrasts"]
    optimisers = {key: val for key, val in optimisers.items() if key not in bin_ps}

if not flat_flag:
    flat_ps = ["flat_coeffs", "non_linearity", "FF"]
    optimisers = {key: val for key, val in optimisers.items() if key not in flat_ps}

if not dark_flag:
    dark_ps = ["dark_current"]
    optimisers = {key: val for key, val in optimisers.items() if key not in dark_ps}

if not val_flag:
    val_ps = ["spectra"]
    optimisers = {key: val for key, val in optimisers.items() if key not in val_ps}

if static_optics:
    dyn_ps = aperture_params
    optimisers = {key: val for key, val in optimisers.items() if key not in dyn_ps}

if nn_adam and "nn_weights" in optimisers:
    from retrain_fns import tempered_adam

    # ValBatchedTrainer updates nn_weights once per cal/flat batch on a single GPU, but
    # once per epoch (accumulated gradient) under RETRAIN_MULTI_GPU=1; the plain Trainer
    # (val_flag=False) is treated as once per epoch too.
    multi_gpu = os.environ.get("RETRAIN_MULTI_GPU") == "1"
    nn_updates_per_epoch = n_batch if (val_flag and not multi_gpu) else 1
    optimisers["nn_weights"] = tempered_adam(
        lr=float(os.environ.get("RETRAIN_NN_ADAM_LR", 1e-5)),
        updates_per_epoch=nn_updates_per_epoch,
        warmup_epochs=n_max,
        T0=T0,
        TF=TF,
        decay=decay,
        seed=int(os.environ.get("RETRAIN_SEED", 0)),
    )
    print(
        f"nn_weights: Adam, lr {os.environ.get('RETRAIN_NN_ADAM_LR', 1e-5)}, "
        f"{nn_updates_per_epoch} updates/epoch, T0 {T0}, TF {TF}, decay {decay}"
    )

optimisers = {key: val for key, val in optimisers.items() if key not in frozen_params}

# The learning rates above were tuned against Fishers from the old, broken linear model
# (flux 1/9), which were too small, so every Fisher-normalised step was larger than its lr
# says. The corrected Fishers (linear_model_v2 cache) are larger by the factors below: the
# median new/old Fisher diagonal over exposures, measured 2026-10-01. Aberrations can't be
# compared across bases, so they take the other pupil parameters' value, and the binary
# parameters take positions'. Each lr is multiplied by its factor so the steps stay the size
# the lrs were tuned for (nn_weights and the pixel maps have no Fisher, so they are untouched).
# RETRAIN_FISHER_LR_SCALE multiplies all of them again, to speed up or slow down a test.
fisher_ratio = {
    "flat_coeffs": 25.0,
    "fluxes": 35.0,
    "positions": 35.0,
    "spectra": 45.0,
    PTT: 47.0,
    "fourier_coeffs": 47.0,
    "zernike_coeffs": 47.0,
    "filter_ptt": 47.0,
    "hole_zernike": 47.0,
    "hole_amplitude": 47.0,
    "defocus": 47.0,
    aperture_params[0]: 47.0,
    aperture_params[1]: 50.0,
    "sigma": 36.0,
    "pas": 35.0,
    "separations": 35.0,
    "contrasts": 35.0,
}
fisher_lr_scale = float(os.environ.get("RETRAIN_FISHER_LR_SCALE", 1.0))
if "linear_model_v2" in fisher_cache:
    optimisers = {
        key: (
            optax.chain(val, optax.scale(fisher_ratio[key] * fisher_lr_scale))
            if key in fisher_ratio
            else val
        )
        for key, val in optimisers.items()
    }
    print(f"Fisher-normalised learning rates scaled by {fisher_ratio}, times {fisher_lr_scale}")

print(optimisers.keys())


# This seems to fix some recompile issues
def fn(x):
    if isinstance(x, jax.Array):
        if "i" in x.dtype.str:
            return x
        return np.array(x, dtype=float)
    return x


model = jtu.map(fn, model)

batches = {}
if cal_flag:
    batches = {**batches, **cal_batch}
if val_flag:
    batches = {**batches, **val_batch}
if flat_flag:
    batches = {**batches, **flat_batch}

summarise_kwargs = {
    "val_flag": val_flag,
    "flat_flag": flat_flag,
    "binary_flag": binary_flag,
    "save_flag": save_flag,
    "cal_flag": cal_flag,
    "dark_flag": dark_flag,
    "static_optics": static_optics,
    "cal_exposures": cal_exposures,
    "val_exposures": val_exposures,
    "flat_exposures": flat_exposures,
    "dark_exposures": dark_exposures,
    "calpsf_exposures": calpsf_exposures,
    "calbin_exposures": calbin_exposures,
    "badpix": badpix,
    "n_batch": n_batch,
    "optimisers": optimisers,
    "amigo_files_path": amigo_files_path,
    "history_stride": trainer_class_kwargs["history_stride"],
    "carry_params": carry_params,
}

common_trainer_kwargs = {
    "model": model,
    "optimisers": optimisers,
    "epochs": epochs,
    "batches": batches,
    "args": args,
    "summarise_kwargs": summarise_kwargs,
}

if val_flag:
    if "nn_weights" in optimisers.keys():
        common_trainer_kwargs["batched_params"] = ["nn_weights"]
    else:
        common_trainer_kwargs["batched_params"] = []

val_trainer_kwargs = {
    "validators": val_batch,
    "validator_params": unfrozen(
        ["positions", "fluxes", "low_visit" if hier_wf else PTT, "spectra"]
        + (["defocus"] if own_defocus else [])
        + (["rotations"] if fit_rotation else [])
        + (["one_on_fs"] if one_on_fs_flag else [])
    ),
}

trainer_kwargs = (
    {**common_trainer_kwargs, **val_trainer_kwargs}
    if val_flag
    else common_trainer_kwargs
)

# with jax.disable_jit():
result = trainer.train(**trainer_kwargs)

# %% [markdown]
# Plotting the results of that fit!

# %% [markdown]
# # Saving results

# %%
summarise_fn(result, save_path, **summarise_kwargs)

# %%



