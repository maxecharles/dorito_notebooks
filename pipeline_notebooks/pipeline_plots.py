"""Fit-quality plots for an amigo-pipeline output folder.

Everything here reads the saved products, so it works on a finished run without
refitting. The uv-plane maps rebuild log-visibility from the saved basis modes and
coefficients: the same linear map the DISCOs use, so the maps carry exactly the
information the binary fit sees, laid out in the uv plane.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd

RESIDUAL_CMAP = plt.get_cmap("RdBu_r").copy()
RESIDUAL_CMAP.set_bad("#bdbdbd")


def _stage_dirs(output):
    output = Path(output)
    for name in ("instrument", "reference"):
        if (output / name / "fit.json").exists():
            yield name, output / name
    for path in sorted((output / "visibility").glob("*/fit.json")):
        yield "visibility", path.parent


def _residuals(path):
    """Per-exposure whitened residuals, scaled so a perfect fit has unit RMS per pixel."""
    rows = json.loads((path / "pixels.json").read_text())
    for row in rows:
        item = onp.load(path / row["file"])
        whitened = item["whitened"].reshape(item["data"].shape)
        image = onp.full(tuple(item["shape"]), onp.nan)
        # Summing over slopes and dividing by sqrt(n) keeps unit variance per pixel.
        image[item["rows"], item["columns"]] = whitened.sum(-1) / onp.sqrt(whitened.shape[-1])
        yield row, image, float(onp.sqrt(onp.mean(whitened**2)))


def fit_summary(output):
    """One row per exposure per fit: convergence and whitened residual RMS (1 = noise-limited)."""
    records = []
    for stage, path in _stage_dirs(output):
        fit = json.loads((path / "fit.json").read_text())
        for row, _, rms in _residuals(path):
            records.append(
                {
                    "stage": stage,
                    "exposure": row["exposure"],
                    "filter": row["filter"],
                    "role": row["role"],
                    "residual_rms_sigma": rms,
                    "status": fit["status"],
                    "gradient": fit["gradient"],
                    "minutes": fit["seconds"] / 60,
                }
            )
    return pd.DataFrame(records).set_index(["stage", "exposure"])


def residual_grid(output, limit=10.0, keys=None):
    """Whitened detector residuals per exposure: point-source fit (top) vs visibility fit (bottom).

    keys picks the exposures to draw (default all), for datasets too big for one row.
    """
    point, vis = {}, {}
    for stage, path in _stage_dirs(output):
        for row, image, rms in _residuals(path):
            (vis if stage == "visibility" else point)[row["exposure"]] = (row, image, rms)
    order = sorted(
        (k for k in vis if keys is None or k in keys),
        key=lambda k: (vis[k][0]["filter"], vis[k][0]["role"], k),
    )
    fig, axes = plt.subplots(
        2, len(order), figsize=(2.7 * len(order), 6.2), layout="constrained", squeeze=False
    )
    for col, key in enumerate(order):
        for row_index, (label, results) in enumerate(
            [("point-source fit", point), ("visibility fit", vis)]
        ):
            ax = axes[row_index, col]
            if key not in results:
                ax.set_axis_off()
                continue
            row, image, rms = results[key]
            m = ax.imshow(
                image,
                origin="lower",
                cmap=RESIDUAL_CMAP,
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            ax.set_title(f"{row['filter']} {row['role']}\n{key}\nRMS {rms:.2f}σ", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label)
    fig.colorbar(m, ax=axes, shrink=0.8, label="Whitened residual (σ)")
    fig.suptitle("Detector residuals (1σ RMS = noise-limited; grey = masked pixels)")
    return fig


def _full_plane(half):
    """Fill the uv grid from its first half using V(-u) = conj V(u); log V(0) = 0."""
    n = 2 * half.size + 1
    flat = onp.zeros(n, dtype=half.dtype if onp.iscomplexobj(half) else float)
    flat[: half.size] = half
    flat[half.size + 1 :] = onp.conj(half[::-1])
    side = int(round(onp.sqrt(n)))
    return flat.reshape(side, side)


def _uv_inputs(output, name, disco):
    output = Path(output)
    filt = name.split("_")[1]
    basis = onp.load(output / "basis" / filt / "basis.npz")
    calibration = dict(onp.load(output / "assembly" / name / "calibration.npz"))
    modes = basis["modes"]
    power = _full_plane((onp.abs(modes) ** 2).sum(0)).real
    sampled = power > 1e-3 * power.max()
    half = float(basis["uv"].max()) / float(disco.wavelength) / 1e6
    return modes, calibration, sampled, [-half, half, -half, half]


def _uv_panel(ax, image, extent, sampled, limit, title):
    m = ax.imshow(
        onp.where(sampled, image, onp.nan),
        origin="lower",
        extent=extent,
        cmap=RESIDUAL_CMAP,
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=10)
    plt.colorbar(m, ax=ax, fraction=0.046, pad=0.03)


def _robust_limit(images, sampled):
    return max(float(onp.quantile(onp.abs(im[sampled]), 0.99)) for im in images)


def _label_axes(axes):
    for ax in axes[-1]:
        ax.set_xlabel("u (Mλ)")
    for ax in axes[:, 0]:
        ax.set_ylabel("v (Mλ)")


def uv_visibilities(output, name, disco):
    """Fitted log-visibility in the uv plane: science, calibrator and their difference.

    Science and calibrator share a colour scale per row so the instrument signature
    common to both is visible; the calibrated column has its own scale.
    """
    modes, calibration, sampled, extent = _uv_inputs(output, name, disco)
    maps = {
        key: _full_plane(calibration[key] @ modes)
        for key in ("science", "calibrator", "calibrated")
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8), layout="constrained")
    for r, (part, label) in enumerate([(onp.real, "ln|V|"), (onp.imag, "phase (rad)")]):
        raw = _robust_limit([part(maps["science"]), part(maps["calibrator"])], sampled)
        for c, (key, image) in enumerate(maps.items()):
            limit = raw if key != "calibrated" else _robust_limit([part(image)], sampled)
            _uv_panel(axes[r, c], part(image), extent, sampled, limit, f"{key.capitalize()}: {label}")
    _label_axes(axes)
    fig.suptitle(f"{name}: fitted log-visibility (instrument frame; grey = not sampled)")
    return fig


def uv_model_comparison(output, name, disco, source, label="model"):
    """Calibrated log-visibility vs a Virgil source model, per uv point in units of σ.

    Both are projected onto the observable subspace (the null removed), exactly as
    in the DISCO, so this is the DISCO comparison drawn in the uv plane. Neighbouring
    uv points are correlated, so the per-point RMS is a guide, not a chi-squared.
    """
    modes, calibration, sampled, extent = _uv_inputs(output, name, disco)
    kernel = calibration["kernel"]
    projector = kernel.T @ onp.linalg.solve(kernel @ kernel.T, kernel)
    u, v = disco.sky_uv
    log_model = onp.log(onp.asarray(source.model(u, v, disco.wavelength)))
    model_latent = onp.real(log_model @ modes.conj().T)
    data = _full_plane(projector @ calibration["calibrated"] @ modes)
    model = _full_plane(projector @ model_latent @ modes)
    covariance = projector @ calibration["covariance"] @ projector.T
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8), layout="constrained")
    for r, (part, quantity, real_modes) in enumerate(
        [(onp.real, "ln|V|", modes.real), (onp.imag, "phase", modes.imag)]
    ):
        sigma = onp.sqrt(onp.einsum("ik,ij,jk->k", real_modes, covariance, real_modes))
        sigma = _full_plane(sigma).real
        with onp.errstate(divide="ignore", invalid="ignore"):  # sigma = 0 only off the mask
            panels = {
                "Data": part(data) / sigma,
                label.capitalize(): part(model) / sigma,
                "Residual": (part(data) - part(model)) / sigma,
            }
        limit = _robust_limit([panels["Data"]], sampled)
        for c, (key, image) in enumerate(panels.items()):
            title = f"{key}: {quantity} / σ"
            if key == "Residual":
                title += f"  (RMS {onp.sqrt(onp.mean(image[sampled] ** 2)):.2f})"
            _uv_panel(axes[r, c], image, extent, sampled, limit, title)
    _label_axes(axes)
    fig.suptitle(f"{name}: calibrated visibilities vs {label} (instrument frame)")
    return fig
