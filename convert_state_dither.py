"""Convert the 0.2 state (focus and PTT keyed by dither position POS1-4, one rotation per program) to the keys of
retrain_dither.py with dither_flag = True: focus per program and filter, PTT per program, filter and primary offset,
and the primary offsets' fixed constants.

    python convert_state_dither.py <state in> <groups.json from check_dither.py> <state out>
        <rotation_per_exposure.csv> <manifest.json>   (OzSTAR's 2026-11-25 per-exposure fits, for 01093's focus)

groups.json (a control run of check_dither.py, dither_flag False) gives, from the notebook's own exposures, which
primary-offset labels sit under every rotation key and every (program, POS).

Constants, relative to the (0, 0) position, "x+0p0y+0p0" (which has none):
  focus (um): from the 0.2 state itself where it separates the positions: x-1p0y+1p0 the mean of 8330's three
    filters and 1509 (18.1, 18.0, 20.1, 24.1), x-2p0y-1p0 and x+1p0y-1p0 from 1509's POS3 and POS4 (F480M only).
    x-2p0y+1p0 (1093 obs 004/006, validators only) is not in the state (it shares "01093_POS2" with obs 019's
    x-1p0y+1p0, a different position): the mean of the six per-exposure differences of OzSTAR's 2026-11-25
    calibrator fits, +38.3, +33.8, +29.8 and +37.2, +35.7, +29.5 um (F380M, F430M, F480M of obs 004 and 006).
  rotation (deg): the per-position offsets of OzSTAR's 2026-11-25 calibrator fits (step4/pwcpos, pwcpos_pairs.txt).
New base values:
  focus per (rotation key, filter): the state's value at the (0, 0) position, so (0, 0) exposures are unchanged.
    01093's visits, weeks apart, differ by ~40 um (-71 against -31 um at (0, 0)), and the state's one "01093_POS1"
    pools them, so 01093's keys (per observation, as the rotation's) start from the per-exposure fits instead:
    the mean of (fit - the position's constant);
  rotation per key: the state's minus the mean of its exposures' position constants, so the key's mean rotation is
    unchanged.
"""

import csv
import json
import math
import sys

import numpy as onp
from astropy.io import fits

state_in, groups_path, state_out, table_path, manifest_path = sys.argv[1:6]
REFERENCE = "x+0p0y+0p0"
ROTATION = {"x-1p0y+1p0": 0.0028, "x-2p0y-1p0": -0.0137, "x+1p0y-1p0": -0.0116, "x-2p0y+1p0": -0.0098}
FOCUS_1093_2POINT = float(onp.mean([38.3, 33.8, 29.8, 37.2, 35.7, 29.5]))

state = onp.load(state_in, allow_pickle=True).item()
groups = json.load(open(groups_path))["groups"]
pos_to_label = {}
for key, labels in groups["pos_to_label"].items():
    pos_to_label[key] = labels[0] if len(labels) == 1 else None  # None: two different positions under one POS
    if len(labels) > 1:
        print(f"{key} covers {labels}: ambiguous, not used for constants or PTT")

# ---- focus
old_focus = {k: float(v) for k, v in state["defocus"].items()}
CALIBRATORS = ("04481", "08330", "01509")  # the programs whose positions the state separates cleanly
base, labelled = {}, {}
for key, value in old_focus.items():
    program, filt, pos = key.split("_")
    label = pos_to_label.get(f"{program}_{pos}")
    if program not in CALIBRATORS and program != "01093":
        if label == REFERENCE:
            base[f"{program}_{filt}"] = value
        continue
    if program in CALIBRATORS:
        labelled[(program, filt, label)] = value
        if label == REFERENCE:
            base[f"{program}_{filt}"] = value
differences = {}
for (program, filt, label), value in labelled.items():
    if label != REFERENCE:
        differences.setdefault(label, []).append(value - base[f"{program}_{filt}"])
focus = {label: float(onp.mean(d)) for label, d in differences.items()}
for label, d in sorted(differences.items()):
    print(f"focus {label}: state differences {onp.round(d, 1).tolist()} -> {focus[label]:+.2f}")
focus["x-2p0y+1p0"] = FOCUS_1093_2POINT
print(f"focus x-2p0y+1p0: per-exposure fits -> {focus['x-2p0y+1p0']:+.2f}")
assert set(focus) == set(ROTATION), (set(focus), set(ROTATION))

# 01093: one focus per observation (the rotation's key) and filter, from OzSTAR's per-exposure fits
paths = {r["path"].split("/")[-1].replace("_nis_calslope.fits", ""): r["path"] for r in json.load(open(manifest_path))["exposures"]}


def label_of(path):
    h = fits.getheader(path, 0)
    x, y = (math.floor(2 * float(h[k]) + 0.5) / 2 + 0.0 for k in ("XOFFSET", "YOFFSET"))
    return f"x{x:+.1f}y{y:+.1f}".replace(".", "p")


fits_1093 = {}
for r in csv.DictReader(open(table_path)):
    if r["program"] == "01093":
        label = label_of(paths[r["file"]])
        fits_1093.setdefault((r["observtn"], r["filter"]), []).append(
            float(r["defocus"]) - (0.0 if label == REFERENCE else focus[label])
        )
for key in groups["rotations"]:
    if not key.startswith("01093_"):
        continue
    obs = key.split("_")[1]
    for (o, filt), values in sorted(fits_1093.items()):
        if o == obs:
            base[f"{key}_{filt}"] = float(onp.mean(values))
            print(f"focus {key}_{filt}: {base[f'{key}_{filt}']:.1f} (mean of {len(values)} per-exposure fits less the constant)")

# ---- rotation
rotations = {}
for key, value in state["rotations"].items():
    counts = groups["rotations"].get(key)
    if counts is None:
        rotations[key] = onp.asarray(float(value))
        print(f"rotation {key}: no exposures in the notebook's set, unchanged")
        continue
    n = sum(counts.values())
    shift = sum(ROTATION.get(label, 0.0) * c for label, c in counts.items()) / n
    rotations[key] = onp.asarray(float(value) - shift)
    print(f"rotation {key}: {float(value):+.5f} - mean constant {shift:+.5f} = {float(rotations[key]):+.5f}  {counts}")

# ---- PTT keys
ptt = {}
for key, value in state["ptt_coeffs"].items():
    parts = key.split("_")
    if len(parts) == 3 and parts[2].startswith("POS"):  # calibrators: program, filter, position
        program, filt, pos = parts
        label = pos_to_label[f"{program}_{pos}"]
        assert label is not None, f"PTT key {key} sits at two positions"
        ptt[f"{program}_{filt}_{label}"] = value
    else:  # validators: per exposure
        ptt[key] = value

new = dict(state)
new["defocus"] = {k: onp.asarray(v) for k, v in base.items()}
new["rotations"] = rotations
new["ptt_coeffs"] = ptt
new["dither_rotation"] = {k: onp.asarray(v) for k, v in ROTATION.items()}
new["dither_defocus"] = {k: onp.asarray(v) for k, v in focus.items()}
onp.save(state_out, new, allow_pickle=True)
print("defocus keys:", sorted(new["defocus"]))
print("ptt keys:", sorted(new["ptt_coeffs"])[:6], "...", len(ptt))
print("written", state_out)
