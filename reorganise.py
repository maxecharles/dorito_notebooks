# %%
import os
import jax.numpy as np

from socket import gethostname

if gethostname() == "glinton":
    morgana = "/media/morgana1/"
else:
    morgana = "/Volumes/morgana1/"

amigo_cache = os.path.join(morgana, "snert/max/data/amigo_files/")
output_path = os.path.join(amigo_cache, "outputs/PDS70/")

# os.mkdir(os.path.join(output_path, "gridded"))
gridded_path = os.path.join(output_path, "gridded")

# %%
for d in os.listdir(output_path):
    if d.endswith("_groups"):
        print(d)

        path = os.path.join(output_path, d)

        try:
            discos = np.load(os.path.join(path, "discos.npy"), allow_pickle=True).item()
        except FileNotFoundError:
            print(f"File not found in {path}")
            continue
        n_groups = d.split("_")[0]
        np.save(os.path.join(gridded_path, f"{n_groups}_discos.npy"), discos)
        print(f"Saved {n_groups}_discos.npy to {gridded_path}")


# %%
