#!/usr/bin/env python3
"""
LSD_PDF_analysis_CALISTA_Landsat9_FINAL.py

Version compatible avec la nouvelle architecture de données l1c :
→ PDF empirique (pas gaussienne)
→ Ref. Merchant et al. (2005)
"""

import os
import glob
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================================
# 1) PATHS
# ==========================================================
FILE_PATTERN = "CALISTA_LC09_L1C_*_02_T1.nc"

#IN_ROOT = "/home/kcorrea/Trishna/data/l1c/"
IN_ROOT = "/home/datawork-cersat-public/cache/project/calista/data/landsat-9/l1c/v1.0/2024/"

#OUT_DIR = IN_ROOT
OUT_DIR = "/home/datawork-cersat-public/cache/users/trishna/processing_KC/result/LUT_pdfLSD/"
os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================================
# 2) PARAMÈTRES
# ==========================================================

LSD_WINDOW = 9
bins = np.linspace(0, 3, 120)  # élargi si possible (important)
centers = 0.5 * (bins[:-1] + bins[1:])

# ==========================================================
# 3) FILE LIST
# ==========================================================

base_pattern = os.path.join(IN_ROOT, "0*", "**", FILE_PATTERN)
#files = sorted(glob.glob(os.path.join(IN_ROOT, "**", FILE_PATTERN), recursive=True))
files = sorted(glob.glob(base_pattern, recursive=True))
print(f"{len(files)} fichiers détectés")

global_cloud = []
global_clear = []

# ==========================================================
# 4) LOOP
# ==========================================================

for f in files:
    print("Processing:", os.path.basename(f))

    try:
        ds = xr.open_dataset(f)
    except:
        continue

    # -------------------------
    # VARIABLES
    # -------------------------
    bt11 = ds["brightness_temperature_11"].values
    landmask = ds["land_mask"].values
    cloud = ds["cloud_mask"].values

    # -------------------------
    # MASQUES
    # -------------------------
    ocean = landmask == 0
    cloud_mask = (cloud > 0) & ocean
    clear_mask = (cloud == 0) & ocean

    # -------------------------
    # CLEAN DATA
    # -------------------------
    bt11 = np.where(ocean, bt11, np.nan)

    # filtrage physique (important)
    bt11[(bt11 < 260) | (bt11 > 320)] = np.nan

    # -------------------------
    # LSD
    # -------------------------
    LSD = generic_filter(bt11, np.nanstd, size=LSD_WINDOW)

    # -------------------------
    # EXTRACTION
    # -------------------------
    LSD_cloud = LSD[cloud_mask]
    LSD_clear = LSD[clear_mask]

    LSD_cloud = LSD_cloud[np.isfinite(LSD_cloud)]
    LSD_clear = LSD_clear[np.isfinite(LSD_clear)]

    if len(LSD_cloud) < 100 or len(LSD_clear) < 100:
        continue

    global_cloud.append(LSD_cloud)
    global_clear.append(LSD_clear)

    ds.close()

# ==========================================================
# 5) CONCAT
# ==========================================================

cloud_all = np.concatenate(global_cloud)
clear_all = np.concatenate(global_clear)

print("Pixels cloud:", len(cloud_all))
print("Pixels clear:", len(clear_all))

# ==========================================================
# 6) PDF EMPIRIQUE
# ==========================================================

pdf_cloud, _ = np.histogram(cloud_all, bins=bins, density=True)
pdf_clear, _ = np.histogram(clear_all, bins=bins, density=True)

# normalisation sécurité
pdf_cloud /= (np.sum(pdf_cloud) + 1e-12)
pdf_clear /= (np.sum(pdf_clear) + 1e-12)

# ==========================================================
# 7) POSTERIOR BAYESIEN
# ==========================================================

P_cloud = pdf_cloud / (pdf_cloud + pdf_clear + 1e-12)

# ==========================================================
# 8) SAVE NETCDF
# ==========================================================

ds_out = xr.Dataset({
    "pdf_clear": (["sigma"], pdf_clear),
    "pdf_cloud": (["sigma"], pdf_cloud),
    "P_cloud_given_sigma": (["sigma"], P_cloud),
})

ds_out.coords["sigma"] = centers

ds_out.attrs.update({
    "title": "Empirical Bayesian LSD LUT",
    "method": "Histogram (non-parametric)",
    "sensor": "Landsat-9",
    "window": LSD_WINDOW,
})

out_file = os.path.join(OUT_DIR, "LSD_LUT_EMPIRICAL_CALISTA_Y2024_0.nc")
ds_out.to_netcdf(out_file)

print("LUT saved:", out_file)

# ==========================================================
# 9) PLOT
# ==========================================================

plt.figure(figsize=(8,6))
plt.plot(centers, pdf_clear, label="Clear", lw=2)
plt.plot(centers, pdf_cloud, label="Cloud", lw=2)
plt.plot(centers, P_cloud, "--", label="P(Cloud|σ)", lw=2)

#plt.xlabel("LSD (K)")
plt.xlabel("LSD (BT 10.8µm) [K]")
plt.ylabel("PDF / Probability")
plt.title("LUT (2024 – Ocean Only)")
plt.legend()
plt.grid()

fig_file = os.path.join(OUT_DIR, "LSD_PDF_empirical_Y2024_0.png")
plt.savefig(fig_file)
plt.close()

print("Figure saved:", fig_file)
