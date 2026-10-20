from pathlib import Path
import pickle
import numpy as np
import torch
import xarray as xr
from torch_geometric.data import HeteroData

BASE = Path(r"D:\Sina\GNN4CD-main")
MINI_PREP = BASE / "mini_run" / "preprocessed"
YEAR_DIR = BASE / "year_2007"
ERA5_DIR = YEAR_DIR / "data_raw" / "era5"
GRIPHO_FILE = YEAR_DIR / "data_raw" / "gripho_2007.nc"
OUT_DIR = YEAR_DIR / "preprocessed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. Load mini_run graph (coords + edges + topo) ---
print("[1/4] Loading mini_run graph for reference ...")
with (MINI_PREP / "low_high_graph.pkl").open("rb") as f:
    g_old = pickle.load(f)

high_lon = g_old["high"].lon.numpy().astype(np.float32)
high_lat = g_old["high"].lat.numpy().astype(np.float32)
low_lon = g_old["low"].lon.numpy().astype(np.float32)
low_lat = g_old["low"].lat.numpy().astype(np.float32)

print(f"      high nodes: {len(high_lon)}")
print(f"      low nodes:  {len(low_lon)}")

# --- 2. Build target (88, 8760) from GRIPHO 2007 ---
print("\n[2/4] Building target from GRIPHO 2007 ...")
with xr.open_dataset(GRIPHO_FILE, engine="netcdf4") as gds:
    pr = gds["pr"].values
    grid_lon = gds["lon"].values
    grid_lat = gds["lat"].values

print(f"      GRIPHO pr shape: {pr.shape}")

n_high = len(high_lon)
n_time = pr.shape[0]
target_np = np.zeros((n_high, n_time), dtype=np.float32)

warn_count = 0
for k in range(n_high):
    d2 = (grid_lon - high_lon[k])**2 + (grid_lat - high_lat[k])**2
    iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
    target_np[k] = pr[:, iy, ix]
    if d2[iy, ix] > 1e-4:
        warn_count += 1
        if warn_count <= 3:
            print(f"      WARN high node {k}: dist²={d2[iy, ix]:.6e}")

print(f"      target shape: {target_np.shape}")
print(f"      target range: {np.nanmin(target_np):.3f} .. {np.nanmax(target_np):.3f} mm/h")
print(f"      target valid: {np.isfinite(target_np).mean()*100:.2f}%")
print(f"      total distance warnings: {warn_count}")

with (OUT_DIR / "target.pkl").open("wb") as f:
    pickle.dump(torch.from_numpy(target_np), f)

# --- 3. Build low.x (169, 8760, 5, 5) from ERA5 2007 ---
print("\n[3/4] Building low.x from ERA5 2007 ...")
VARIABLES = ["q", "t", "u", "v", "z"]
era5 = {}
for var in VARIABLES:
    with xr.open_dataset(ERA5_DIR / f"{var}_2007.nc", engine="netcdf4") as ds:
        era5[var] = ds[var].values
        if var == "q":
            era5_lat = ds["latitude"].values
            era5_lon = ds["longitude"].values
            era5_level = ds["level"].values
    print(f"      {var}: shape={era5[var].shape}")

print(f"      levels (file order): {era5_level.tolist()}")
print(f"      lat: {era5_lat.min():.2f} .. {era5_lat.max():.2f} (n={len(era5_lat)})")
print(f"      lon: {era5_lon.min():.2f} .. {era5_lon.max():.2f} (n={len(era5_lon)})")

era5_lon_2d, era5_lat_2d = np.meshgrid(era5_lon, era5_lat)

n_low = len(low_lon)
n_var = len(VARIABLES)
n_lev = len(era5_level)

low_x_np = np.zeros((n_low, n_time, n_var, n_lev), dtype=np.float32)

warn_count = 0
for k in range(n_low):
    d2 = (era5_lon_2d - low_lon[k])**2 + (era5_lat_2d - low_lat[k])**2
    iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
    if d2[iy, ix] > 1e-6:
        warn_count += 1
        if warn_count <= 3:
            print(f"      WARN low node {k}: dist²={d2[iy, ix]:.6e}")
    for vi, var in enumerate(VARIABLES):
        # era5[var] shape: (time, level, lat, lon)
        low_x_np[k, :, vi, :] = era5[var][:, :, iy, ix]

print(f"      low.x shape: {low_x_np.shape}")
print(f"      low.x range: {low_x_np.min():.3f} .. {low_x_np.max():.3f}")
print(f"      total distance warnings: {warn_count}")

# --- 4. Assemble new HeteroData ---
print("\n[4/4] Assembling new graph ...")
g_new = HeteroData()

# High nodes: coords + topo (static, copied from mini_run)
g_new["high"].x = g_old["high"].x
g_new["high"].lon = g_old["high"].lon
g_new["high"].lat = g_old["high"].lat

# Low nodes: coords + new time series
g_new["low"].x = torch.from_numpy(low_x_np)
g_new["low"].lon = g_old["low"].lon
g_new["low"].lat = g_old["low"].lat

# Edges (unchanged)
g_new["high", "within", "high"].edge_index = g_old["high", "within", "high"].edge_index
g_new["low", "to", "high"].edge_index = g_old["low", "to", "high"].edge_index

with (OUT_DIR / "low_high_graph.pkl").open("wb") as f:
    pickle.dump(g_new, f)

print(f"\nSaved: {OUT_DIR / 'low_high_graph.pkl'}")
print(f"Saved: {OUT_DIR / 'target.pkl'}")
print("\nDONE")