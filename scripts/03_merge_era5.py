from pathlib import Path
import xarray as xr
import shutil

ERA5_DIR = Path(r"D:\Sina\GNN4CD-main\year_2007\data_raw\era5")
YEAR = "2007"
VARIABLES = ["q", "t", "u", "v", "z"]
MONTHS = [f"{m:02d}" for m in range(1, 13)]

print("=" * 70)
print(f"Merging monthly ERA5 files into yearly files for {YEAR}")
print("=" * 70)

for var in VARIABLES:
    monthly_files = sorted(ERA5_DIR / f"{var}_{YEAR}_{m}.nc" for m in MONTHS)
    missing = [f for f in monthly_files if not f.exists()]
    if missing:
        print(f"[SKIP] {var}: missing {len(missing)} months")
        continue

    out = ERA5_DIR / f"{var}_{YEAR}.nc"
    if out.exists():
        print(f"[SKIP] {out.name} already exists ({out.stat().st_size/1e6:.1f} MB)")
        continue

    print(f"\n[MERGE] {var}: {len(monthly_files)} files ...")

    # باز کردن همه فایل‌ها و ادغام
    ds = xr.open_mfdataset(monthly_files, combine="by_coords", engine="netcdf4")

    # اصلاح نام ابعاد: valid_time → time, pressure_level → level
    rename_map = {}
    if "valid_time" in ds.dims:
        rename_map["valid_time"] = "time"
    if "pressure_level" in ds.dims:
        rename_map["pressure_level"] = "level"
    if rename_map:
        ds = ds.rename(rename_map)
        print(f"  renamed: {rename_map}")

    # حذف ابعاد اضافی
    for extra in ["number", "expver"]:
        if extra in ds.coords or extra in ds.dims:
            ds = ds.drop_vars(extra, errors="ignore")
            print(f"  dropped: {extra}")

    ds.to_netcdf(out, engine="netcdf4")
    ds.close()
    print(f"  saved: {out.name} ({out.stat().st_size/1e6:.1f} MB)")

print("\n" + "=" * 70)
print("MERGE DONE")
print("=" * 70)

# بررسی نهایی
for var in VARIABLES:
    out = ERA5_DIR / f"{var}_{YEAR}.nc"
    if out.exists():
        with xr.open_dataset(out, engine="netcdf4") as ds:
            print(f"  {out.name}: dims={dict(ds.sizes)}, coords={list(ds.coords)}")