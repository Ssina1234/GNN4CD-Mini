from pathlib import Path
import xarray as xr

SRC = Path(r"D:\Sina\GNN4CD-main\mini_run\data_raw\gripho-v1_1h_TSmin30pct_2001-2016_10km.nc")
DST = Path(r"D:\Sina\GNN4CD-main\year_2007\data_raw\gripho_2007.nc")
YEAR = 2007

if not SRC.exists():
    raise FileNotFoundError(f"Source not found: {SRC}")

DST.parent.mkdir(parents=True, exist_ok=True)

with xr.open_dataset(SRC, engine="netcdf4") as ds:
    ds_year = ds.sel(time=slice(f"{YEAR}-01-01", f"{YEAR}-12-31T23:59:59"))
    print(f"Year {YEAR}")
    print(f"  steps: {len(ds_year['time'])}")
    print(f"  first: {ds_year['time'].values[0]}")
    print(f"  last:  {ds_year['time'].values[-1]}")
    print(f"  pr: {tuple(ds_year['pr'].shape)}")
    print(f"  unit: {ds_year['pr'].attrs.get('units', 'unknown')}")
    print(f"  max: {float(ds_year['pr'].max()):.2f} mm/h")
    print(f"  valid %: {float(ds_year['pr'].notnull().mean())*100:.2f}%")

    ds_year[["pr"]].to_netcdf(DST, engine="netcdf4")

print(f"\nSaved: {DST}")
print(f"Size:  {DST.stat().st_size / 1e6:.1f} MB")