import cdsapi
import argparse
from pathlib import Path
import time

def download_era5(year: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    variables = {
        "q": "specific_humidity",
        "t": "temperature",
        "u": "u_component_of_wind",
        "v": "v_component_of_wind",
        "z": "geopotential",
    }
    levels = ["200", "500", "700", "850", "1000"]
    area = [47.25, 7.25, 42.75, 11.75]  # North, West, South, East

    for short, long in variables.items():
        for month in range(1, 13):
            month_str = f"{month:02d}"
            target = output_dir / f"{short}_{year}_{month_str}.nc"
            
            if target.exists():
                print(f"[SKIP] {target.name}")
                continue
                
            print(f"[DOWNLOAD] {short} - {year}-{month_str} ...")
            try:
                client.retrieve(
                    "reanalysis-era5-pressure-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": [long],
                        "pressure_level": levels,
                        "year": year,
                        "month": month_str,
                        "day": [f"{d:02d}" for d in range(1, 32)],
                        "time": [f"{h:02d}:00" for h in range(24)],
                        "area": area,
                        "format": "netcdf",
                        "grid": [0.25, 0.25],
                    },
                    str(target),
                )
                print(f"  Saved: {target.name}")
                time.sleep(2) # جلوگیری از محدودیت نرخ درخواست
            except Exception as e:
                print(f"  Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2007", help="Year to download")
    parser.add_argument("--output_dir", default="data_raw/era5", help="Output directory")
    args = parser.parse_args()
    
    # مسیرها رو نسبت به ریشه پروژه تنظیم می‌کنیم
    base_dir = Path(__file__).resolve().parent.parent
    out_path = base_dir / args.output_dir
    
    download_era5(args.year, out_path)