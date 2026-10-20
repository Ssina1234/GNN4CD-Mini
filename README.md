# GNN4CD-Mini: Year 2007 Precipitation Downscaling

A limited-scale reproduction of the GNN4CD model, designed to demonstrate the execution of the original Graph Neural Network architecture on real GRIPHO and ERA5 data for the year 2007.

## Key Results (Best Model: Epoch 6)
- **Pearson Correlation (r)**: `0.225`
- **F1-Score (Wet/Dry)**: `0.314`
- **RMSE**: `0.539 mm/h`
- **Bias**: `-0.012 mm/h`

## Prerequisites
- Python 3.10+
- Git
- A free Copernicus Climate Data Store (CDS) account.

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/GNN4CD-Mini.git
cd GNN4CD-Mini
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

pip install -r requirements.txt