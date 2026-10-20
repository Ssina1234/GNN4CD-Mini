from pathlib import Path
import pickle
import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "year_2007"

with (OUT / "G_predictions.pkl").open("rb") as f:
    res = pickle.load(f)

pred = res["pr_gnn4cd"]   # (88, 876)
obs  = res["pr_gripho"]   # (88, 876)

finite = np.isfinite(pred) & np.isfinite(obs)
pred_f = pred[finite]
obs_f  = obs[finite]

print("=" * 70)
print("OVERALL STATISTICS")
print("=" * 70)
print(f"  Total points:       {finite.sum():,}")
print(f"  Pred range:         {pred_f.min():.4f} .. {pred_f.max():.4f}")
print(f"  Obs range:          {obs_f.min():.4f} .. {obs_f.max():.4f}")
print(f"  Pred mean:          {pred_f.mean():.4f}")
print(f"  Obs mean:           {obs_f.mean():.4f}")
print(f"  Pred std:           {pred_f.std():.4f}")
print(f"  Obs std:            {obs_f.std():.4f}")

rmse = np.sqrt(np.mean((pred_f - obs_f) ** 2))
mae = np.mean(np.abs(pred_f - obs_f))
bias = np.mean(pred_f - obs_f)
corr = np.corrcoef(pred_f, obs_f)[0, 1]

print(f"\n  RMSE:               {rmse:.4f} mm/h")
print(f"  MAE:                {mae:.4f} mm/h")
print(f"  Bias:               {bias:+.4f} mm/h")
print(f"  Pearson r:          {corr:.4f}")

# Correlation per spatial sample (across time)
print("\n" + "=" * 70)
print("SPATIAL CORRELATION (per grid point, across all 876 hours)")
print("=" * 70)
corr_per_node = []
for k in range(pred.shape[0]):
    p = pred[k]
    o = obs[k]
    m = np.isfinite(p) & np.isfinite(o)
    if m.sum() > 10 and p[m].std() > 1e-6 and o[m].std() > 1e-6:
        corr_per_node.append(np.corrcoef(p[m], o[m])[0, 1])

corr_per_node = np.array(corr_per_node)
print(f"  Nodes evaluated:    {len(corr_per_node)}/88")
print(f"  Mean corr:          {corr_per_node.mean():.4f}")
print(f"  Median corr:        {np.median(corr_per_node):.4f}")
print(f"  Min/Max corr:       {corr_per_node.min():.4f} / {corr_per_node.max():.4f}")

# Percentile correlations (paper-style)
print("\n" + "=" * 70)
print("PAPER-STYLE PERCENTILE CORRELATIONS")
print("=" * 70)

# میانگین فضایی در هر ساعت
pred_mean_time = pred.mean(axis=0)   # (876,)
obs_mean_time  = obs.mean(axis=0)

m = np.isfinite(pred_mean_time) & np.isfinite(obs_mean_time)
print(f"  Average corr (spatial mean over time): "
      f"{np.corrcoef(pred_mean_time[m], obs_mean_time[m])[0,1]:.4f}")

# برای p99 و p99.9: نقشه‌ی میانگین زمانی در فضای گره‌ها
pred_mean_space = np.nanmean(pred, axis=1)  # (88,)
obs_mean_space  = np.nanmean(obs, axis=1)

m2 = np.isfinite(pred_mean_space) & np.isfinite(obs_mean_space)
print(f"  Mean-map corr (across 88 nodes): "
      f"{np.corrcoef(pred_mean_space[m2], obs_mean_space[m2])[0,1]:.4f}")

# p99 روی فضای زمانی-مکانی
p99_obs = np.percentile(obs_f, 99)
p99_pred = np.percentile(pred_f, 99)
print(f"\n  Obs p99 threshold:  {p99_obs:.4f} mm/h")
print(f"  Pred p99 value:     {p99_pred:.4f} mm/h")

mask_heavy = obs_f >= p99_obs
if mask_heavy.sum() > 10:
    print(f"  Corr on heavy-rain points (obs >= p99): "
          f"{np.corrcoef(pred_f[mask_heavy], obs_f[mask_heavy])[0,1]:.4f}")
    print(f"  RMSE on heavy-rain points: "
          f"{np.sqrt(np.mean((pred_f[mask_heavy] - obs_f[mask_heavy])**2)):.4f} mm/h")

# dry/wet discrimination
print("\n" + "=" * 70)
print("DRY/WET CLASSIFICATION")
print("=" * 70)
for thr in [0.1, 0.5, 1.0, 2.0]:
    obs_wet = obs_f >= thr
    pred_wet = pred_f >= thr
    tp = (obs_wet & pred_wet).sum()
    fp = (~obs_wet & pred_wet).sum()
    fn = (obs_wet & ~pred_wet).sum()
    tn = (~obs_wet & ~pred_wet).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"  Threshold {thr:.1f} mm/h: "
          f"P={precision:.3f}, R={recall:.3f}, TP={tp}, FP={fp}, FN={fn}")

print("=" * 70)