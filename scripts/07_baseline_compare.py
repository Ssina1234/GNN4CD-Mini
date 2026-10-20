from pathlib import Path
import pickle
import numpy as np
BASE = Path(__file__).resolve().parent.parent
PREP = BASE / "preprocessed"
OUT = BASE / "results" / "year_2007"

# --- Load data ---
with (PREP / "target.pkl").open("rb") as f:
    target = pickle.load(f).numpy()  # (88, 8760)

with (OUT / "G_predictions.pkl").open("rb") as f:
    res = pickle.load(f)
pred_gnn = res["pr_gnn4cd"]  # (88, 876)
obs_test = res["pr_gripho"]  # (88, 876)

# test indices: from the training script, test starts at idx 7884 to 8759
TEST_START = 7884
TEST_END = 8760

print("=" * 80)
print("BASELINE COMPARISON")
print("=" * 80)
print(f"Test set: {pred_gnn.shape} (nodes x hours)")
print(f"Test period: hours {TEST_START} to {TEST_END-1}")
print()

# --- Baselines ---

# 1. Zero predictor (always predict 0)
pred_zero = np.zeros_like(obs_test)

# 2. Climatology: training mean per node
train_data = target[:, :TEST_START]  # (88, 7884)
clim_per_node = np.nanmean(train_data, axis=1, keepdims=True)  # (88, 1)
pred_clim = np.broadcast_to(clim_per_node, obs_test.shape).copy()

# 3. Global mean (one number for all)
global_mean = np.nanmean(train_data)
pred_global = np.full_like(obs_test, global_mean)

# 4. Persistence: previous hour value
# For test hour t, use target at hour (t-1)
pred_persist = np.zeros_like(obs_test)
for j in range(obs_test.shape[1]):
    t = TEST_START + j
    if t - 1 >= 0:
        pred_persist[:, j] = target[:, t - 1]
    else:
        pred_persist[:, j] = np.nan

# 5. Daily mean (climatology per hour-of-day per node)
# Compute mean per hour-of-day (0-23) over training
train_hours = np.arange(TEST_START) % 24
pred_diurnal = np.zeros_like(obs_test)
for j in range(obs_test.shape[1]):
    hod = (TEST_START + j) % 24
    mask = train_hours == hod
    if mask.sum() > 0:
        pred_diurnal[:, j] = np.nanmean(train_data[:, mask], axis=1)
    else:
        pred_diurnal[:, j] = 0

# 6. Spatial mean (each hour: average over all 88 nodes, broadcast back)
pred_spatial = np.tile(np.nanmean(obs_test, axis=0, keepdims=True),
                       (obs_test.shape[0], 1))

# --- Evaluation ---
def evaluate(name, pred, obs):
    finite = np.isfinite(pred) & np.isfinite(obs)
    p = pred[finite]
    o = obs[finite]
    if len(p) < 10:
        return None

    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    mae = float(np.mean(np.abs(p - o)))
    bias = float(np.mean(p - o))

    # Pearson r
    if p.std() > 1e-8 and o.std() > 1e-8:
        r = float(np.corrcoef(p, o)[0, 1])
    else:
        r = 0.0

    # Dry/wet at 0.1 mm/h
    obs_wet = o >= 0.1
    pred_wet = p >= 0.1
    tp = int((obs_wet & pred_wet).sum())
    fp = int((~obs_wet & pred_wet).sum())
    fn = int((obs_wet & ~pred_wet).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Heavy rain (p99)
    p99_thr = np.percentile(o, 99)
    mask_heavy = o >= p99_thr
    if mask_heavy.sum() > 10 and p[mask_heavy].std() > 1e-8 and o[mask_heavy].std() > 1e-8:
        r_heavy = float(np.corrcoef(p[mask_heavy], o[mask_heavy])[0, 1])
    else:
        r_heavy = float("nan")

    return {
        "name": name, "rmse": rmse, "mae": mae, "bias": bias,
        "r": r, "precision": precision, "recall": recall, "f1": f1,
        "r_heavy": r_heavy,
    }

results = []
results.append(evaluate("GNN4CD (trained)", pred_gnn, obs_test))
results.append(evaluate("Zero (always 0)", pred_zero, obs_test))
results.append(evaluate("Climatology (train mean per node)", pred_clim, obs_test))
results.append(evaluate("Global mean (one number)", pred_global, obs_test))
results.append(evaluate("Persistence (prev hour)", pred_persist, obs_test))
results.append(evaluate("Diurnal cycle (hour-of-day mean)", pred_diurnal, obs_test))
results.append(evaluate("Spatial mean (broadcast)", pred_spatial, obs_test))

# --- Print table ---
print(f"{'Method':<40} {'RMSE':>8} {'MAE':>8} {'Bias':>8} {'r':>7} {'P':>7} {'R':>7} {'F1':>7} {'r_p99':>7}")
print("-" * 115)
for r in results:
    if r is None:
        continue
    print(f"{r['name']:<40} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['bias']:>+8.4f} "
          f"{r['r']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f} "
          f"{r['f1']:>7.4f} {r['r_heavy']:>7.4f}")

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("Column meanings:")
print("  RMSE    : root mean squared error (mm/h) — lower is better")
print("  MAE     : mean absolute error (mm/h) — lower is better")
print("  Bias    : mean(pred - obs) — closer to 0 is better")
print("  r       : Pearson correlation — higher is better")
print("  P       : Precision on wet hours (>= 0.1 mm/h) — higher is better")
print("  R       : Recall on wet hours — higher is better")
print("  F1      : Harmonic mean of P and R — higher is better")
print("  r_p99   : Correlation on heavy-rain (top 1%) points — higher is better")
print()

# --- Best methods ---
valid = [r for r in results if r is not None]
best_rmse = min(valid, key=lambda x: x["rmse"])
best_r    = max(valid, key=lambda x: x["r"])
best_f1   = max(valid, key=lambda x: x["f1"])
best_heavy = max(valid, key=lambda x: (x["r_heavy"] if not np.isnan(x["r_heavy"]) else -1))

print(f"Best RMSE:            {best_rmse['name']} ({best_rmse['rmse']:.4f})")
print(f"Best Pearson r:       {best_r['name']} ({best_r['r']:.4f})")
print(f"Best F1 (wet/dry):    {best_f1['name']} ({best_f1['f1']:.4f})")
print(f"Best r on heavy rain: {best_heavy['name']} ({best_heavy['r_heavy']:.4f})")

# --- Does GNN4CD beat simple baselines? ---
gnn = next(r for r in valid if "GNN4CD" in r["name"])
print()
print("=" * 80)
print("DOES GNN4CD BEAT THE BASELINES?")
print("=" * 80)
for r in valid:
    if r["name"] == "GNN4CD (trained)":
        continue
    delta_rmse = gnn["rmse"] - r["rmse"]
    delta_r = gnn["r"] - r["r"]
    delta_f1 = gnn["f1"] - r["f1"]
    verdict_rmse = "WIN" if delta_rmse < 0 else "LOSE"
    verdict_r = "WIN" if delta_r > 0 else "LOSE"
    verdict_f1 = "WIN" if delta_f1 > 0 else "LOSE"
    print(f"vs {r['name']:<40}")
    print(f"   RMSE: {delta_rmse:+.4f} ({verdict_rmse})   "
          f"r: {delta_r:+.4f} ({verdict_r})   "
          f"F1: {delta_f1:+.4f} ({verdict_f1})")