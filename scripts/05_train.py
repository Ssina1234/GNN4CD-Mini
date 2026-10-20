from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, r"D:\Sina\GNN4CD-main")

import csv
import os
import pickle
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader

import dataset
from dataset import Dataset_Graph, Iterable_Graph
from models.GNN4CD_model import GNN4CD_model
from utils.tools import compute_input_statistics, prepare_target, standardize_input
from utils.train_test import Tester

torch.set_num_threads(4)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

BASE = Path(__file__).resolve().parent.parent
RUN = BASE 
PREP = BASE / "data_raw" / "preprocessed"
GRAPH_FILE = PREP / "low_high_graph.pkl"
TARGET_FILE = PREP / "target.pkl"
GRIPHO_FILE = BASE / "data_raw" / "gripho_2007.nc"
OUTPUT_DIR = BASE / "results" / "year_2007"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CHECKPOINT = OUTPUT_DIR / "checkpoint_2007.pt"
OUTPUT_PREDICTIONS = OUTPUT_DIR / "G_predictions.pkl"
OUTPUT_LOG = OUTPUT_DIR / "training.log"

SEED = 42
SEQ_L = 24
EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 10
GRAD_CLIP = 5.0

QMSE_ALPHA = 0.1
QMSE_THRESHOLD = 1.0
QMSE_WEIGHT_FACTOR = 2.0

# Blocked random split
BLOCK_SIZE = 30           # hours per block
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
# TEST_FRAC = 0.10


def log(msg):
    print(msg, flush=True)
    with OUTPUT_LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def qmse_loss(pred, target, alpha=QMSE_ALPHA, threshold=QMSE_THRESHOLD, weight_factor=QMSE_WEIGHT_FACTOR):
    diff2 = (pred - target) ** 2
    weights = 1.0 + alpha * (target >= threshold).float() * weight_factor
    return (weights * diff2).mean()


def blocked_random_split(time_count, seq_l, block_size=30, train_frac=0.8, val_frac=0.1, seed=42):
    """
    Split indices into train/val/test by shuffling blocks of consecutive hours.
    Avoids temporal leakage while preserving seasonal diversity.
    """
    rng = np.random.RandomState(seed)

    # Valid indices start at seq_l (need 25 input hours)
    valid_start = seq_l
    valid_end = time_count
    n_hours = valid_end - valid_start

    # Number of blocks
    n_blocks = (n_hours + block_size - 1) // block_size

    # Shuffle block order
    block_order = rng.permutation(n_blocks)

    # Assign blocks to splits
    n_train_blocks = int(n_blocks * train_frac)
    n_val_blocks = int(n_blocks * val_frac)

    train_blocks = set(block_order[:n_train_blocks])
    val_blocks = set(block_order[n_train_blocks:n_train_blocks + n_val_blocks])
    test_blocks = set(block_order[n_train_blocks + n_val_blocks:])

    train_idx, val_idx, test_idx = [], [], []

    for b in range(n_blocks):
        start = valid_start + b * block_size
        end = min(start + block_size, valid_end)
        idxs = list(range(start, end))
        if b in train_blocks:
            train_idx.extend(idxs)
        elif b in val_blocks:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    return (
        torch.tensor(sorted(train_idx), dtype=torch.long),
        torch.tensor(sorted(val_idx), dtype=torch.long),
        torch.tensor(sorted(test_idx), dtype=torch.long),
    )


def main():
    OUTPUT_LOG.write_text("", encoding="utf-8")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    log("[1/6] Loading graph and target ...")
    with GRAPH_FILE.open("rb") as f:
        graph = pickle.load(f)
    with TARGET_FILE.open("rb") as f:
        raw_target = pickle.load(f)
    if not torch.is_tensor(raw_target):
        raw_target = torch.as_tensor(raw_target)
    raw_target = raw_target.float()

    time_count = int(raw_target.shape[1])
    high_nodes = int(raw_target.shape[0])
    log(f"      target shape: {tuple(raw_target.shape)}")
    log(f"      low.x shape:  {tuple(graph['low'].x.shape)}")
    log(f"      high.x shape: {tuple(graph['high'].x.shape)}")

    # ============================================================
    # BLOCKED RANDOM SPLIT
    # ============================================================
    log(f"\n      Using BLOCKED RANDOM SPLIT (block_size={BLOCK_SIZE})")
    train_idxs, val_idxs, test_idxs = blocked_random_split(
        time_count, SEQ_L, block_size=BLOCK_SIZE,
        train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=SEED,
    )

    log(f"      train: {len(train_idxs)} samples")
    log(f"      val:   {len(val_idxs)} samples")
    log(f"      test:  {len(test_idxs)} samples")

    # Show that test spans different months
    def month_distribution(idxs):
        months = (idxs.numpy() // 730).clip(0, 11)
        counts = np.bincount(months, minlength=12)
        return counts

    log(f"      train month distribution: {month_distribution(train_idxs).tolist()}")
    log(f"      val   month distribution: {month_distribution(val_idxs).tolist()}")
    log(f"      test  month distribution: {month_distribution(test_idxs).tolist()}")

    target_train = prepare_target(raw_target.clone(), "Rall", threshold=0.1)

    train_valid = [int(i) for i in train_idxs if torch.isfinite(target_train[:, int(i)]).any()]
    train_idxs = torch.tensor(train_valid, dtype=torch.long)
    log(f"      train after NaN filter: {len(train_idxs)}")

    # ============================================================
    log("\n[2/6] Computing normalization statistics ...")
    # ============================================================
    args_stats = SimpleNamespace(output_path=str(PREP) + os.sep, log_file="norm.log")
    means_low, stds_low, means_high, stds_high = compute_input_statistics(
        graph["low"].x[:, train_idxs, :, :],
        graph["high"].x,
        args_stats,
        accelerator=None,
    )
    means_low = np.asarray(means_low, dtype=np.float32)
    stds_low = np.asarray(stds_low, dtype=np.float32)
    stds_low[~np.isfinite(stds_low) | (np.abs(stds_low) < 1e-8)] = 1.0

    means_high = torch.as_tensor(means_high, dtype=torch.float32)
    stds_high = torch.as_tensor(stds_high, dtype=torch.float32)
    stds_high = torch.where(
        torch.isfinite(stds_high) & (stds_high.abs() >= 1e-8),
        stds_high, torch.ones_like(stds_high),
    )
    for name, obj in [("means_low", means_low), ("stds_low", stds_low),
                      ("means_high", means_high), ("stds_high", stds_high)]:
        with (PREP / f"{name}.pkl").open("wb") as f:
            pickle.dump(obj, f)

    graph["low"].x, graph["high"].x = standardize_input(
        graph["low"].x, graph["high"].x,
        means_low, stds_low, means_high, stds_high,
        args_stats, accelerator=None,
    )
    graph["low"].x = torch.flatten(graph["low"].x, start_dim=2, end_dim=-1)

    if not torch.isfinite(graph["low"].x).all():
        raise RuntimeError("Non-finite values in low.x")
    if not torch.isfinite(graph["high"].x).all():
        raise RuntimeError("Non-finite values in high.x")

    # ============================================================
    log("\n[3/6] Building dataset ...")
    # ============================================================
    weights = torch.ones_like(target_train)
    graph_dataset = Dataset_Graph(
        graph=graph, targets=target_train, model_name="GNN4CD_model",
        seq_l=SEQ_L, w=weights,
    )
    collate = dataset.custom_collate_fn_graph

    train_sampler = Iterable_Graph(
        dataset_graph=graph_dataset, shuffle=True, idxs_vector=train_idxs,
    )
    train_loader = DataLoader(
        graph_dataset, batch_size=1, num_workers=0,
        sampler=train_sampler, collate_fn=collate,
    )
    val_loader = DataLoader(
        graph_dataset, batch_size=1, num_workers=0,
        sampler=val_idxs.tolist(), collate_fn=collate,
    )
    test_loader = DataLoader(
        graph_dataset, batch_size=1, num_workers=0,
        sampler=test_idxs.tolist(), collate_fn=collate,
    )

    # ============================================================
    log("\n[4/6] Initializing model ...")
    # ============================================================
    model = GNN4CD_model(seq_l=SEQ_L + 1, high_in=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"      trainable params: {n_params:,}")

    # ============================================================
    log(f"\n[5/6] Training for {EPOCHS} epochs (patience={PATIENCE}) ...")
    # ============================================================
    best_val = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        losses = []
        train_iter = iter(train_loader)
        num_batches = len(train_loader)
        log(f"\n      Epoch {epoch+1}/{EPOCHS} ({num_batches} batches)")

        for step in range(1, num_batches + 1):
            batch = next(train_iter)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch).squeeze(-1)
            target = batch["high"].y
            mask = torch.isfinite(target)
            if not mask.any():
                continue
            loss = qmse_loss(pred[mask], target[mask])
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 500 == 0 or step == num_batches:
                log(f"        step {step}/{num_batches}  loss={losses[-1]:.6f}")

        train_loss = float(np.mean(losses)) if losses else float("nan")

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch).squeeze(-1)
                target = batch["high"].y
                mask = torch.isfinite(target)
                if mask.any():
                    val_losses.append(float(qmse_loss(pred[mask], target[mask])))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        log(f"      Epoch {epoch+1} ({elapsed/60:.1f} min): "
            f"train={train_loss:.6f}, val={val_loss:.6f}, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_class": "models.GNN4CD_model.GNN4CD_model",
                "seq_l": SEQ_L + 1, "high_in": 1,
                "epoch": epoch + 1,
                "train_loss": train_loss, "val_loss": val_loss,
                "seed": SEED,
            }, OUTPUT_CHECKPOINT)
            log(f"        [BEST] checkpoint saved (val={val_loss:.6f})")
        else:
            patience_counter += 1
            log(f"        no improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                log(f"        [EARLY STOP]")
                break

    log(f"\n      Best epoch: {best_epoch}/{EPOCHS}, best val: {best_val:.6f}")

    # ============================================================
    log("\n[6/6] Loading best model and running Tester ...")
    # ============================================================
    ckpt = torch.load(OUTPUT_CHECKPOINT, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_args = SimpleNamespace(
        model_type="Rall",
        output_path=str(OUTPUT_DIR) + os.sep,
        log_file="tester.log",
    )
    pred_tensor, _ = Tester().test(model, test_loader, args=test_args, accelerator=None)
    predictions = np.squeeze(pred_tensor.detach().cpu().numpy(), axis=-1).T

    lon_high = graph["high"].lon.numpy()
    lat_high = graph["high"].lat.numpy()
    observed = raw_target[:, test_idxs].numpy()

    with xr.open_dataset(GRIPHO_FILE, engine="netcdf4", decode_times=True) as gds:
        physical_times = gds["time"].values
    test_times = physical_times[test_idxs.numpy()]

    finite = np.isfinite(predictions) & np.isfinite(observed)
    rmse = float(np.sqrt(np.mean((predictions[finite] - observed[finite]) ** 2)))
    mae = float(np.mean(np.abs(predictions[finite] - observed[finite])))
    bias = float(np.mean(predictions[finite] - observed[finite]))
    r = float(np.corrcoef(predictions[finite], observed[finite])[0, 1])

    log("\n" + "=" * 60)
    log("FINAL RESULTS (v4 - blocked random split)")
    log("=" * 60)
    log(f"  predictions shape: {predictions.shape}")
    log(f"  prediction range:  {np.nanmin(predictions):.4f} .. {np.nanmax(predictions):.4f}")
    log(f"  observed range:    {np.nanmin(observed):.4f} .. {np.nanmax(observed):.4f}")
    log(f"  RMSE:              {rmse:.4f} mm/h")
    log(f"  MAE:               {mae:.4f} mm/h")
    log(f"  Bias:              {bias:+.4f} mm/h")
    log(f"  Pearson r:         {r:.4f}")
    log(f"  best epoch:        {best_epoch}/{EPOCHS}")
    log("=" * 60)

    result = {
        "lon": lon_high, "lat": lat_high, "times": test_times,
        "pr_gripho": observed, "pr_gnn4cd": predictions,
        "model_type": "Rall", "rmse": rmse, "pearson_r": r,
        "note": "Year 2007 GNN4CD v4 (blocked random split).",
    }
    with OUTPUT_PREDICTIONS.open("wb") as f:
        pickle.dump(result, f)
    log(f"\nSaved: {OUTPUT_PREDICTIONS}")


if __name__ == "__main__":
    main()