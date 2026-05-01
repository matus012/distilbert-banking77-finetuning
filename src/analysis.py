"""Cross-run aggregation helpers for analysis notebook."""
import json
import os
import re

import pandas as pd


def load_all_runs(experiments_dir: str = "experiments") -> pd.DataFrame:
    """Scan experiment subdirs, return one row per run with config + test metrics."""
    rows = []
    for name in sorted(os.listdir(experiments_dir)):
        run_dir = os.path.join(experiments_dir, name)
        if not os.path.isdir(run_dir) or name.startswith("_"):
            continue
        config_path = os.path.join(run_dir, "config.json")
        metrics_path = os.path.join(run_dir, "test_metrics.json")
        csv_path = os.path.join(run_dir, "metrics.csv")
        log_path = os.path.join(run_dir, "training_log.txt")
        if not (os.path.exists(config_path) and os.path.exists(metrics_path)):
            continue

        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        with open(metrics_path, encoding="utf-8") as f:
            tm = json.load(f)

        # best val_macro_f1 from training curves
        val_macro_f1 = float("nan")
        if os.path.exists(csv_path):
            df_csv = pd.read_csv(csv_path)
            if "val_macro_f1" in df_csv.columns and len(df_csv) > 0:
                val_macro_f1 = float(df_csv["val_macro_f1"].max())

        # wall clock from training_log.txt
        wall_clock = float("nan")
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    m = re.search(r"Training done in ([\d.]+)s", line)
                    if m:
                        wall_clock = float(m.group(1))

        rows.append({
            "run_name": cfg.get("run_name", name),
            "model_name": cfg.get("model_name", ""),
            "learning_rate": cfg.get("learning_rate", float("nan")),
            "batch_size": cfg.get("per_device_train_batch_size", float("nan")),
            "num_epochs": cfg.get("num_train_epochs", float("nan")),
            "freeze_strategy": cfg.get("freeze_strategy", "none"),
            "val_macro_f1": val_macro_f1,
            "test_acc": tm.get("accuracy", float("nan")),
            "test_macro_f1": tm.get("macro_f1", float("nan")),
            "test_weighted_f1": tm.get("weighted_f1", float("nan")),
            "wall_clock_seconds": wall_clock,
        })

    return pd.DataFrame(rows)


def load_training_curves(run_name: str, experiments_dir: str = "experiments") -> pd.DataFrame:
    """Read metrics.csv for a run, return epoch-level training stats."""
    csv_path = os.path.join(experiments_dir, run_name, "metrics.csv")
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "val_loss": "eval_loss",
        "val_acc": "eval_accuracy",
        "val_macro_f1": "eval_macro_f1",
    })
    return df


def load_per_class_f1(run_name: str, experiments_dir: str = "experiments") -> pd.DataFrame:
    """Parse classification_report.txt, return per-class precision/recall/f1/support."""
    report_path = os.path.join(experiments_dir, run_name, "classification_report.txt")
    skip_names = {"accuracy", "macro avg", "weighted avg", ""}
    rows = []
    with open(report_path, encoding="utf-8") as f:
        for line in f:
            # skip header line (contains "precision    recall")
            if "precision" in line and "recall" in line:
                continue
            stripped = line.strip()
            if not stripped:
                continue
            # lines like: "  class_name     0.xxxx    0.xxxx    0.xxxx   NNN"
            parts = stripped.rsplit(maxsplit=4)
            if len(parts) < 5:
                continue
            class_name = parts[0].strip()
            if class_name in skip_names:
                continue
            try:
                precision, recall, f1, support = (
                    float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4])
                )
            except (ValueError, IndexError):
                continue
            rows.append({
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            })
    return pd.DataFrame(rows)
