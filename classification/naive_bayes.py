from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.logistic_regression import (
    _clean_windows_extended_path,
    _collect_sample,
    _collect_sample_from_7z_archives,
    _discover_7z_archives,
    _discover_schema_files,
)


def _load_paths_from_rows_csv_long(rows_csv_path: Path) -> list[Path]:
    if not rows_csv_path.exists():
        return []
    rows_df = pd.read_csv(rows_csv_path)
    if "path" not in rows_df.columns:
        return []

    files: list[Path] = []
    for raw in rows_df["path"].astype(str):
        cleaned = raw.strip()
        if cleaned:
            files.append(Path(cleaned))
    return files


def _save_confusion_matrix(cm: np.ndarray, cm_path: Path) -> None:
    cm_df = pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])
    cm_df.to_csv(cm_path)


def _save_confusion_matrix_plot(cm: np.ndarray, plot_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred_0", "pred_1"])
    ax.set_yticks([0, 1], labels=["actual_0", "actual_1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Naive Bayes Confusion Matrix")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _save_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, plot_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="red", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Naive Bayes ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a flow-level Naive Bayes model on IEC104 CSV files.")
    parser.add_argument("--data-root", type=str, default="", help="Optional dataset root to discover 84-col CSV files.")
    parser.add_argument("--max-rows", type=int, default=120_000, help="Maximum sampled rows for training.")
    parser.add_argument("--chunk-size", type=int, default=120_000, help="Chunk size while reading CSVs.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--var-smoothing", type=float, default=1e-9, help="Portion of largest variance added for stability.")
    args = parser.parse_args()

    shared_outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir = shared_outputs_dir / "naive_bayes_outputs"
    plots_dir = outputs_dir / "plots"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows_csv_candidates = [
        shared_outputs_dir / "rows_per_file_84col.csv",
        shared_outputs_dir / "eda_outputs" / "rows_per_file_84col.csv",
    ]
    rows_csv = next((p for p in rows_csv_candidates if p.exists()), rows_csv_candidates[0])
    files = _load_paths_from_rows_csv_long(rows_csv)
    archives: list[Path] = []

    if not files:
        summary_candidates = [
            shared_outputs_dir / "eda_summary_binary_84col.json",
            shared_outputs_dir / "eda_outputs" / "eda_summary_binary_84col.json",
        ]
        summary_path = next((p for p in summary_candidates if p.exists()), summary_candidates[0])
        summary_root = None
        if summary_path.exists():
            try:
                summary_root = _clean_windows_extended_path(pd.read_json(summary_path, typ="series")["dataset_root"])
            except Exception:
                summary_root = None

        user_root = Path(args.data_root) if args.data_root else None
        discover_root = user_root if user_root and user_root.exists() else summary_root
        if discover_root is not None and discover_root.exists():
            if discover_root.is_file() and discover_root.suffix.lower() == ".7z":
                archives = [discover_root]
            elif discover_root.is_dir():
                files = _discover_schema_files(discover_root)
                if not files:
                    archives = _discover_7z_archives(discover_root)

    if not files and not archives:
        raise FileNotFoundError(
            "No source flow CSV files were found. Provide a valid dataset path with "
            "`--data-root \"C:\\path\\to\\datasets\"` (folder with CSV or .7z files) "
            "or regenerate rows_per_file_84col.csv from this machine."
        )

    if files:
        data, feature_cols, files_used = _collect_sample(
            files=files,
            max_rows=args.max_rows,
            chunk_size=args.chunk_size,
            random_state=args.random_state,
        )
        source_details = f"Files used: {files_used}"
    else:
        data, feature_cols, archives_used, csv_members_used = _collect_sample_from_7z_archives(
            archives=archives,
            max_rows=args.max_rows,
            chunk_size=args.chunk_size,
            random_state=args.random_state,
        )
        source_details = f"7z archives used: {archives_used}\nCSV members read: {csv_members_used}"

    if data.empty or not feature_cols:
        raise RuntimeError("Failed to build a training sample. Check source CSV format and numeric feature availability.")

    x = data[feature_cols]
    y = data["label_bin"].to_numpy()
    if np.unique(y).size < 2:
        raise RuntimeError("Only one class is present in sampled data. Increase max rows or adjust the dataset source.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=args.random_state, stratify=y
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB(var_smoothing=args.var_smoothing)),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    model_log_loss = float(log_loss(y_test, np.clip(y_prob, 1e-15, 1 - 1e-15)))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    _save_roc_curve(fpr, tpr, roc_auc, plots_dir / "flow_nb_roc_curve.png")
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
        outputs_dir / "flow_nb_roc_points.csv", index=False
    )

    _save_confusion_matrix(cm, outputs_dir / "flow_nb_confusion_matrix.csv")
    _save_confusion_matrix_plot(cm, plots_dir / "flow_nb_confusion_matrix.png")

    metrics_text = (
        "Flow-level classification with Gaussian Naive Bayes (multi-feature)\n"
        f"Sampled rows: {len(data)}\n"
        f"{source_details}\n"
        f"Feature count: {len(feature_cols)}\n"
        f"var_smoothing: {args.var_smoothing}\n"
        f"Accuracy: {accuracy:.6f}\n"
        f"Precision: {precision:.6f}\n"
        f"Recall: {recall:.6f}\n"
        f"Specificity: {specificity:.6f}\n"
        f"F1: {f1:.6f}\n"
        f"Log Loss: {model_log_loss:.6f}\n"
        f"ROC AUC: {roc_auc:.6f}\n"
        "Confusion Matrix (rows=actual [0,1], cols=pred [0,1]):\n"
        f"{cm.tolist()}\n"
    )
    print(metrics_text)
    (outputs_dir / "flow_nb_metrics.txt").write_text(metrics_text, encoding="utf-8")


if __name__ == "__main__":
    main()
