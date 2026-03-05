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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.logistic_regression import (
    _clean_windows_extended_path,
    _collect_sample_from_7z_archives_with_groups,
    _collect_sample_with_groups,
    _count_csv_members_in_archives,
    _discover_7z_archives,
    _discover_schema_files,
    _load_paths_from_rows_csv,
    _split_with_group_holdout,
)


def _parse_hidden_layers(raw: str) -> tuple[int, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("hidden_layer_sizes cannot be empty.")
    layers: list[int] = []
    for value in values:
        try:
            n = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid hidden layer size: {value}") from exc
        if n <= 0:
            raise ValueError(f"Hidden layer sizes must be > 0, got {n}")
        layers.append(n)
    return tuple(layers)


def _save_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, plot_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="red", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("MLP ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


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
    ax.set_title("MLP Confusion Matrix")

    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a flow-level MLP classifier on IEC104 CSV files.")
    parser.add_argument("--data-root", type=str, default="", help="Optional dataset root to discover 84-col CSV files.")
    parser.add_argument("--max-rows", type=int, default=200_000, help="Maximum sampled rows for training.")
    parser.add_argument("--chunk-size", type=int, default=120_000, help="Chunk size while reading CSVs.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--use-all-datasets",
        action="store_true",
        help=(
            "Use every discovered dataset file (all CSVs across archives/folders) by ignoring rows_per_file_84col.csv "
            "and disabling early stopping from max_rows."
        ),
    )
    parser.add_argument(
        "--hidden-layer-sizes",
        type=str,
        default="64,32",
        help="Comma-separated hidden layer sizes for MLPClassifier.",
    )
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "logistic"], help="Activation function.")
    parser.add_argument("--alpha", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument("--learning-rate-init", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--max-iter", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Mini-batch size.")
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping.")
    parser.add_argument("--n-iter-no-change", type=int, default=8, help="Epochs with no improvement before stop.")
    parser.add_argument("--validation-fraction", type=float, default=0.1, help="Validation fraction when early stopping is enabled.")
    args = parser.parse_args()

    hidden_layers = _parse_hidden_layers(args.hidden_layer_sizes)

    shared_outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir = shared_outputs_dir / "mlp_outputs"
    plots_dir = outputs_dir / "plots"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows_csv = shared_outputs_dir / "rows_per_file_84col.csv"
    files = [] if args.use_all_datasets else _load_paths_from_rows_csv(rows_csv)
    archives: list[Path] = []

    if not files:
        summary_path = shared_outputs_dir / "eda_summary_binary_84col.json"
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
                if args.use_all_datasets:
                    archives = _discover_7z_archives(discover_root)
                    if not archives:
                        files = _discover_schema_files(discover_root)
                else:
                    files = _discover_schema_files(discover_root)
                    if not files:
                        archives = _discover_7z_archives(discover_root)

    if not files and not archives:
        raise FileNotFoundError(
            "No source flow CSV files were found. Provide a valid dataset path with "
            "`--data-root \"C:\\path\\to\\datasets\"` (folder with CSV or .7z files) "
            "or regenerate rows_per_file_84col.csv from this machine."
        )

    effective_max_rows = args.max_rows if not args.use_all_datasets else 1_000_000_000

    if files:
        data, feature_cols, files_used, groups = _collect_sample_with_groups(
            files=files,
            max_rows=effective_max_rows,
            chunk_size=args.chunk_size,
            random_state=args.random_state,
        )
        source_details = f"Dataset files used: {files_used}/{len(files)}\nFiles used: {files_used}"
    else:
        total_dataset_files = _count_csv_members_in_archives(archives)
        data, feature_cols, archives_used, csv_members_used, groups = _collect_sample_from_7z_archives_with_groups(
            archives=archives,
            max_rows=effective_max_rows,
            chunk_size=args.chunk_size,
            random_state=args.random_state,
        )
        source_details = (
            f"Dataset files used: {csv_members_used}/{total_dataset_files}\n"
            f"7z archives used: {archives_used}\n"
            f"CSV members read: {csv_members_used}"
        )

    if data.empty or not feature_cols:
        raise RuntimeError("Failed to build a training sample. Check source CSV format and numeric feature availability.")

    x = data[feature_cols]
    y = data["label_bin"].to_numpy()

    if np.unique(y).size < 2:
        raise RuntimeError("Only one class is present in sampled data. Increase max rows or adjust the dataset source.")

    x_train, x_test, y_train, y_test, groups_train, groups_test = _split_with_group_holdout(
        x=x,
        y=y,
        groups=groups,
        test_size=0.2,
        random_state=args.random_state,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation=args.activation,
                    alpha=args.alpha,
                    learning_rate_init=args.learning_rate_init,
                    max_iter=args.max_iter,
                    batch_size=args.batch_size,
                    early_stopping=args.early_stopping,
                    n_iter_no_change=args.n_iter_no_change,
                    validation_fraction=args.validation_fraction,
                    random_state=args.random_state,
                ),
            ),
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
    roc_aoc = roc_auc
    model_log_loss = float(log_loss(y_test, np.clip(y_prob, 1e-15, 1 - 1e-15)))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    _save_roc_curve(fpr, tpr, roc_auc, plots_dir / "flow_mlp_roc_curve.png")
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
        outputs_dir / "flow_mlp_roc_points.csv", index=False
    )

    _save_confusion_matrix(cm, outputs_dir / "flow_mlp_confusion_matrix.csv")
    _save_confusion_matrix_plot(cm, plots_dir / "flow_mlp_confusion_matrix.png")

    classifier: MLPClassifier = model.named_steps["classifier"]
    if getattr(classifier, "coefs_", None):
        first_layer = classifier.coefs_[0]
        mean_abs_weight = np.mean(np.abs(first_layer), axis=1)
        weights_df = pd.DataFrame({"feature": feature_cols, "mean_abs_input_weight": mean_abs_weight})
        weights_df.sort_values("mean_abs_input_weight", ascending=False, inplace=True)
        weights_df.head(40).to_csv(outputs_dir / "flow_mlp_top_input_weights.csv", index=False)

    metrics_text = (
        "Flow-level MLP classification (multi-feature)\n"
        f"Sampled rows: {len(data)}\n"
        f"{source_details}\n"
        f"Feature count: {len(feature_cols)}\n"
        f"Use all dataset files: {'yes' if args.use_all_datasets else 'no'}\n"
        f"Effective max rows: {effective_max_rows}\n"
        f"Group-aware split: yes\n"
        f"Train groups: {len(np.unique(groups_train))}\n"
        f"Test groups: {len(np.unique(groups_test))}\n"
        f"hidden_layer_sizes: {hidden_layers}\n"
        f"activation: {args.activation}\n"
        f"alpha: {args.alpha}\n"
        f"learning_rate_init: {args.learning_rate_init}\n"
        f"batch_size: {args.batch_size}\n"
        f"max_iter: {args.max_iter}\n"
        f"early_stopping: {args.early_stopping}\n"
        f"n_iter_no_change: {args.n_iter_no_change}\n"
        f"validation_fraction: {args.validation_fraction}\n"
        f"Accuracy: {accuracy:.6f}\n"
        f"Precision: {precision:.6f}\n"
        f"Recall: {recall:.6f}\n"
        f"Specificity: {specificity:.6f}\n"
        f"F1: {f1:.6f}\n"
        f"Log Loss: {model_log_loss:.6f}\n"
        f"ROC AUC: {roc_auc:.6f}\n"
        f"ROC AOC: {roc_aoc:.6f}\n"
        "Confusion Matrix (rows=actual [0,1], cols=pred [0,1]):\n"
        f"{cm.tolist()}\n"
    )
    print(metrics_text)

    metrics_path = outputs_dir / "flow_mlp_metrics.txt"
    metrics_path.write_text(metrics_text, encoding="utf-8")


if __name__ == "__main__":
    main()
