from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

try:
    import py7zr
except Exception:  # pragma: no cover - optional dependency
    py7zr = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.logistic_regression import (
    _clean_windows_extended_path,
    _collect_sample_from_7z_archives_with_groups,
    _collect_sample_with_groups,
    _discover_7z_archives,
    _discover_schema_files,
    _load_paths_from_rows_csv,
    _split_with_group_holdout,
)


def _parse_csv_values(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _dedupe_keep_order(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _parse_n_estimators(raw: str) -> list[int]:
    values = _parse_csv_values(raw)
    parsed: list[int] = []
    for value in values:
        try:
            n = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid n_estimators value: {value}") from exc
        if n <= 0:
            raise ValueError(f"n_estimators must be > 0, got {n}")
        parsed.append(n)
    if not parsed:
        raise ValueError("n_estimators grid is empty.")
    return _dedupe_keep_order(parsed)


def _parse_max_depth(raw: str) -> list[int | None]:
    values = _parse_csv_values(raw)
    parsed: list[int | None] = []
    for value in values:
        lowered = value.lower()
        if lowered in {"none", "null"}:
            parsed.append(None)
            continue
        try:
            depth = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid max_depth value: {value}") from exc
        if depth <= 0:
            raise ValueError(f"max_depth must be > 0 or None, got {depth}")
        parsed.append(depth)
    if not parsed:
        raise ValueError("max_depth grid is empty.")
    return _dedupe_keep_order(parsed)


def _parse_max_features(raw: str) -> list[str | int | float | None]:
    values = _parse_csv_values(raw)
    parsed: list[str | int | float | None] = []
    for value in values:
        lowered = value.lower()
        if lowered in {"none", "null"}:
            parsed.append(None)
            continue
        if lowered in {"sqrt", "log2"}:
            parsed.append(lowered)
            continue
        try:
            as_int = int(value)
        except ValueError:
            as_int = None
        if as_int is not None:
            if as_int <= 0:
                raise ValueError(f"max_features as int must be > 0, got {as_int}")
            parsed.append(as_int)
            continue
        try:
            as_float = float(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid max_features value: {value} (use sqrt, log2, None, int, or float in (0,1])"
            ) from exc
        if not (0.0 < as_float <= 1.0):
            raise ValueError(f"max_features as float must be in (0, 1], got {as_float}")
        parsed.append(as_float)
    if not parsed:
        raise ValueError("max_features grid is empty.")
    return _dedupe_keep_order(parsed)


def _save_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, plot_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="red", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Random Forest ROC Curve")
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
    ax.set_title("Random Forest Confusion Matrix")

    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _load_source_files(args: argparse.Namespace, outputs_dir: Path) -> tuple[list[Path], list[Path]]:
    rows_csv = outputs_dir / "rows_per_file_84col.csv"
    files = [] if args.use_all_datasets else _load_paths_from_rows_csv(rows_csv)
    archives: list[Path] = []

    if not files:
        summary_path = outputs_dir / "eda_summary_binary_84col.json"
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

    return files, archives


def _count_csv_members_in_archives(archives: list[Path]) -> int:
    if py7zr is None:
        return 0

    total_csv_members = 0
    for archive in archives:
        try:
            with py7zr.SevenZipFile(archive, mode="r") as seven:
                total_csv_members += sum(1 for name in seven.getnames() if name.lower().endswith(".csv"))
        except Exception:
            continue

    return total_csv_members


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a flow-level Random Forest model on IEC104 CSV files with bagging "
            "and GridSearchCV tuning."
        )
    )
    parser.add_argument("--data-root", type=str, default="", help="Optional dataset root to discover 84-col CSV files.")
    parser.add_argument("--max-rows", type=int, default=200_000, help="Maximum sampled rows for training.")
    parser.add_argument("--chunk-size", type=int, default=120_000, help="Chunk size while reading CSVs.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--cv-folds", type=int, default=3, help="Cross-validation folds for GridSearchCV.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs used by GridSearchCV.")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level for GridSearchCV.")
    parser.add_argument(
        "--use-all-datasets",
        action="store_true",
        help=(
            "Use every discovered dataset file (all CSVs across archives/folders) by disabling "
            "early stopping from max_rows."
        ),
    )
    parser.add_argument(
        "--n-estimators-grid",
        type=str,
        default="100,200,300",
        help="Comma-separated values to tune n_estimators.",
    )
    parser.add_argument(
        "--max-depth-grid",
        type=str,
        default="None,10,20,30",
        help="Comma-separated values to tune max_depth (use None for unlimited).",
    )
    parser.add_argument(
        "--max-features-grid",
        type=str,
        default="sqrt,log2,None",
        help="Comma-separated values to tune max_features (sqrt/log2/None/int/float).",
    )
    args = parser.parse_args()

    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >= 2.")

    n_estimators_grid = _parse_n_estimators(args.n_estimators_grid)
    max_depth_grid = _parse_max_depth(args.max_depth_grid)
    max_features_grid = _parse_max_features(args.max_features_grid)

    outputs_dir = PROJECT_ROOT / "outputs"
    plots_dir = outputs_dir / "plots"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    files, archives = _load_source_files(args, outputs_dir)
    if not files and not archives:
        raise FileNotFoundError(
            "No source flow CSV files were found. Provide a valid dataset path with "
            "`--data-root \"C:\\path\\to\\datasets\"` (folder with CSV or .7z files) "
            "or regenerate rows_per_file_84col.csv from this machine."
        )

    effective_max_rows = args.max_rows if not args.use_all_datasets else 1_000_000_000

    if files:
        total_dataset_files = len(files)
        data, feature_cols, files_used, groups = _collect_sample_with_groups(
            files=files,
            max_rows=effective_max_rows,
            chunk_size=args.chunk_size,
            random_state=args.random_state,
        )
        source_details = f"Dataset files used: {files_used}/{total_dataset_files}\nFiles used: {files_used}"
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

    base_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=args.random_state,
                    bootstrap=True,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__n_estimators": n_estimators_grid,
        "classifier__max_depth": max_depth_grid,
        "classifier__max_features": max_features_grid,
    }

    if len(np.unique(groups_train)) < args.cv_folds:
        raise RuntimeError(
            f"Not enough training groups ({len(np.unique(groups_train))}) for cv-folds={args.cv_folds}."
        )

    cv_splitter = StratifiedGroupKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv_splitter,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        refit=True,
        return_train_score=False,
    )
    search.fit(x_train, y_train, groups=groups_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(x_test)
    y_prob = best_model.predict_proba(x_test)[:, 1]

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
    _save_roc_curve(fpr, tpr, roc_auc, plots_dir / "flow_rf_roc_curve.png")
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
        outputs_dir / "flow_rf_roc_points.csv", index=False
    )

    _save_confusion_matrix(cm, outputs_dir / "flow_rf_confusion_matrix.csv")
    _save_confusion_matrix_plot(cm, plots_dir / "flow_rf_confusion_matrix.png")

    classifier = best_model.named_steps["classifier"]
    feat_imp_df = pd.DataFrame({"feature": feature_cols, "importance": classifier.feature_importances_})
    feat_imp_df.sort_values("importance", ascending=False, inplace=True)
    feat_imp_df.to_csv(outputs_dir / "flow_rf_feature_importances.csv", index=False)

    cv_results_df = pd.DataFrame(search.cv_results_)
    if not cv_results_df.empty:
        useful_cols = [
            "param_classifier__n_estimators",
            "param_classifier__max_depth",
            "param_classifier__max_features",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]
        available_cols = [col for col in useful_cols if col in cv_results_df.columns]
        cv_results_df = cv_results_df.loc[:, available_cols].sort_values("rank_test_score")
        cv_results_df.rename(
            columns={
                "param_classifier__n_estimators": "n_estimators",
                "param_classifier__max_depth": "max_depth",
                "param_classifier__max_features": "max_features",
            },
            inplace=True,
        )
        cv_results_df.to_csv(outputs_dir / "flow_rf_cv_results.csv", index=False)

    best_params = {key.replace("classifier__", ""): value for key, value in search.best_params_.items()}
    metrics_text = (
        "Flow-level Random Forest classification (bagging + hyperparameter tuning)\n"
        f"Sampled rows: {len(data)}\n"
        f"{source_details}\n"
        f"Feature count: {len(feature_cols)}\n"
        f"Use all dataset files: {'yes' if args.use_all_datasets else 'no'}\n"
        f"Effective max rows: {effective_max_rows}\n"
        f"Group-aware split: yes\n"
        f"Train groups: {len(np.unique(groups_train))}\n"
        f"Test groups: {len(np.unique(groups_test))}\n"
        f"Group-aware CV: StratifiedGroupKFold\n"
        f"CV folds: {args.cv_folds}\n"
        f"Grid n_estimators: {n_estimators_grid}\n"
        f"Grid max_depth: {max_depth_grid}\n"
        f"Grid max_features: {max_features_grid}\n"
        f"Best params: {best_params}\n"
        f"Best CV F1: {search.best_score_:.6f}\n"
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

    metrics_path = outputs_dir / "flow_rf_metrics.txt"
    metrics_path.write_text(metrics_text, encoding="utf-8")


if __name__ == "__main__":
    main()
