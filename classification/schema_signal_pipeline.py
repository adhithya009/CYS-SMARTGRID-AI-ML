from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

try:
    import py7zr
except Exception:  # pragma: no cover - optional dependency
    py7zr = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classification.logistic_regression import _discover_7z_archives, _split_with_group_holdout


AMBIGUOUS_LABELS = {"", "NAN", "<NA>", "NULL", "NONE", "UNKNOWN", "UNLABELED", "UNLABELLED", "?"}


@dataclass
class SchemaSample:
    data: pd.DataFrame
    feature_cols: list[str]
    groups: np.ndarray
    sources_used: int


class SignalFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, clip_low: float = 0.01, clip_high: float = 0.99, skew_threshold: float = 2.0):
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.skew_threshold = skew_threshold
        self.feature_names_: list[str] = []
        self.lower_: pd.Series | None = None
        self.upper_: pd.Series | None = None
        self.log_cols_: list[str] = []

    def _as_frame(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x.copy()
        if not self.feature_names_:
            raise RuntimeError("Feature names are not initialized.")
        return pd.DataFrame(x, columns=self.feature_names_)

    def fit(self, x: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> "SignalFeatureEngineer":
        df = self._as_frame(x) if self.feature_names_ else (x.copy() if isinstance(x, pd.DataFrame) else pd.DataFrame(x))
        self.feature_names_ = [str(col) for col in df.columns]
        df = df.reindex(columns=self.feature_names_)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.apply(pd.to_numeric, errors="coerce").astype(np.float64, copy=False)
        self.lower_ = df.quantile(self.clip_low)
        self.upper_ = df.quantile(self.clip_high)
        clipped = df.clip(lower=self.lower_, upper=self.upper_, axis=1)
        skew = clipped.skew(numeric_only=True)
        self.log_cols_ = [col for col, val in skew.items() if np.isfinite(val) and abs(float(val)) >= self.skew_threshold]
        return self

    def transform(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if self.lower_ is None or self.upper_ is None:
            raise RuntimeError("SignalFeatureEngineer is not fitted.")
        df = self._as_frame(x)
        df = df.reindex(columns=self.feature_names_, fill_value=np.nan)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.apply(pd.to_numeric, errors="coerce").astype(np.float64, copy=False)
        df = df.clip(lower=self.lower_, upper=self.upper_, axis=1)
        if self.log_cols_:
            vals = df[self.log_cols_]
            df.loc[:, self.log_cols_] = np.sign(vals) * np.log1p(np.abs(vals))
        return df


def _discover_schema_files_any(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    files: list[Path] = []
    for p in sorted(data_root.rglob("*.csv")):
        try:
            cols = pd.read_csv(p, nrows=0).columns.tolist()
        except Exception:
            continue
        if "Label" in cols:
            files.append(p)
    return files


def _normalize_label(label_series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    label_str = label_series.astype("string").str.strip().str.upper()
    missing_mask = label_str.isna() | label_str.isin({"<NA>"})
    ambiguous_mask = label_str.fillna("").isin(AMBIGUOUS_LABELS)
    valid_mask = ~(missing_mask | ambiguous_mask)
    label_bin = pd.Series(np.where(label_str.eq("NORMAL"), 0, 1), index=label_series.index, dtype="int64")
    return label_bin, valid_mask, missing_mask | ambiguous_mask


def _safe_metric_value(func, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(func(y_true, y_pred))
    except Exception:
        return float("nan")


def _threshold_objective(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, target_metric: str) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    if target_metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if target_metric == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    if target_metric == "precision":
        return float(precision_score(y_true, y_pred, zero_division=0))
    raise ValueError(f"Unsupported target metric: {target_metric}")


def _tune_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_metric: str) -> tuple[float, float, pd.DataFrame]:
    if len(y_true) == 0:
        return 0.5, float("nan"), pd.DataFrame(columns=["threshold", "score"])
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = np.array([_threshold_objective(y_true, y_prob, thr, target_metric) for thr in thresholds], dtype=float)
    best_idx = int(np.nanargmax(scores))
    curve_df = pd.DataFrame({"threshold": thresholds, "score": scores})
    return float(thresholds[best_idx]), float(scores[best_idx]), curve_df


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15))),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def _save_confusion_matrix(cm: np.ndarray, cm_path: Path) -> None:
    cm_df = pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])
    cm_df.to_csv(cm_path)


def _save_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, plot_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="red", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def _save_confusion_matrix_plot(cm: np.ndarray, plot_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred_0", "pred_1"])
    ax.set_yticks([0, 1], labels=["actual_0", "actual_1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _build_classifier(args: argparse.Namespace):
    if args.model == "logistic":
        return LogisticRegression(
            max_iter=args.max_iter,
            solver=args.logistic_solver,
            tol=args.logistic_tol,
            random_state=args.random_state,
            class_weight="balanced",
            n_jobs=args.n_jobs,
        )
    if args.model == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.rf_max_depth,
            max_features=args.rf_max_features,
            random_state=args.random_state,
            class_weight="balanced_subsample",
            n_jobs=args.n_jobs,
        )
    if args.model == "mlp":
        hidden_layers = tuple(int(part.strip()) for part in args.hidden_layer_sizes.split(",") if part.strip())
        return MLPClassifier(
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
        )
    raise ValueError(f"Unsupported model: {args.model}")


def _build_pipeline(args: argparse.Namespace) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("signal", SignalFeatureEngineer(clip_low=args.clip_low, clip_high=args.clip_high, skew_threshold=args.skew_threshold)),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if args.model in {"logistic", "mlp"}:
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("classifier", _build_classifier(args)))
    return Pipeline(steps=steps)


def _oversample_minority(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    if len(np.unique(y_train)) < 2:
        return x_train, y_train, 0
    counts = np.bincount(y_train.astype(int))
    if len(counts) < 2 or counts[0] == counts[1]:
        return x_train, y_train, 0

    minority_class = int(np.argmin(counts))
    majority_class = int(np.argmax(counts))
    need = int(counts[majority_class] - counts[minority_class])
    if need <= 0:
        return x_train, y_train, 0

    rng = np.random.default_rng(random_state)
    minority_idx = np.where(y_train == minority_class)[0]
    sampled_idx = rng.choice(minority_idx, size=need, replace=True)
    x_aug = pd.concat([x_train, x_train.iloc[sampled_idx]], ignore_index=True)
    y_aug = np.concatenate([y_train, y_train[sampled_idx]])
    return x_aug, y_aug, need


def _upsert_source_report(report: dict[str, dict], source_id: str, schema_cols: int | None) -> dict:
    if source_id not in report:
        report[source_id] = {
            "source_id": source_id,
            "schema_cols": schema_cols if schema_cols is not None else -1,
            "total_rows": 0,
            "dropped_ambiguous_or_missing_label": 0,
            "valid_label_rows": 0,
            "normal_rows": 0,
            "attack_rows": 0,
            "sampled_rows": 0,
        }
    elif schema_cols is not None and report[source_id]["schema_cols"] < 0:
        report[source_id]["schema_cols"] = schema_cols
    return report[source_id]


def _collect_schema_samples(
    files: list[Path],
    archives: list[Path],
    max_rows: int,
    chunk_size: int,
    sample_per_chunk: int,
    random_state: int,
    allowed_schemas: set[int] | None,
) -> tuple[dict[int, SchemaSample], pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    schema_frames: dict[int, list[pd.DataFrame]] = {}
    schema_groups: dict[int, list[pd.Series]] = {}
    schema_feature_cols: dict[int, list[str]] = {}
    schema_sources_used: dict[int, set[str]] = {}
    schema_row_counts: dict[int, int] = {}
    source_report: dict[str, dict] = {}
    total_sampled_rows = 0

    def process_chunk(chunk: pd.DataFrame, source_id: str) -> None:
        nonlocal total_sampled_rows
        schema_cols = len(chunk.columns)
        if allowed_schemas is not None and schema_cols not in allowed_schemas:
            return

        rec = _upsert_source_report(source_report, source_id, schema_cols)
        rec["total_rows"] += int(len(chunk))

        if "Label" not in chunk.columns:
            rec["dropped_ambiguous_or_missing_label"] += int(len(chunk))
            return

        label_bin, valid_label_mask, dropped_mask = _normalize_label(chunk["Label"])
        rec["dropped_ambiguous_or_missing_label"] += int(dropped_mask.sum())
        rec["valid_label_rows"] += int(valid_label_mask.sum())
        if valid_label_mask.sum() <= 0:
            return

        numeric_chunk = chunk.select_dtypes(include=[np.number]).copy()
        if numeric_chunk.empty:
            return

        numeric_chunk = numeric_chunk.loc[valid_label_mask].copy()
        label_vals = label_bin.loc[valid_label_mask].to_numpy(dtype=int)
        if len(numeric_chunk) == 0:
            return

        rec["normal_rows"] += int((label_vals == 0).sum())
        rec["attack_rows"] += int((label_vals == 1).sum())

        if schema_cols not in schema_feature_cols:
            schema_feature_cols[schema_cols] = [c for c in numeric_chunk.columns]
            if not schema_feature_cols[schema_cols]:
                return

        feature_cols = schema_feature_cols[schema_cols]
        numeric_chunk = numeric_chunk.reindex(columns=feature_cols)
        numeric_chunk["label_bin"] = label_vals

        remaining = max_rows - total_sampled_rows
        if remaining <= 0:
            return
        take_n = min(sample_per_chunk, len(numeric_chunk), remaining)
        if take_n <= 0:
            return

        if take_n < len(numeric_chunk):
            idx = rng.choice(len(numeric_chunk), size=take_n, replace=False)
            sample = numeric_chunk.iloc[idx].reset_index(drop=True)
        else:
            sample = numeric_chunk.reset_index(drop=True)

        schema_frames.setdefault(schema_cols, []).append(sample)
        schema_groups.setdefault(schema_cols, []).append(pd.Series(np.repeat(source_id, len(sample)), dtype="string"))
        schema_sources_used.setdefault(schema_cols, set()).add(source_id)
        schema_row_counts[schema_cols] = schema_row_counts.get(schema_cols, 0) + len(sample)
        rec["sampled_rows"] += int(len(sample))
        total_sampled_rows += int(len(sample))

    for path in files:
        try:
            source_id = str(path.resolve())
        except Exception:
            source_id = str(path)
        try:
            for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
                process_chunk(chunk, source_id)
                if total_sampled_rows >= max_rows:
                    break
        except Exception:
            continue
        if total_sampled_rows >= max_rows:
            break

    if total_sampled_rows < max_rows:
        if py7zr is None and archives:
            raise ModuleNotFoundError("py7zr is required to read .7z files. Install with: python -m pip install py7zr")
        for archive in archives:
            try:
                with py7zr.SevenZipFile(archive, mode="r") as seven:
                    members = [m for m in seven.getnames() if m.lower().endswith(".csv")]
                    for member in members:
                        if total_sampled_rows >= max_rows:
                            break
                        seven.reset()
                        with tempfile.TemporaryDirectory(prefix="iec104_schema_") as tmpdir:
                            seven.extract(targets=[member], path=tmpdir)
                            extracted = Path(tmpdir) / member
                            if not extracted.exists():
                                continue
                            source_id = f"{archive.resolve()}::{member}"
                            try:
                                for chunk in pd.read_csv(extracted, chunksize=chunk_size, low_memory=False):
                                    process_chunk(chunk, source_id)
                                    if total_sampled_rows >= max_rows:
                                        break
                            except Exception:
                                continue
            except Exception:
                continue
            if total_sampled_rows >= max_rows:
                break

    schema_samples: dict[int, SchemaSample] = {}
    for schema_cols, frames in schema_frames.items():
        if not frames:
            continue
        feature_cols = schema_feature_cols[schema_cols]
        data = pd.concat(frames, ignore_index=True)
        groups = pd.concat(schema_groups[schema_cols], ignore_index=True).to_numpy(dtype=object)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid_mask = ~data[feature_cols].isna().all(axis=1)
        data = data.loc[valid_mask].reset_index(drop=True)
        groups = groups[valid_mask.to_numpy()]
        data["label_bin"] = data["label_bin"].astype(int)
        schema_samples[schema_cols] = SchemaSample(
            data=data,
            feature_cols=feature_cols,
            groups=groups,
            sources_used=len(schema_sources_used.get(schema_cols, set())),
        )

    report_df = pd.DataFrame(source_report.values())
    if not report_df.empty:
        report_df.sort_values(["schema_cols", "source_id"], inplace=True)
    return schema_samples, report_df


def _load_data_sources(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    data_root = Path(args.data_root) if args.data_root else None
    if data_root is None or not data_root.exists():
        raise FileNotFoundError(
            "A valid dataset root is required. Pass --data-root pointing to a folder with CSV or .7z files."
        )
    if data_root.is_file() and data_root.suffix.lower() == ".7z":
        return [], [data_root]

    files = _discover_schema_files_any(data_root)
    archives = _discover_7z_archives(data_root)
    return files, archives


def _save_model_signal_details(model: Pipeline, feature_cols: list[str], out_dir: Path) -> None:
    imputer: SimpleImputer = model.named_steps["imputer"]
    all_feature_names = list(feature_cols)
    if getattr(imputer, "add_indicator", False) and getattr(imputer, "indicator_", None) is not None:
        all_feature_names.extend([f"missing__{feature_cols[i]}" for i in imputer.indicator_.features_])

    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_)
        names = all_feature_names if len(all_feature_names) == len(values) else [f"feature_{i}" for i in range(len(values))]
        df = pd.DataFrame({"feature": names, "importance": values})
        df.sort_values("importance", ascending=False, inplace=True)
        df.to_csv(out_dir / "feature_importances.csv", index=False)
    elif hasattr(classifier, "coef_"):
        values = np.asarray(classifier.coef_[0])
        names = all_feature_names if len(all_feature_names) == len(values) else [f"feature_{i}" for i in range(len(values))]
        df = pd.DataFrame({"feature": names, "coefficient": values})
        df["abs_coefficient"] = df["coefficient"].abs()
        df.sort_values("abs_coefficient", ascending=False, inplace=True)
        df.to_csv(out_dir / "top_coefficients.csv", index=False)
    elif hasattr(classifier, "coefs_") and classifier.coefs_:
        first_layer = classifier.coefs_[0]
        weights = np.mean(np.abs(first_layer), axis=1)
        names = all_feature_names if len(all_feature_names) == len(weights) else [f"feature_{i}" for i in range(len(weights))]
        df = pd.DataFrame({"feature": names, "mean_abs_input_weight": weights})
        df.sort_values("mean_abs_input_weight", ascending=False, inplace=True)
        df.to_csv(out_dir / "top_input_weights.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe schema-specific classification pipeline with label quality checks, feature engineering, "
            "class balancing, and validation threshold tuning."
        )
    )
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root with CSV or .7z files.")
    parser.add_argument("--model", type=str, default="random_forest", choices=["logistic", "random_forest", "mlp"])
    parser.add_argument("--target-metric", type=str, default="f1", choices=["f1", "recall", "precision"])
    parser.add_argument("--max-rows", type=int, default=1_000_000_000, help="Global row cap after sampling.")
    parser.add_argument("--chunk-size", type=int, default=120_000, help="Chunk size while reading CSVs.")
    parser.add_argument("--sample-per-chunk", type=int, default=3000, help="Rows sampled from each chunk.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Group-aware test split size.")
    parser.add_argument("--val-size", type=float, default=0.2, help="Group-aware validation split size (from train_val).")
    parser.add_argument("--min-schema-rows", type=int, default=5000, help="Minimum sampled rows to train a schema model.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--clip-low", type=float, default=0.01)
    parser.add_argument("--clip-high", type=float, default=0.99)
    parser.add_argument("--skew-threshold", type=float, default=2.0)

    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-max-features", type=str, default="sqrt")

    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--logistic-solver", type=str, default="lbfgs", choices=["lbfgs", "newton-cg", "saga"])
    parser.add_argument("--logistic-tol", type=float, default=1e-3)
    parser.add_argument("--hidden-layer-sizes", type=str, default="64,32")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "logistic"])
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--learning-rate-init", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--n-iter-no-change", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)

    parser.set_defaults(oversample_minority=True)
    parser.add_argument(
        "--no-oversample-minority",
        dest="oversample_minority",
        action="store_false",
        help="Disable minority oversampling on training split.",
    )
    parser.add_argument(
        "--schemas",
        type=str,
        default="84,112,119",
        help="Comma-separated schema column counts to model. Example: 84,112,119",
    )
    args = parser.parse_args()

    allowed_schemas = {int(part.strip()) for part in args.schemas.split(",") if part.strip()}
    project_root = Path(__file__).resolve().parents[1]
    outputs_root = project_root / "outputs" / "schema_signal_pipeline"
    outputs_root.mkdir(parents=True, exist_ok=True)

    files, archives = _load_data_sources(args)
    schema_samples, label_quality_df = _collect_schema_samples(
        files=files,
        archives=archives,
        max_rows=args.max_rows,
        chunk_size=args.chunk_size,
        sample_per_chunk=args.sample_per_chunk,
        random_state=args.random_state,
        allowed_schemas=allowed_schemas,
    )

    if label_quality_df.empty:
        raise RuntimeError("No valid labeled rows were collected. Check dataset path and label column quality.")

    label_quality_path = outputs_root / "label_quality_report.csv"
    label_quality_df.to_csv(label_quality_path, index=False)

    schema_summary_rows: list[dict] = []
    overall_y_true: list[np.ndarray] = []
    overall_y_pred: list[np.ndarray] = []
    overall_y_prob: list[np.ndarray] = []

    for schema_cols in sorted(schema_samples):
        sample = schema_samples[schema_cols]
        data = sample.data
        feature_cols = sample.feature_cols
        groups = sample.groups
        y = data["label_bin"].to_numpy(dtype=int)
        x = data[feature_cols].copy()

        if len(data) < args.min_schema_rows:
            schema_summary_rows.append(
                {
                    "schema_cols": schema_cols,
                    "status": "skipped_too_few_rows",
                    "rows": len(data),
                    "unique_groups": int(len(np.unique(groups))),
                }
            )
            continue
        if len(np.unique(groups)) < 3:
            schema_summary_rows.append(
                {
                    "schema_cols": schema_cols,
                    "status": "skipped_too_few_groups",
                    "rows": len(data),
                    "unique_groups": int(len(np.unique(groups))),
                }
            )
            continue

        try:
            x_train_val, x_test, y_train_val, y_test, groups_train_val, groups_test = _split_with_group_holdout(
                x=x,
                y=y,
                groups=groups,
                test_size=args.test_size,
                random_state=args.random_state,
            )
            x_train, x_val, y_train, y_val, groups_train, groups_val = _split_with_group_holdout(
                x=x_train_val,
                y=y_train_val,
                groups=groups_train_val,
                test_size=args.val_size,
                random_state=args.random_state + 73,
            )
        except RuntimeError as exc:
            schema_summary_rows.append(
                {
                    "schema_cols": schema_cols,
                    "status": f"skipped_split_failed:{exc}",
                    "rows": len(data),
                    "unique_groups": int(len(np.unique(groups))),
                }
            )
            continue

        overlaps = (
            len(set(groups_train).intersection(set(groups_val)))
            + len(set(groups_train).intersection(set(groups_test)))
            + len(set(groups_val).intersection(set(groups_test)))
        )
        if overlaps != 0:
            raise RuntimeError(f"Group overlap detected for schema {schema_cols}.")

        oversampled_rows = 0
        if args.oversample_minority:
            x_train, y_train, oversampled_rows = _oversample_minority(
                x_train=x_train,
                y_train=y_train,
                random_state=args.random_state,
            )

        model = _build_pipeline(args)
        model.fit(x_train, y_train)

        val_prob = model.predict_proba(x_val)[:, 1]
        threshold, best_val_score, threshold_curve_df = _tune_threshold(
            y_true=y_val,
            y_prob=val_prob,
            target_metric=args.target_metric,
        )

        test_prob = model.predict_proba(x_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)
        metrics = _compute_metrics(y_true=y_test, y_pred=test_pred, y_prob=test_prob)
        cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
        fpr, tpr, roc_thresholds = roc_curve(y_test, test_prob)

        schema_dir = outputs_root / f"schema_{schema_cols}"
        plots_dir = schema_dir / "plots"
        schema_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).to_csv(
            schema_dir / "roc_points.csv",
            index=False,
        )
        threshold_curve_df.to_csv(schema_dir / "threshold_tuning_curve.csv", index=False)
        _save_confusion_matrix(cm, schema_dir / "confusion_matrix.csv")
        _save_confusion_matrix_plot(cm, plots_dir / "confusion_matrix.png", title=f"Schema {schema_cols} Confusion Matrix")
        _save_roc_curve(fpr, tpr, metrics["roc_auc"], plots_dir / "roc_curve.png", title=f"Schema {schema_cols} ROC Curve")
        _save_model_signal_details(model=model, feature_cols=feature_cols, out_dir=schema_dir)

        metrics_payload = {
            "schema_cols": schema_cols,
            "model": args.model,
            "target_metric_for_threshold": args.target_metric,
            "threshold": threshold,
            "best_validation_target_metric": best_val_score,
            "rows": len(data),
            "train_rows": len(x_train),
            "val_rows": len(x_val),
            "test_rows": len(x_test),
            "train_groups": int(len(np.unique(groups_train))),
            "val_groups": int(len(np.unique(groups_val))),
            "test_groups": int(len(np.unique(groups_test))),
            "source_files_used": sample.sources_used,
            "oversampled_rows_added": oversampled_rows,
            "metrics": metrics,
        }
        (schema_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        schema_summary_rows.append(
            {
                "schema_cols": schema_cols,
                "status": "ok",
                "rows": len(data),
                "source_files_used": sample.sources_used,
                "train_groups": int(len(np.unique(groups_train))),
                "val_groups": int(len(np.unique(groups_val))),
                "test_groups": int(len(np.unique(groups_test))),
                "threshold": threshold,
                "val_target_metric": best_val_score,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "log_loss": metrics["log_loss"],
            }
        )

        overall_y_true.append(y_test)
        overall_y_pred.append(test_pred)
        overall_y_prob.append(test_prob)

    summary_df = pd.DataFrame(schema_summary_rows)
    summary_df.to_csv(outputs_root / "schema_summary.csv", index=False)

    if not overall_y_true:
        raise RuntimeError("All schemas were skipped. Check schema filters or data quality.")

    y_true_all = np.concatenate(overall_y_true)
    y_pred_all = np.concatenate(overall_y_pred)
    y_prob_all = np.concatenate(overall_y_prob)
    overall_metrics = _compute_metrics(y_true=y_true_all, y_pred=y_pred_all, y_prob=y_prob_all)
    overall_cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    fpr_all, tpr_all, thr_all = roc_curve(y_true_all, y_prob_all)

    _save_confusion_matrix(overall_cm, outputs_root / "overall_confusion_matrix.csv")
    _save_confusion_matrix_plot(overall_cm, outputs_root / "overall_confusion_matrix.png", title="Overall Confusion Matrix")
    _save_roc_curve(fpr_all, tpr_all, overall_metrics["roc_auc"], outputs_root / "overall_roc_curve.png", title="Overall ROC Curve")
    pd.DataFrame({"fpr": fpr_all, "tpr": tpr_all, "threshold": thr_all}).to_csv(
        outputs_root / "overall_roc_points.csv",
        index=False,
    )

    overall_text = (
        "Schema-specific leakage-safe classification summary\n"
        f"Model: {args.model}\n"
        f"Target metric (threshold tuning): {args.target_metric}\n"
        f"Schemas requested: {sorted(allowed_schemas)}\n"
        f"Schemas trained: {int((summary_df['status'] == 'ok').sum())}\n"
        f"Accuracy: {overall_metrics['accuracy']:.6f}\n"
        f"Precision: {overall_metrics['precision']:.6f}\n"
        f"Recall: {overall_metrics['recall']:.6f}\n"
        f"Specificity: {overall_metrics['specificity']:.6f}\n"
        f"F1: {overall_metrics['f1']:.6f}\n"
        f"Log Loss: {overall_metrics['log_loss']:.6f}\n"
        f"ROC AUC: {overall_metrics['roc_auc']:.6f}\n"
        "Confusion Matrix (rows=actual [0,1], cols=pred [0,1]):\n"
        f"{overall_cm.tolist()}\n"
    )
    print(overall_text)
    (outputs_root / "overall_metrics.txt").write_text(overall_text, encoding="utf-8")


if __name__ == "__main__":
    main()
