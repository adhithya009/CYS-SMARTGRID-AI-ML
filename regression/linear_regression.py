from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import py7zr
except Exception:  # pragma: no cover - optional dependency
    py7zr = None


def _clean_windows_extended_path(path: str) -> Path:
    cleaned = path.strip()
    if cleaned.startswith("\\\\?\\"):
        cleaned = cleaned[4:]
    return Path(cleaned)


def _load_paths_from_rows_csv(rows_csv_path: Path) -> list[Path]:
    if not rows_csv_path.exists():
        return []
    rows_df = pd.read_csv(rows_csv_path)
    if "path" not in rows_df.columns:
        return []

    files: list[Path] = []
    for raw in rows_df["path"].astype(str):
        p = _clean_windows_extended_path(raw)
        if p.exists():
            files.append(p)
    return files


def _discover_schema_files(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []

    files: list[Path] = []
    for p in sorted(data_root.rglob("*.csv")):
        try:
            cols = pd.read_csv(p, nrows=0).columns.tolist()
        except Exception:
            continue
        if "Label" in cols and len(cols) == 84:
            files.append(p)
    return files


def _discover_7z_archives(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []

    archives = sorted(data_root.rglob("*.7z"))
    preferred = [p for p in archives if "balanced_iec104_train_test_csv_files" in p.name.lower()]
    others = [p for p in archives if p not in preferred]
    return preferred + others


def _collect_sample(
    files: list[Path],
    max_rows: int,
    chunk_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, list[str], int]:
    rng = np.random.default_rng(random_state)
    frames: list[pd.DataFrame] = []
    feature_cols: list[str] | None = None
    sampled_rows = 0
    files_used = 0

    for path in files:
        used_this_file = False
        try:
            for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
                if "Label" not in chunk.columns:
                    continue

                label_bin = np.where(chunk["Label"].astype("string") == "NORMAL", 0, 1)
                numeric_chunk = chunk.select_dtypes(include=[np.number]).copy()
                numeric_chunk["label_bin"] = label_bin

                if feature_cols is None:
                    feature_cols = [c for c in numeric_chunk.columns if c != "label_bin"]
                    if not feature_cols:
                        continue

                numeric_chunk = numeric_chunk.reindex(columns=feature_cols + ["label_bin"])

                remaining = max_rows - sampled_rows
                if remaining <= 0:
                    break

                take_n = min(3000, len(numeric_chunk), remaining)
                if take_n <= 0:
                    continue

                if take_n < len(numeric_chunk):
                    idx = rng.choice(len(numeric_chunk), size=take_n, replace=False)
                    sample = numeric_chunk.iloc[idx]
                else:
                    sample = numeric_chunk

                frames.append(sample)
                used_this_file = True
                sampled_rows += len(sample)

            if used_this_file:
                files_used += 1
            if sampled_rows >= max_rows:
                break
        except Exception:
            continue

    if not frames or feature_cols is None:
        return pd.DataFrame(), [], 0

    data = pd.concat(frames, ignore_index=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna(how="all", subset=feature_cols)
    data["label_bin"] = data["label_bin"].astype(int)
    return data, feature_cols, files_used


def _collect_sample_from_7z_archives(
    archives: list[Path],
    max_rows: int,
    chunk_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, list[str], int, int]:
    if py7zr is None:
        raise ModuleNotFoundError(
            "py7zr is required to read .7z files. Install it with: python -m pip install py7zr"
        )

    rng = np.random.default_rng(random_state)
    frames: list[pd.DataFrame] = []
    feature_cols: list[str] | None = None
    sampled_rows = 0
    archives_used = 0
    csv_members_used = 0

    for archive in archives:
        used_this_archive = False
        try:
            with py7zr.SevenZipFile(archive, mode="r") as seven:
                csv_members = [n for n in seven.getnames() if n.lower().endswith(".csv")]
                for member in csv_members:
                    remaining = max_rows - sampled_rows
                    if remaining <= 0:
                        break

                    seven.reset()
                    with tempfile.TemporaryDirectory(prefix="iec104_") as tmpdir:
                        seven.extract(targets=[member], path=tmpdir)
                        extracted_csv = Path(tmpdir) / member
                        if not extracted_csv.exists():
                            continue

                        for chunk in pd.read_csv(extracted_csv, chunksize=chunk_size, low_memory=False):
                            if "Label" not in chunk.columns:
                                continue

                            label_bin = np.where(chunk["Label"].astype("string") == "NORMAL", 0, 1)
                            numeric_chunk = chunk.select_dtypes(include=[np.number]).copy()
                            numeric_chunk["label_bin"] = label_bin

                            if feature_cols is None:
                                feature_cols = [c for c in numeric_chunk.columns if c != "label_bin"]
                                if not feature_cols:
                                    continue

                            numeric_chunk = numeric_chunk.reindex(columns=feature_cols + ["label_bin"])

                            remaining = max_rows - sampled_rows
                            if remaining <= 0:
                                break

                            take_n = min(3000, len(numeric_chunk), remaining)
                            if take_n <= 0:
                                continue

                            if take_n < len(numeric_chunk):
                                idx = rng.choice(len(numeric_chunk), size=take_n, replace=False)
                                sample = numeric_chunk.iloc[idx]
                            else:
                                sample = numeric_chunk

                            frames.append(sample)
                            sampled_rows += len(sample)
                            used_this_archive = True

                    if used_this_archive:
                        csv_members_used += 1
                    if sampled_rows >= max_rows:
                        break
        except Exception:
            continue

        if used_this_archive:
            archives_used += 1
        if sampled_rows >= max_rows:
            break

    if not frames or feature_cols is None:
        return pd.DataFrame(), [], 0, 0

    data = pd.concat(frames, ignore_index=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna(how="all", subset=feature_cols)
    data["label_bin"] = data["label_bin"].astype(int)
    return data, feature_cols, archives_used, csv_members_used


def _save_pred_scatter(y_true: np.ndarray, y_pred: np.ndarray, plot_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.4, s=12, label="Predictions")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="red", label="Ideal")
    plt.xlabel("Actual label_bin")
    plt.ylabel("Predicted label_bin")
    plt.title("Linear Regression: Actual vs Predicted")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a flow-level linear regression model on IEC104 CSV files.")
    parser.add_argument("--data-root", type=str, default="", help="Optional dataset root to discover 84-col CSV files.")
    parser.add_argument("--max-rows", type=int, default=200_000, help="Maximum sampled rows for training.")
    parser.add_argument("--chunk-size", type=int, default=120_000, help="Chunk size while reading CSVs.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = project_root / "outputs"
    plots_dir = outputs_dir / "plots"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows_csv = outputs_dir / "rows_per_file_84col.csv"
    files = _load_paths_from_rows_csv(rows_csv)
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

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=args.random_state, stratify=y
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    regressor = model.named_steps["regressor"]
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": regressor.coef_})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df.sort_values("abs_coefficient", ascending=False, inplace=True)

    coef_path = outputs_dir / "flow_linear_top_coefficients.csv"
    coef_df.head(40).to_csv(coef_path, index=False)

    y_pred_clipped = np.clip(y_pred, 0.0, 1.0)
    _save_pred_scatter(y_test, y_pred_clipped, plots_dir / "flow_linear_actual_vs_pred.png")

    metrics_text = (
        "Flow-level linear regression (multi-feature)\n"
        f"Sampled rows: {len(data)}\n"
        f"{source_details}\n"
        f"Feature count: {len(feature_cols)}\n"
        f"R2: {r2:.6f}\n"
        f"RMSE: {rmse:.6f}\n"
        f"MAE: {mae:.6f}\n"
    )
    print(metrics_text)

    metrics_path = outputs_dir / "flow_linear_metrics.txt"
    metrics_path.write_text(metrics_text, encoding="utf-8")


if __name__ == "__main__":
    main()
