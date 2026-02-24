# IEC104 EDA (84-column schema, binary target)

- Total CSV files found: 313
- Selected files (84 columns, excluding `Balanced_IEC104_Train_Test_CSV_Files`): 192
- Rows in selected files: 2594422
- Duplicate rows in selected files (within-file): 0
- NORMAL (0): 1965605
- ATTACK (1): 628817
- Attack rate: 24.237267%
- Header errors: 0
- Read errors: 0

## Output files
- `outputs/eda_summary_binary_84col.json`
- `outputs/label_distribution_binary_84col.csv`
- `outputs/missingness_84col.csv`
- `outputs/numeric_quality_84col.csv`
- `outputs/rows_per_file_84col.csv`

## Plots
- `outputs/plots/class_distribution_binary.png`
- `outputs/plots/missingness_top20.png`
- `outputs/plots/zero_pct_top20.png`
- `outputs/plots/infinite_values_top20.png` (if any infinite values exist)
- `outputs/plots/rows_per_file_hist.png`
- `outputs/plots/correlation_heatmap_top25.png`
- `outputs/plots/top_features_distribution_by_class.png`