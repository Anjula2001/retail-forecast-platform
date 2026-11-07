# retail-forecast-platform
An end-to-end retail sales forecasting platform with reproducible ML pipelines, model evaluation, and AWS deployment.

## Run a validation experiment

Install the package from the repository root:

```bash
pip install -e .
```

Then run the candidate comparison:

```bash
retail-forecast-experiment
```

By default, the command reads `data/processed/weekly_sales.csv` and
`configs/forecast.toml`, then creates a timestamped run directory under
`artifacts/experiments/`. Each run contains:

- `metrics.json`: MAE, WAPE, and row counts for each candidate on the same validation rows.
- `validation_predictions.csv`: actuals and candidate predictions for each scored product-week.
- `models.joblib`: fitted Random Forest models.
- `metadata.json`: input and configuration SHA-256 hashes, date boundaries, model parameters,
  software versions, and artifact filenames.

Paths can be overridden with `--input`, `--config`, and `--output-dir`.
