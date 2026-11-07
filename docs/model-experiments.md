# Initial forecasting experiments

## Evaluation
- Scope: 20 products selected using training data only.
- Training: 2009-12-07 through 2011-06-05.
- Validation: 2011-06-06 through 2011-09-04.
- Protocol: rolling one-week-ahead forecasts.
- Same 260 product-week rows scored for all candidates.
- Test-period performance has not been evaluated.

## Results

| Candidate | MAE | WAPE (%) |
| --- | ---: | ---: |
| Previous 4-week average | 230.824038 | 58.075684 |
| Random Forest: sales-history features | 240.990183 | 60.633501 |
| Random Forest: sales history + product ID | 241.306679 | 60.713132 |

## Decision
The 4-week average remains the leading validation candidate.

Adding product identity did not improve this Random Forest candidate.
This does not establish that product identity is generally unhelpful.

These results apply to this product subset and validation period.
Final generalization performance remains unconfirmed.

## Next steps
- Training-period isolation is covered by tests in `tests/test_selection.py` and `tests/test_training.py`.
- Candidate training, validation comparison, and artifact saving are available with
  `retail-forecast-experiment`.
- Each run records parameters, input and config hashes, and software versions in `metadata.json`.
- The final holdout result below remains the recorded test result; the experiment command uses
  validation data only.

Next, define the operational forecast input/output contract and retraining cadence before choosing
an AWS deployment shape.

## Final holdout evaluation

- Method: previous 4-week average, selected on validation.
- Test period: 2011-09-05 through 2011-12-04.
- Protocol: one-week-ahead forecasts using the four preceding actual weeks.
- Scored rows: 260 of 260.
- MAE: 223.53 units per product-week.
- WAPE: 45.08%.
- Run output: `artifacts/final_test/metrics.json`.
