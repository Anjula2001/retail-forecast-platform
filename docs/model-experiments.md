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
- Test training-period isolation and preprocessing behavior.
- Automate candidate training, comparison and artifact saving.
- Record experiment parameters, data identity and software versions.
- Freeze the chosen approach before final holdout evaluation.