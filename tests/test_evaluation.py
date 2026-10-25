import pandas as pd
import pytest
from retail_forecast.evaluation import evaluate_forecasts


def test_metrics_use_only_paired_actuals_and_predictions():
    forecasts = pd.DataFrame(
        {
            "units_sold": [20, 30, 40, None],
            "prediction": [10, 30, None, 100],
        }
    )

    result = evaluate_forecasts(forecasts)

    assert result["rows"] == 4
    assert result["known_actuals"] == 3
    assert result["scored_rows"] == 2
    assert result["mae"] == pytest.approx(5.0)
    assert result["wape_pct"] == pytest.approx(20.0)


def test_wape_is_undefined_when_scored_actuals_total_zero():
    forecasts = pd.DataFrame(
        {
            "units_sold": [0, 0],
            "prediction": [1, 3],
        }
    )

    result = evaluate_forecasts(forecasts)

    assert result["mae"] == pytest.approx(2.0)
    assert result["wape_pct"] is None


def test_metrics_reject_data_without_scorable_pairs():
    forecasts = pd.DataFrame(
        {
            "units_sold": [10, None],
            "prediction": [None, 20],
        }
    )

    with pytest.raises(ValueError, match="No paired"):
        evaluate_forecasts(forecasts)
