import pandas as pd
import pytest
from retail_forecast.training import (
    FEATURE_COLUMNS,
    train_random_forest,
    train_random_forest_with_product_id,
)


@pytest.mark.parametrize("include_product_id", [False, True])
def test_training_excludes_outside_period_data(include_product_id):
    features = pd.DataFrame(
        {
            "StockCode": ["BEFORE", "A", "A", "FUTURE"],
            "week_start": pd.to_datetime(
                [
                    "2009-11-30",
                    "2009-12-07",
                    "2009-12-14",
                    "2011-06-06",
                ]
            ),
            "units_sold": [10000.0, 10.0, 10.0, 20000.0],
        }
    )

    for column in FEATURE_COLUMNS:
        features[column] = 5.0

    original = features.copy(deep=True)

    trainer = train_random_forest_with_product_id if include_product_id else train_random_forest

    model = trainer(
        features,
        train_start=pd.Timestamp("2009-12-07"),
        train_end_exclusive=pd.Timestamp("2011-06-06"),
    )

    if include_product_id:
        inputs = features.loc[[1], ["StockCode", *FEATURE_COLUMNS]]
        encoder = model.named_steps["preprocessing"].named_transformers_["product"]
        assert encoder.categories_[0].tolist() == ["A"]
    else:
        inputs = features.loc[[1], list(FEATURE_COLUMNS)].astype("float64")

    assert model.predict(inputs)[0] == pytest.approx(10.0)
    pd.testing.assert_frame_equal(features, original)
