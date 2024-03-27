import numpy as np

from postprocessing_component import predict_fn, output_fn


def test_predict_returns_prediction_as_first_column():
    input_data = [
        [0.8, 0.2],
        [0.2, 0.8],
        [0.4, 0.6]
    ]

    categories = ["home_not_win", "home_win"]

    response = predict_fn(input_data, categories)

    assert response == [
        ("home_not_win", 0.8),
        ("home_win", 0.8),
        ("home_win", 0.6)
    ]


def test_output_does_not_return_array_if_single_prediction():
    prediction = [("home_win", 0.6)]
    response, _ = output_fn(prediction, "application/json")

    assert response["prediction"] == "home_win"


def test_output_returns_array_if_multiple_predictions():
    prediction = [("home_win", 0.6), ("home_not_win", 0.8)]
    response, _ = output_fn(prediction, "application/json")

    assert len(response) == 2
    assert response[0]["prediction"] == "home_win"
    assert response[1]["prediction"] == "home_not_win"
