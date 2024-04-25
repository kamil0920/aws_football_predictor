import numpy as np

from postprocessing_component import predict_fn, output_fn, input_fn


def test_predict_returns_prediction_as_first_column():
    input_data = [
        0.8213,
        0.21321,
        0.4321
    ]

    categories = ["home_not_win", "home_win"]

    response = predict_fn(input_data, categories)

    assert response == [
        ("home_win", 0.8213),
        ("home_not_win", 0.21321),
        ("home_not_win", 0.4321)
    ]


def test_output_does_not_return_array_if_single_prediction():
    prediction = [("home_win", np.array(0.6))]
    response, _ = output_fn(prediction, "text/csv")

    assert response["prediction"] == "home_win"


def test_output_returns_array_if_multiple_predictions():
    prediction = [("home_win", np.array(0.6)), ("home_not_win", np.array(0.8))]
    response, _ = output_fn(prediction, "text/csv")

    assert len(response) == 2
    assert response[0]["prediction"] == "home_win"
    assert response[1]["prediction"] == "home_not_win"


def test_input_returns_array_if_multiple_predictions():
    response = input_fn('0.3566516637802124\n0.4344041347503662\n', 'text/csv')
    assert len(response) == 2
