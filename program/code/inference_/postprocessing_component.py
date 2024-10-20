import os
import numpy as np
import json
import joblib
from io import StringIO
import pandas as pd

try:
    from sagemaker_containers.beta.framework import encoders, worker
except ImportError:
    # We don't have access to the `worker` instance when testing locally.
    # We'll set it to None so we can change the way functions create
    # a response.
    worker = None


def model_fn(model_dir):
    """
    Deserializes the target model and returns the list of fitted categories.
    """

    model = joblib.load(os.path.join(model_dir, "target.joblib"))
    return model.named_transformers_["result_match"].categories_[0]


def input_fn(input_data, content_type):
    if content_type == "application/json":
        data = json.loads(input_data)
        predictions = data.get("predictions")
        if predictions is None:
            raise ValueError("JSON input does not contain 'predictions' key.")
        return predictions

    elif "text/csv" in content_type:
        if isinstance(input_data, bytes):
            input_data = input_data.decode('utf-8')

        data = StringIO(input_data)
        predictions = pd.read_csv(data, header=None)
        return predictions.values

    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Transforms the prediction into its corresponding category.
    """
    predictions = [1 if x > 0.5 else 0 for x in input_data]

    return [
        (model[prediction], confidence)
        for prediction, confidence in zip(predictions, input_data)
    ]


def parse_confidence(lst, func):
    return [(x[0], func(x[1])) for x in lst]


def output_fn(prediction, accept):
    if accept == "text/csv":
        prediction = parse_confidence(prediction, lambda x: x.item())
        return (
            worker.Response(encoders.encode(prediction, accept), mimetype=accept)
            if worker
            else (prediction, accept)
        )

    if accept == "application/json":
        response = []
        for p, c in prediction:
            response.append({"prediction": p, "confidence": c.item()})

        # If there's only one prediction, we'll return it
        # as a single object.
        if len(response) == 1:
            response = response[0]

        return (
            worker.Response(json.dumps(response), mimetype=accept)
            if worker
            else (response, accept)
        )

    raise Exception(f"{accept} accept type is not supported.")