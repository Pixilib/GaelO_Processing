import json
import importlib

from django.http import JsonResponse

from ..models.Index_model import model_list
from ..models.AbstractInference import AbstractInference


def handle(request, model_name=''):
    method = request.method
    if(method == 'POST'):
        body_unicode = request.body.decode('utf-8')
        print(type(body_unicode))
        body = json.loads(body_unicode)
        print(type(body))
        tensorflow_response = prediction(body, model_name)
        return JsonResponse(tensorflow_response,safe=False)


def prediction(idImage: str, model_name: str) -> dict:
    """[Get the prediction result for a given image and model]

    Args:
        idImage (str): [Get the name of the model to be used]
        model_name (str): [Get the name of the model to be used]

    Returns:
        dict: [The result of the prediction ]
    """
    inferenceInstance = __getInferenceModel(model_name)  
    results = inferenceInstance.predict(idImage)
    return results


def __getInferenceModel(model_name) -> AbstractInference :
    class_name = model_list[model_name]
    InferenceClass = getattr(importlib.import_module(
        "app.gaelo_processing.models.Inferences."+class_name), class_name)
    return InferenceClass()
