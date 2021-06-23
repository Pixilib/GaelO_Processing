import requests
import json

from django.http import JsonResponse
from django.conf import settings


def handle(request, model_name):
    method = request.method
    if(method == 'GET'):
        metadata = get_metadata(model_name)
        return JsonResponse(metadata)


def get_metadata(model_name) -> dict:
    #TODO : A PASSER DANS INFERENCE CLASSE
    metadata = requests.get('http://'+settings.TENSORFLOW_SERVING_ADDRESS+':' +
                            settings.TENSORFLOW_SERVING_METADATA+'/v1/models/'+model_name+'/metadata')
    metadata = metadata.content
    metadata = metadata.decode('utf-8')
    metadata = json.loads(metadata)
    return metadata
