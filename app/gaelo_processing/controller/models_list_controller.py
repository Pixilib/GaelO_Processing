from django.http import JsonResponse

from ..models.Index_model import model_list


def handle(request):
    method = request.method
    if(method == 'GET'):
        list = model_list
        return JsonResponse(list)
