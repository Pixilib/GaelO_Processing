from django.http import JsonResponse

from ..models.Index_model import model_list


def handle(request):
    """[Get models list]

    Returns:
        [JsonResponse]: [Json which contains the list of inference models ]
    """
    method = request.method
    if(method == 'GET'):
        list = model_list
        print(list)
        return JsonResponse(list)