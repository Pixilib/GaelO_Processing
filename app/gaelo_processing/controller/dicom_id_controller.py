import os

from django.http import JsonResponse
from django.conf import settings


def handle(request):
    method = request.method
    if(method == 'GET'):
        id_list = get_dicom_id()
        print(id_list)
        return JsonResponse(id_list, safe=False)

def get_dicom_id() -> list:
    """

    Returns:
        [list]: [Return a list with all Dicom id content in the storage]
    """
    storage_folder = settings.STORAGE_DIR+'/dicom'
    list_id = []
    for f in os.listdir(storage_folder):
        if os.path.isdir(os.path.join(storage_folder, f)):
            id = f[12::]
            list_id.append(id)
    return list_id
