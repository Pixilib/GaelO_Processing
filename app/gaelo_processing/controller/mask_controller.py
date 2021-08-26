import hashlib
import base64
import os

from django.conf import settings
from django.http import JsonResponse


def handle(request):
    method = request.method
    if(method == 'POST'):
        data = request.read()
        id_mask = create_mask(data)
        return JsonResponse({'id': id_mask})
    if(method == 'GET'):
        id_list = get_mask_id()
        return JsonResponse(id_list, safe=False)


def create_mask(data: str) -> str:
    """[Store an mask with unique ID]

       Content of the POST request

        Create a new instance mask with unique ID in HASH 
        Returns:
        [str]:[id mask]
    """
    data_path = settings.STORAGE_DIR  
    mask_md5 = hashlib.md5(str(data).encode())
    mask_id = mask_md5.hexdigest()
    mask = open(data_path+'/mask/mask_'+mask_id+'.nii', 'wb')
    mask.write(data)
    mask.close()
    print(mask_id)
    return mask_id

 


def get_mask_id():
    """

    Returns:
        [list]: [Return a list with all Mask id content in the storage]
    """
    storage_folder = settings.STORAGE_DIR+'/mask'
    list_id = []
    for file in os.listdir(storage_folder):
        if os.path.isfile(os.path.join(storage_folder, file)):
            id = file[5:-4]
            list_id.append(id)
    return list_id
