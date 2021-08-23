import hashlib
import base64
import os
import json

from django.conf import settings
from django.http import JsonResponse
from django.http.response import HttpResponse


def handle(request):
    method = request.method
    if(method == 'POST'):
        data = request.read()
        img_id = create_image(data)
        return JsonResponse({'id': img_id})
    if(method == 'GET'):
        id_list = get_image_id()
        # id_list=json.dumps(id_list)
        # print(type(id_list))
        return JsonResponse(id_list, safe=False)


def create_image(data: str) -> str:
    """[Store an image with unique ID]

       Content of the POST request

        Create a new instance image with unique ID in HASH  
        Returns: 
        [str]:[id image]
        """
    data_path = settings.STORAGE_DIR
    image_md5 = hashlib.md5(str(data).encode())
    image_id = image_md5.hexdigest()
    image = open(data_path+'/image/image_'+image_id+'.nii', 'wb')
    image.write(data)
    image.close()
    return image_id
    


def get_image_id() -> list:
    """

    Returns:
        [list]: [Return a list with all Image id content in the storage]
    """
    storage_folder = settings.STORAGE_DIR+'/image'
    list_id = []   
    i=0 
    for f in os.listdir(storage_folder):   
        if os.path.isfile(os.path.join(storage_folder, f)):
            for i in range(len(list_id)):
                i=i+1
            id = f[6:-4]
            list_id.append(id)#{"id"+str(i):id}
    return list_id
