import os

from django.conf import settings
from django.http import HttpResponse
from django.http.response import Http404


def handle(request, idImage=''):
    method = request.method
    if(method == 'GET'):
        return download_image_file(idImage)


def download_image_file(idImage: str) -> None:
    """[Download the image file]

    Args:
        idMask (str): [Input image]

    Raises:
        Http404: [If file not found raise 404 error]

    Returns:
        [NONE]
    """
    image_path = settings.STORAGE_DIR+'/image/image_'+idImage+'.nii'
    if os.path.exists(image_path):
        with open(image_path, 'rb') as image:
            response = HttpResponse(image.read(), content_type="image/nii")
            response['Content-Disposition'] = 'inline; filename=' + \
                os.path.basename(image_path)+'.nii'
            return response
    else:
        raise Http404
