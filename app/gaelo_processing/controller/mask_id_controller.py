from ..data_object.pyradiomics_response import NumpyArrayEncoder
from ..models.SitkImage import get_metadata_dictionary

import os
import SimpleITK as sitk

from django.http import HttpResponse, JsonResponse
from django.conf import settings


def handle(request, idMask=''):
    method = request.method
    if(method == 'DELETE'):
        delete_mask(idMask)
        return HttpResponse(status=200)
    if(method == 'GET'):
        metadata = get_metadata(idMask)
        return JsonResponse(metadata, NumpyArrayEncoder)


def delete_mask(idMask: str) -> None:
    """[Delete the Mask]

        Args:
            idMask (int): [Input idMask]

        Removes the specified mask     
        """
    os.remove(settings.STORAGE_DIR+"/mask/mask_"+idMask+".nii")


def get_metadata(idMask: str) -> dict:
    """[Get the metadata from an mask]

    Args:
        idImage (str): [Input mask]

    Returns:
        dict: [return formated dictionary ready ready to be sent as a JSON]
    """
    path = settings.STORAGE_DIR+"/mask/mask_"+idMask+".nii"
    image = sitk.ReadImage(path)
    return get_metadata_dictionary(image)
