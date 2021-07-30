import os

from django.http import HttpResponse
from django.conf import settings


def handle(request, idSerie=''):
    method = request.method
    if(method == 'DELETE'):
        delete_dicom(idSerie)
        return HttpResponse(status=200)

def delete_dicom(idSerie: str) -> None:
    """[Delete the Dicom]

        Args:
            idSerie (str): [Input idSerie]

        Removes the specified dicom serie     
        """
    os.rmdir(settings.STORAGE_DIR+"/dicom/dicom_serie_"+idSerie)


