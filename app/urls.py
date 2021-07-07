from django.urls import path

from .gaelo_processing.controller import pyradiomics
from .gaelo_processing.controller import image_controller
from .gaelo_processing.controller import mask_controller
from .gaelo_processing.controller import image_id_controller
from .gaelo_processing.controller import mask_id_controller
from .gaelo_processing.controller import dl_image_file_controller
from .gaelo_processing.controller import dl_mask_file_controller
from .gaelo_processing.controller import tensorflow_controller
from .gaelo_processing.controller import dicom_zip_controller
from .gaelo_processing.controller import models_list_controller
from .gaelo_processing.controller import models_metadata_controller
from .gaelo_processing.controller import dicom_controller
from .gaelo_processing.controller import dicom_id_controller

urlpatterns = [
    path('radiomics/image/<str:idImage>/mask/<str:idMask>',pyradiomics.handle,name='post_radiomics'),
    path('images/<str:idImage>',image_id_controller.handle,name='image_id_controller'),
    path('masks/<str:idMask>',mask_id_controller.handle,name='mask_id_controller'),
    path('images',image_controller.handle,name='image'),
    path('masks',mask_controller.handle,name='mask'),
    path('images/<str:idImage>/file',dl_image_file_controller.handle,name='dl_image_file'),
    path('masks/<str:idMask>/file',dl_mask_file_controller.handle,name='dl_mask_file'),   
    path('models/<str:model_name>/inference', tensorflow_controller.handle,name='get_inference'),
    path('models/<str:model_name>/metadata',models_metadata_controller.handle,name='model_metadata'),
    path('models',models_list_controller.handle,name='get_list_models'),
    path('dicom',dicom_zip_controller.handle,name='get_dicom_zip'),
    path('dicom/<str:idSerie>',dicom_controller.handle,name='del_dicom'),
    path('dicoms',dicom_id_controller.handle, name='get_dicom_ids'),
]