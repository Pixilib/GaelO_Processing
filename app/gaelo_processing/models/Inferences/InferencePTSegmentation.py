import SimpleITK as sitk
import numpy as np
import tensorflow as tf

from django.conf import settings
from tensorflow.core.framework.tensor_pb2 import TensorProto
from ..AbstractInference import AbstractInference

from dicom_to_cnn.model.fusion import Fusion

class InferencePTSegmentation(AbstractInference):  

    def pre_process(self, dictionaire:dict) -> TensorProto:
        idPT=str(dictionaire['id'][0])
        idCT=str(dictionaire['id'][1])
        data_path = settings.STORAGE_DIR
        path_ct=data_path+'/image/image_'+idCT+'_CT.nii'
        path_pt=data_path+'/image/image_'+idPT+'_PT.nii'
        img_ct=sitk.ReadImage(path_ct)
        img_pt=sitk.ReadImage(path_pt)
        fusion_object=Fusion()
        fusion_object.set_origin_image(img_pt)
        fusion_object.set_target_volume((128,128,256),(4.0, 4.0, 4.0),(1,0,0,0,1,0,0,0,1))
        ct_resampled = fusion_object.resample(img_ct,-1000.0)
        pt_resampled = fusion_object.resample(img_pt,0)

        ct_array = sitk.GetArrayFromImage(ct_resampled)
        pt_array = sitk.GetArrayFromImage(pt_resampled)
        data = np.zeros((256, 128, 128, 2), dtype='float32')
        data[:,:,:,0] = np.array(ct_array).astype('float32')
        data[:,:,:,1] = np.array(pt_array).astype('float32')

        # image = sitk.GetImageFromArray(data, True)
        # sitk.WriteImage(image, data_path+'/image/image_fusion.nii')
        #continuer vers inference
       
        return tf.make_tensor_proto(data, shape=[1,128,128,256,2])

    def post_process(self, result) -> dict:       
        print(result)
        return result

    def get_input_name(self) -> str:
        return 'input_1'
    
    def get_model_name(self) -> str:
        return 'pt_segmentation_model'