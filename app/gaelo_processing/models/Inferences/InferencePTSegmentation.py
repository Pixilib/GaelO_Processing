import hashlib
import SimpleITK as sitk
from SimpleITK.SimpleITK import ImageReaderBase, Transform
from SimpleITK.extra import GetImageFromArray
import numpy as np
from numpy.core.arrayprint import array2string
from numpy.core.fromnumeric import reshape, resize
import tensorflow as tf

from django.conf import settings
from tensorflow.core.framework.tensor_pb2 import TensorProto
from ..AbstractInference import AbstractInference

from dicom_to_cnn.model.fusion.Fusion import Fusion
from dicom_to_cnn.model.segmentation.dicom_seg.DICOMSEG_Writer import DICOMSEG_Writer
from dicom_to_cnn.model.segmentation.rtstruct.RTSS_Writer import RTSS_Writer

class InferencePTSegmentation(AbstractInference):  

    def pre_process(self, dictionaire:dict) -> TensorProto:
        idPT=str(dictionaire['id'][0])
        idCT=str(dictionaire['id'][1])
        methode=str(dictionaire['methode'])
        self.methode=methode
        data_path = settings.STORAGE_DIR
        path_ct=data_path+'/image/image_'+idCT+'_CT.nii'
        path_pt=data_path+'/image/image_'+idPT+'_PT.nii'
        img_ct=sitk.ReadImage(path_ct)
        img_pt=sitk.ReadImage(path_pt)

        #save original direction, spacing and origin
        self.spacing=img_pt.GetSpacing()
        self.direction=img_pt.GetDirection()
        self.origin=img_pt.GetOrigin()
        self.size=img_pt.GetSize()
        fusion_object=Fusion()
        fusion_object.set_origin_image(img_pt)
        fusion_object.set_target_volume((128,128,256),(4.0, 4.0, 4.0),(1,0,0,0,1,0,0,0,1))
        ct_resampled = fusion_object.resample(img_ct,-1000.0)
        pt_resampled = fusion_object.resample(img_pt,0)
        self.pt_resampled_origin = pt_resampled.GetOrigin()
        self.pt_resampled_spacing = pt_resampled.GetSpacing()
        self.pt_resampled_direction = pt_resampled.GetDirection()

        ct_array = sitk.GetArrayFromImage(ct_resampled)
        pt_array = sitk.GetArrayFromImage(pt_resampled)

        #Normalize PET
        pt_array[np.where(pt_array < 0)] = 0 #0 SUV
        pt_array[np.where(pt_array > 25)] = 25 #25 SUV
        pt_array = pt_array[:,:,]/25

        #Normalize CT
        ct_array[np.where(ct_array < -1000)] = -1000 #-1000 SUV
        ct_array[np.where(ct_array > 1000)] = 1000 #1000 SUV
        ct_array=ct_array+1000
        ct_array = ct_array[:,:,]/2000
        data=np.stack((pt_array,ct_array),axis=-1).astype('float32')
        return tf.make_tensor_proto(data, shape=[1,256,128,128,2])

    def post_process(self, result) -> dict:  
        results=result.outputs['tf.math.sigmoid_4']
        shape=tf.TensorShape(results.tensor_shape)
        array = np.array(results.float_val).reshape(shape.as_list())
        array=np.around(array).astype(np.int16)
        image = sitk.GetImageFromArray(array[0,:,:,:,0])
        
        image.SetDirection(self.pt_resampled_direction)
        image.SetOrigin(self.pt_resampled_origin)
        image.SetSpacing(self.pt_resampled_spacing)
        
        transformation = sitk.ResampleImageFilter()        
        transformation.SetOutputDirection(self.direction)
        transformation.SetOutputOrigin(self.origin)
        transformation.SetOutputSpacing(self.spacing)
        transformation.SetSize(self.size)
        transformation.SetDefaultPixelValue(0.0)
        transformation.SetInterpolator(sitk.sitkNearestNeighbor)
        image=transformation.Execute(image)
        if self.methode=="save_as_mask":
            save=InferencePTSegmentation
            id_mask=save.__save_to_nifti(image)
        #return result

    def get_input_name(self) -> str:
        return 'input'
    
    def get_model_name(self) -> str:
        return 'pt_segmentation_model'

    def __save_to_nifti(image:sitk.Image):
        data_path = settings.STORAGE_DIR
        mask_md5 = hashlib.md5(str(image).encode())
        id_mask = mask_md5.hexdigest()
        sitk.WriteImage(image, data_path+'/mask/mask_'+id_mask+'.nii')
        return id_mask

    def __save_to_rtstruct():
        return
