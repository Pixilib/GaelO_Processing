import hashlib
import SimpleITK as sitk
from SimpleITK.SimpleITK import ImageReaderBase, Transform
from SimpleITK.extra import GetImageFromArray
import numpy as np
from numpy.core.arrayprint import array2string
from numpy.core.fromnumeric import reshape, resize
import tensorflow as tf
from abc import ABC

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
        method=str(dictionaire['method'])
        self.method=method
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
        if self.method=="save_as_mask":
            save=InferencePTSegmentation
            id_mask=save.__save_to_nifti(image)
        if self.method=='save_as_dicomseg':
            mask_path=settings.STORAGE_DIR+'/mask/mask_9d6b3c32f48ad50a34778618a6e9303e.nii'
            serie_path=settings.STORAGE_DIR+'dicom/1.2.276.0.7230010.3.1.4.2267612261.1368.1197888473.4/1.2.276.0.7230010.3.1.4.2267612261.1368.1197888484.513/7030106071217114802'
            save=InferencePTSegmentation
            save.__save_to_dicomseg(image,serie_path)
        if self.method=='save_as_rtstruct':
            save=InferencePTSegmentation
            serie_path=settings.STORAGE_DIR+'dicom/1.2.276.0.7230010.3.1.4.2267612261.1368.1197888473.4/1.2.276.0.7230010.3.1.4.2267612261.1368.1197888484.513/7030106071217114802'
            save.__save_to_rtsruct(image,serie_path)
        #return result

    def get_input_name(self) -> str:
        return 'input'
    
    def get_model_name(self) -> str:
        return 'pt_segmentation_model'

    def __save_to_nifti(mask:sitk.Image):
        data_path = settings.STORAGE_DIR
        mask_md5 = hashlib.md5(str(mask).encode())
        id_mask = mask_md5.hexdigest()
        sitk.WriteImage(mask, data_path+'/mask/mask_'+id_mask+'.nii')
        return id_mask

    def __save_to_dicomseg(mask:sitk.Image,dicom_path:str):        
        dicomseg=DICOMSEG_Writer       
        
        #directory_path=settings.STORAGE_DIR+'/dicom/dicomseg'
        #filename='dicomseg_9d6b3c32f48ad50a34778618a6e9303e'
        #dicomseg.save_file(self=dicomseg,filename=filename,directory_path=directory_path)

    def __save_to_rtsruct(mask:sitk.Image,serie_path:str):
        rtsrtuct=RTSS_Writer
        rtsrtuct.mask_img=mask
        rtsrtuct.serie_path=serie_path
      