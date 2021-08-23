import hashlib
import SimpleITK as sitk
import numpy as np
import tensorflow as tf

from django.conf import settings
from tensorflow.core.framework.tensor_pb2 import TensorProto
from ..AbstractInference import AbstractInference

from dicom_to_cnn.model.fusion.Fusion import Fusion
from dicom_to_cnn.model.segmentation.rtstruct.RTSS_Writer import RTSS_Writer
from dicom_to_cnn.model.segmentation.dicom_seg.DICOMSEG_Writer import DICOMSEG_Writer
from dicom_to_cnn.model.post_processing.clustering.Watershed import Watershed


class InferencePTSegmentation(AbstractInference):  

    def pre_process(self, dictionnaire:dict) -> TensorProto:
        """[summary]

        Args:
            dictionnaire (dict): [Dictionary containing the id of the images]

        Returns:
            TensorProto: [description]
        """

        idPT=str(dictionnaire['id'][0])
        idCT=str(dictionnaire['id'][1])
        idserie_dicom=str(dictionnaire['id'][2])

        
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
        """[Provide the result in the correct format]

        Args:
            result ([dict]): [Gross result of the prediction]

        Returns:
            dict: [Id of images generated]
        """
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

        if (self.sousSeg=='True'):
            sousSeg=Watershed(image,self.img_pt)
            sousSeg=sousSeg.applied_watershed_model()
            save_nifti=InferencePTSegmentation
            save_nifti.__save_to_nifti(sousSeg)
        else:
        
            #save to nifti
            save_nifti=InferencePTSegmentation
            id_mask=save_nifti.__save_to_nifti(image)
            mask_path=settings.STORAGE_DIR+'/mask/mask_'+id_mask+'.nii'
            serie_path=settings.STORAGE_DIR+'/dicom/dicom_serie_001_pt'
            #save to dicomseg and dicomrt
            save=InferencePTSegmentation
            ids=save.__save_to_dicomseg_rtstruct(mask_path,serie_path)
            dict={'id_mask':id_mask, 'id_dicomseg': ids[0] ,'id_dicomrt':ids[1] }
        
            return dict

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

    def __save_to_dicomseg_rtstruct(mask_path:str,serie_path:str):     
        mask_img = sitk.ReadImage(mask_path) 
        mask_img = sitk.Cast(mask_img, sitk.sitkUInt16)

        #Save at dicomseg
        directory_path_seg=settings.STORAGE_DIR+'/dicom/dicomseg'
        dicomseg = DICOMSEG_Writer(mask_img, serie_path=serie_path)
        dicomseg_md5 = hashlib.md5(str(dicomseg).encode())
        id_dicomseg = dicomseg_md5.hexdigest()    
        filename_seg='dicomseg_'+id_dicomseg
        
        dicomseg.setDictName('dict')
        dicomseg.setBodyPartExaminated('all body')
        dicomseg.setSeriesDescription('description')
        dicomseg.setAutoRoiName()
        dicomseg.save_file(filename_seg, directory_path_seg)

        #Save at dicomrt
        directory_path_rt=settings.STORAGE_DIR+'/dicom/dicomrt'
        rtstruct = RTSS_Writer(mask_img, serie_path=serie_path)
        rtstruct_md5 = hashlib.md5(str(rtstruct).encode())
        id_dicomrt = rtstruct_md5.hexdigest()
        filename_rt='dicomrt_'+id_dicomrt        
        rtstruct.setDictName('dict')
        rtstruct.setBodyPartExaminated('all body')
        rtstruct.setSeriesDescription('description')
        rtstruct.setAutoRTROIInterpretedType()
        rtstruct.setAutoRoiName()
        rtstruct.save_file(filename_rt, directory_path_rt)
        return id_dicomseg,id_dicomrt #{'id_dicomrt': id_dicomrt, 'id_dicomseg': id_dicomseg}