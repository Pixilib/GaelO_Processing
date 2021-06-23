import hashlib
import numpy as np
import SimpleITK as sitk

from django.conf import settings

from dicom_to_cnn.tools.pre_processing import series
from dicom_to_cnn.model.reader.Nifti import Nifti 
from dicom_to_cnn.model.post_processing.mip.MIP_Generator import MIP_Generator 
from dicom_to_cnn.model.fusion.Fusion import Fusion


class DicomToCnn:
    def to_nifti(self,folder_path: str):
        """[Get DICOM seerie path and transform to nifti]

        Args:
            folder_path (str): [DICOM series folder path]
        """
        data_path = settings.STORAGE_DIR
        path = folder_path       
        nifti=series.get_series_object(path)        
        nifti_str=str(nifti)
        nifti_str=nifti_str[1:44]
        if nifti_str=='dicom_to_cnn.model.reader.SeriesCT.SeriesCT':  
            nifti.get_instances_ordered()   
            nifti.get_numpy_array()
            image_md5 = hashlib.md5(str(nifti).encode())
            image_id = image_md5.hexdigest()
            img=nifti.export_nifti(data_path+'/image/image_'+image_id+'_CT.nii')
        if nifti_str=='dicom_to_cnn.model.reader.SeriesPT.SeriesPT':
            nifti.get_instances_ordered()   
            nifti.get_numpy_array()
            nifti.set_ExportType('suv')
            image_md5 = hashlib.md5(str(nifti).encode())
            image_id = image_md5.hexdigest()
            img=nifti.export_nifti(data_path+'/image/image_'+image_id+'_PT.nii')


    def generate_mip(self,idImage:str):
        data_path = settings.STORAGE_DIR
        directory=settings.STORAGE_DIR+'/image'
        path_ct =data_path+'/image/image_'+idImage+'.nii'
        objet = Nifti(path_ct)
        resampled = objet.resample(shape=(256, 256, 1024))
        resampled[np.where(resampled < 500)] = 0 #500 UH
        normalize = resampled[:,:,:,]/np.max(resampled)
        mip_generator = MIP_Generator(normalize)
        mip_generator.project(angle=0)
        print(mip_generator.project(angle=0))
        mip_generator.save_as_png('image_2D_'+idImage,  directory, vmin=0, vmax=1)       

    def fusion(self,idImage:str):
        data_path = settings.STORAGE_DIR
        path_ct=data_path+'/image/image_'+idImage+'_CT.nii'
        path_pt=data_path+'/image/image_'+idImage+'_PT.nii'
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

        image = sitk.GetImageFromArray(data, True)
        sitk.WriteImage(image, data_path+'/image/image_fusion.nii')