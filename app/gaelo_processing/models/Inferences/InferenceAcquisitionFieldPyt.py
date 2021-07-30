import numpy as np
from django.conf import settings
from io import BytesIO
import json

from dicom_to_cnn.model.reader.Nifti import Nifti 
from dicom_to_cnn.model.post_processing.mip.MIP_Generator import MIP_Generator 
from ..AbstractPytorch import AbstractPytorch

class InferenceAcquisitionFieldPyt(AbstractPytorch):  

    def pre_process(self, dictionnaire:dict) ->BytesIO :
        """[Pre_process for pytorch]

        Args:
            dictionnaire (dict): [Dictionary containing the id of the images]

        Returns:
            BytesIO: [description]
        """
        dict=dictionnaire
        data_path = settings.STORAGE_DIR
        idImage=str(dict['id'])
        path_image =data_path+'/image/image_'+idImage+'_CT.nii'
        objet = Nifti(path_image)
        resampled = objet.resample(shape=(256, 256, 1024))
        mip_generator = MIP_Generator(resampled)
        array=mip_generator.project(angle=0)
        array[np.where(array < 500)] = 0 #500 UH
        array[np.where(array > 1024)] = 1024 #1024 UH
        array = array[:,:,]/1024
        array = np.expand_dims(array, axis=0)
        array = array.astype(np.double)
        np_bytes = BytesIO()
        np.save(np_bytes, array, allow_pickle=True)
        np_bytes = np_bytes.getvalue()
        return np_bytes

    def post_process(self, result) -> dict:
        prediction = result.prediction.decode('utf-8')
        prediction = json.loads(prediction)

        #vertex true/false
        list=prediction[0]
        list2=list[0]
        if list2[0]>list2[1]:
            head=True
        else :
            head=False

        #see hips/knee/foot ?
        list=prediction[1]
        list2=list[0]
        maxPosition = list2.index(max(list2))
        if(maxPosition == 0) : leg='Hips'
        if(maxPosition == 1) : leg='Knee'
        if(maxPosition == 2) : leg='Foot'

        #right_arm down true/false
        list=prediction[2]
        list2=list[0]
        list2[0]
        if list2[0]>list2[1]:
            right_arm=True
        else :
            right_arm=False

        #lef_arm down true/false
        list=prediction[3]
        list2=list[0]
        list2[0]
        if list2[0]>list2[1]:
            left_arm=True
        else :
            left_arm=False
        
        dict={'head':head,'leg':leg,'right_arm':right_arm,'left_arm':left_arm}
        print(dict)
        return dict

        
    def get_input_name(self) -> str:
        pass
    
    def get_model_name(self) -> str:
        return 'aquisition_field_model_pytorch'
        
