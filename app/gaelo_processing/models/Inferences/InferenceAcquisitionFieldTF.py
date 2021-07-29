import numpy as np
import tensorflow as tf
from django.conf import settings

from tensorflow.core.framework.tensor_pb2 import TensorProto
from ..AbstractTensorflow import AbstractTensorflow

from skimage.transform import resize

from dicom_to_cnn.model.reader.Nifti import Nifti 
from dicom_to_cnn.model.post_processing.mip.MIP_Generator import MIP_Generator 


class InferenceAcquisitionFieldTF(AbstractTensorflow):  

    def pre_process(self, dictionnaire:dict) -> TensorProto:
        """[summary]

        Args:
            dictionnaire (dict): [Dictionary containing the id of the images]

        Returns:
            TensorProto: [description]
        """
        dict=dictionnaire
        data_path = settings.STORAGE_DIR
        idImage=str(dict['id'])
        nifti_path =data_path+'/image/image_'+idImage+'_CT.nii'
        resampled_array = Nifti(nifti_path).resample((256, 256, 1024))
        resampled_array[np.where(resampled_array < 500)] = 0 #500 UH
        normalize = resampled_array[:,:,:,]/np.max(resampled_array) #normalize
        mip_generator = MIP_Generator(normalize)
        mip = mip_generator.project(angle=0)
        mip = np.expand_dims(mip, -1)
        mip = np.array(mip).astype('float32')

        #objet = Nifti(path_ct)
        #resampled = objet.resample(shape=(256, 256, 1024))
        #mip_generator = MIP_Generator(resampled)
        #array=mip_generator.project(angle=0)
        #Ici va disparaitre avec un nouvel entrainement sur des tailles natives et tete en bas (reference dicom)
        #et image normalisee de 0 a 1024 puis normalise de 0 a 1
        #array = np.flip(array, 0)
        #array = resize(array, (503, 136))
        #fin
        #array[np.where(array < 500)] = 0 #500 UH
        #array[np.where(array > 1024)] = 0 #1024 UH
        #array = array[:,:,]/1024
        #array = np.array(array).astype('float32')
        return tf.make_tensor_proto(mip, shape=[1,1024,256,1])

    def post_process(self, result) -> dict:
        resultDict = {}
        # print(result)
        for output in result.outputs:
            outputResult = result.outputs[output]
            resultDict[str(output)] = list(outputResult.float_val)
            
        dict={}
        #lef_arm down true/false
        if resultDict['left_arm'][0]>resultDict['left_arm'][1]:
            left_arm=True
        else :
            left_arm=False
        #right_arm down true/false
        if resultDict['right_arm'][0]>resultDict['right_arm'][1]:
            right_arm=True
        else :
            right_arm=False
        #vertex true/false
        if resultDict['head'][0]>resultDict['head'][1]:
            head=True
        else :
            head=False
        
        maxPosition = resultDict['legs'].index(max(resultDict['legs']))
        if(maxPosition == 0) : leg='Hips'
        if(maxPosition == 1) : leg='Knee'
        if(maxPosition == 2) : leg='Foot'

        dict={'left_arm_down':left_arm,'right_arm_down':right_arm,'head':head,'leg':leg}
        print(dict)
        return dict

    def get_input_name(self) -> str:
        return 'input_1'
    
    def get_model_name(self) -> str:
        return 'aquisition_field_model'
        
