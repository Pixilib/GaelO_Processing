from tensorflow.core.framework.tensor_pb2 import TensorProto
from ..AbstractInference import AbstractInference

from dicom_to_cnn.model.fusion import Fusion
from dicom_to_cnn.model.segmentation.rtstruct import *
from dicom_to_cnn.model.segmentation.dicom_seg.DICOMSEG_Writer import DICOMSEG_Writer

class InferencePTSegmentation(AbstractInference):  

    def pre_process(self, idImage:str) -> TensorProto:
        #extraire ct et pt depuis dicom
        #image ct et pt fusions
        return

    def post_process(self, result) -> dict:
        #orga results
        #rtStruct
        return 'hello'

    def get_input_name(self) -> str:
        return 'input_1'
    
    def get_model_name(self) -> str:
        return 'pt_segmentation_model'