from abc import ABC, abstractmethod

from tensorflow.core.framework.tensor_pb2 import TensorProto


class AbstractInference(ABC):
    
    @abstractmethod
    def predict(self,idImage):
        pass

    @abstractmethod
    def pre_process(self, idImage) -> TensorProto:
        pass

    @abstractmethod
    def post_process(self, result) -> dict:
        pass

    @abstractmethod
    def get_input_name(self) -> str:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass