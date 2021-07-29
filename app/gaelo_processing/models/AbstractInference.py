from abc import ABC, abstractmethod
import grpc
import importlib


from django.conf import settings
from google.protobuf.wrappers_pb2 import Int64Value
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow_serving.apis.model_pb2 import ModelSpec
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis import prediction_service_pb2_grpc


class AbstractInference(ABC):

    # def __getServer(idImage):
    #     server=idImage['server']
    #     class_name = server_list[server]
    #     AbstractClass=getattr(importlib.import_module(
    #         "app.gaelo_processing.models."+class_name), class_name)
    #     return AbstractClass
    
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

    print('je suis dans abstract inference')