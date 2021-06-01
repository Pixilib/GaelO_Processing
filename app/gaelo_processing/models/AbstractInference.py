from abc import ABC, abstractmethod
import grpc

from django.conf import settings
from google.protobuf.wrappers_pb2 import Int64Value
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow_serving.apis.model_pb2 import ModelSpec
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis import prediction_service_pb2_grpc

class AbstractInference(ABC):

    def predict(self, idImage:str):
        """[summary]

        Args:
            idImage (str): [description]

        Returns:
            [dictionary]: [return formated dictionary ready ready to be sent as a JSON]
        """
        # call pre_process
        input_tensor=self.pre_process(idImage)
        channel = grpc.insecure_channel(settings.TENSORFLOW_SERVING_ADDRESS+':'+settings.TENSORFLOW_SERVING_PORT)        
        version = Int64Value(value=1)#version hardcodee
        model_spec = ModelSpec(version=version, name=self.get_model_name(), signature_name='serving_default')
        grpc_request = PredictRequest(model_spec= model_spec)
        grpc_request.inputs[self.get_input_name()].CopyFrom(input_tensor)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        result = stub.Predict(grpc_request,10)
        #call post_process
        formated_result=self.post_process(result)
        return formated_result

    @abstractmethod
    def pre_process(self,idImage) -> TensorProto:
        pass

    @abstractmethod
    def post_process(self, result) ->dict:
        pass

    @abstractmethod
    def get_input_name(self) ->str:
        pass

    @abstractmethod
    def get_model_name(self) ->str:
        pass