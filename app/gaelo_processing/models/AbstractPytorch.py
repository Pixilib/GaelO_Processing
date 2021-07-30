import grpc

from django.conf import settings
from .AbstractInference import AbstractInference
from .pytorch.serve import inference_pb2
from .pytorch.serve import inference_pb2_grpc


class AbstractPytorch(AbstractInference):
    """[Open GRPC serve for pytorch]

    Args:
        idImage ([str]): [id of image input]

    Returns:
        [dictionary]: [return formated dictionary ready to be sent as a JSON]
    """
   
    def predict(self, idImage:str)->dict:
        #call pre_process
        model_input = self.pre_process(idImage)
        model_name = 'model_classification' # model_name hardcodee
        channel = grpc.insecure_channel(settings.PYTORCH_SERVE_ADDRESS +':'+settings.PYTORCH_SERVE_PORT, options=(('grpc.enable_http_proxy', 0),))
        stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
        data = model_input
        input_data = {'data': data}
        result = stub.Predictions(
            inference_pb2.PredictionsRequest(model_name=model_name, input=input_data))
        #call post_process
        formated_result = self.post_process(result)
        return formated_result
      
