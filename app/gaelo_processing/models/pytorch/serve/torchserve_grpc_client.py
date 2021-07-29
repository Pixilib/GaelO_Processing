from django.conf import settings
import grpc
import inference_pb2
import inference_pb2_grpc
import management_pb2
import management_pb2_grpc
import sys


def get_inference_stub():
    channel = grpc.insecure_channel('localhost:7070', options=(('grpc.enable_http_proxy', 0),))
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    return stub


def get_management_stub():
    channel = grpc.insecure_channel('localhost:7071')
    stub = management_pb2_grpc.ManagementAPIsServiceStub(channel)
    return stub


def infer(stub, model_name, model_input):
    data = model_input
    input_data = {'data': data}
    print(type(data))
    print(model_name)
    response = stub.Predictions(
        inference_pb2.PredictionsRequest(model_name=model_name, input=input_data))

    try:
        prediction = response.prediction.decode('utf-8')
        print(prediction)
    except grpc.RpcError as e:
        exit(1)


def register(stub, model_name, url):
    print("Registering ", model_name)
    params = {
        'url': url,
        'initial_workers': 1,
        'synchronous': True,
        'model_name': model_name
    }
    try:
        response = stub.RegisterModel(management_pb2.RegisterModelRequest(**params))
        print(f"Model {model_name} registered successfully")
    except grpc.RpcError as e:
        print(f"Failed to register model {model_name}.")
        print(str(e.details()))
        exit(1)


def unregister(stub, model_name):
    try:
        response = stub.UnregisterModel(management_pb2.UnregisterModelRequest(model_name=model_name))
        print(f"Model {model_name} unregistered successfully")
    except grpc.RpcError as e:
        print(f"Failed to unregister model {model_name}.")
        print(str(e.details()))
        exit(1)

def preprocessing(path_image, output_shape, angle):
    objet = Nifti(path_image)
    resampled = objet.resample(shape=output_shape)
    mip_generator = MIP_Generator(resampled)
    array=mip_generator.project(angle=angle)
    array[np.where(array < 500)] = 0 #500 UH
    array[np.where(array > 1024)] = 1024 #1024 UH
    array = array[:,:,]/1024
    array = np.expand_dims(array, axis=0)
    array = array.astype(np.double)
    np_bytes = BytesIO()
    np.save(np_bytes, array, allow_pickle=True)
    np_bytes = np_bytes.getvalue()
    return np_bytes

from dicom_to_cnn.model.reader.Nifti import Nifti 
from dicom_to_cnn.model.post_processing.mip.MIP_Generator import MIP_Generator 
import torch
from torch import nn
import numpy as np 
import os 

from io import BytesIO

if __name__ == '__main__':
    # args:
    # 1-> api name [infer, register, unregister]
    # 2-> model name
    # 3-> model input for prediction
    #creation du stub 
    url = 'D:/data_docker/torchserve/Lorine/grpc_pytorch/model-store/model_classification.mar'
    model_name = 'model_classification'
    #api = globals()['register']
    #api(get_management_stub(), 'model_classification',url)

    #register(model_name, url)

    output_shape= (256,256,1024)
    angle=0
    x ='D:/Code/Gaelo_Processing/app/storage/image/image_2332ea61df354d2874bb35346f4dbe81_CT.nii'
    input = preprocessing(x, output_shape, angle )
    
    infer(get_inference_stub(), model_name, input)
