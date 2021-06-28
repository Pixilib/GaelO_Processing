from django.test import TestCase
from django.test import Client


class test_tensorflow_controller(TestCase):
    #def test_predict(self):
        #setting = {"id": "9d1f6b1606e0b318a8dff0928564ccc4"}
        #c = Client()
        #response = c.post('/app/models/aquisition_field_model/inference',
          #                setting, content_type='application/json')
        #self.assertTrue(response.status_code == 200)
        #print('test predict for aquisition_field_model validate')


    def test_predict(self):
        setting = {"id": ["0fc1be2550af09dee640a0edc041f499","2332ea61df354d2874bb35346f4dbe81"],"method":"save_as_dicomseg"}
        
        c = Client()
        response = c.post('/app/models/pt_segmentation_model/inference', setting, content_type='application/json')
        self.assertTrue(response.status_code == 200)
        print('test predict pt_segmentation_model for tensorflow validate')
