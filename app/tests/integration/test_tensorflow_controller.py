from django.test import TestCase
from django.test import Client


class test_tensorflow_controller(TestCase):
    def test_predict(self):
        setting = {"id": "1.3.12.2.1107.5.1.4.45520.30000013102404450495300000013"}
        c = Client()
        response = c.post('/app/models/aquisition_field_model_pytorch/inference',
                         setting, content_type='application/json')
        self.assertTrue(response.status_code == 200)
        print('test predict for aquisition_field_model validate')


    #def test_predict(self):
        #setting = {"id": ["2482d6aa8c2323bf5ae5a284f24c5853","a8caacd0392188600ff7ff9212f26632","001"],"sous-segmentation":"False"}
        #c = Client()
        #response = c.post('/app/models/pt_segmentation_model/inference', setting, content_type='application/json')
        #self.assertTrue(response.status_code == 200)
        #print('test predict pt_segmentation_model for tensorflow validate')
