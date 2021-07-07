from django.test import TestCase
from django.test import Client
from django.conf import settings
 
 
class test_dicom_id_controller(TestCase):

    def test_get_id(self):
        c = Client()
        response = c.get('/app/dicoms')
        self.assertTrue(response.status_code == 200)
        print('test get_id for dicom validate')