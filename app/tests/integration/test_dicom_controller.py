from django.test import TestCase
from django.test import Client
from django.conf import settings
 
 
class test_dicom_controller(TestCase):

    def test_delete_dicom(self):
        c = Client()
        response = c.delete('/app/dicom/52')
        self.assertTrue(response.status_code == 200)
        print('test delete_dicom validate')