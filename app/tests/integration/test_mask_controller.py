import base64
import io

from django.test import TestCase
from django.test import Client
from django.conf import settings


class test_mask_controller(TestCase):

    def test_create_mask(self):
        data_path = settings.STORAGE_DIR+"/mask/mask_8.nii"
        data = io.open(data_path, "rb", buffering = 0)
        data=data.read()
        c = Client()
        response = c.post('/app/masks', data,
                          content_type='image/nii')
        self.assertTrue(response.status_code == 200)
        print('test create mask validate')

    def test_get_id(self):
        c = Client()
        response = c.get('/app/masks')
        self.assertTrue(response.status_code == 200)
        print('test get_id for mask validate')
