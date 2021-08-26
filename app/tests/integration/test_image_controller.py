import base64
import io

from django.test import TestCase
from django.test import Client
from django.conf import settings


class test_image_controller(TestCase):

    # def test_create_image(self):
    #     data_path = settings.STORAGE_DIR+"/image/image_8.nii"
    #     data = io.open(data_path, "rb", buffering = 0)
    #     data=data.read()
    #     c = Client()
    #     response = c.post('/app/images', data,
    #                       content_type='image/nii')
    #     self.assertTrue(response.status_code == 200)
    #     print('test create image_validate validate')

    def test_get_id(self):
        c = Client()
        response = c.get('/app/images')
        self.assertTrue(response.status_code == 200)
        print('test get_id for images validate')
