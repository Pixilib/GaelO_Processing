from django.test import TestCase
from django.test import Client


class MyTest(TestCase):

    def test_dl_mask_file_controller(self):
        c = Client()
        response = c.get('/app/masks/8/file')
        self.assertTrue(response.status_code == 200)
        print('test dl_file_mask_controller validate')
