from django.test import TestCase

from ...gaelo_processing.controller import models_metadata_controller


class test_models_metadata_controller(TestCase):

    def test_get_metadata(self):
        data = models_metadata_controller.get_metadata(
            'pt_segmentation_model')
        print(data)
        print('test_models_metadata validate')
