# import base64
# import SimpleITK as sitk

# from django.test import TestCase
# from django.test import Client
# from django.conf import settings

# class MyTest(TestCase):

#     def test_post(self):
#         c = Client()
#         setting = {"setting": {
#             "minimumROIDimensions": 2,
#             "minimumROISize": 1,
#             "binWidth": 3.5,
#             "normalize": False,
#             "force2D": True,
#             "label": 1,
#             "geometryTolerance": 0.5,
#             "correctMask": False,
#             "additionalInfo": True,
#             "label_channel": 0,
#             "binCount": 5,
#             "normalizeScale": 1.5,
#             "removeOutliers": 1,
#             "resampledPixelSpacing": [1, 2, 3],
#             "interpolator": "sitkLinear"
#         }
#         }
#         response = c.post('/app/radiomics/image/8/mask/8',setting, content_type='application/json')
#         self.assertTrue(response.status_code == 200)

#     def test_delete_image(self):
#         image = open(settings.STORAGE_DIR+'/image/image_3.nii', 'wb')
#         image.close()
#         c = Client()
#         response = c.delete('/app/image/3')
#         self.assertTrue(response.status_code == 200)

#     def test_delete_mask(self):
#         mask = open(settings.STORAGE_DIR+'/mask/mask_3.nii', 'wb')
#         mask.close()
#         c = Client()
#         response = c.delete('/app/mask/3')
#         self.assertTrue(response.status_code == 200)