from django.test import TestCase
# from ...gaelo_processing.utips.utips import Utips
from ...gaelo_processing.data_transform.dicom_to_nifti import Dicom_To_Nifti

class MyTest(TestCase):
#     def test_extract_zip_files(self):
#         file_zip=Utips.unzip_file(self,"orthanc.zip")
#         print('Validate')




    def test_to_nifti(self):
        Dicom_To_Nifti.to_nifti('C:/Users/Nicolas/AppData/Local/Temp/gaelo_pross_unzip_k97n346b')
        print('Validate')