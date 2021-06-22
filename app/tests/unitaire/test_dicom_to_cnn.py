from django.test import TestCase

from ...gaelo_processing.data_transform.dicom_to_nifti import DicomToCnn

class test_dicom_to_cnn(TestCase):
    # def test_to_nifti(self):
    #     DicomToCnn.to_nifti(self,'C:/Users/Nicolas/AppData/Local/Temp/gaelo_pross_unzip_jz60e_9p')

    def test_generate_mip(self):
        DicomToCnn.generate_mip(self,'1.3.12.2.1107.5.1.4.45520.30000013102404450495300000013')
        print('test dicom-to-cnn validate')