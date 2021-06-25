from django.test import TestCase
from django.conf import settings

from ...gaelo_processing.data_transform.dicom_to_nifti import DicomToCnn

class test_dicom_to_cnn(TestCase):
    def test_to_nifti(self):
        data_path = settings.STORAGE_DIR
        DicomToCnn.to_nifti(self,data_path+'/dicom/1.2.276.0.7230010.3.1.4.2267612261.1368.1197888473.4/1.2.276.0.7230010.3.1.4.2267612261.1368.1197888473.5/7030106071217114750')

    # def test_generate_mip(self):
    #     DicomToCnn.generate_mip(self,'9d1f6b1606e0b318a8dff0928564ccc4')
    #     print('test dicom-to-cnn.generate_mip validate')

    # def test_fusion(self):
    #     DicomToCnn.fusion(self,'1.3.12.2.1107.5.1.4.45520.30000013102404450495300000013')
    #     print('test dicom-to-cnn.fusion validate')