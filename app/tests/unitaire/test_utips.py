import zipfile
from django.test import TestCase
from ...gaelo_processing.utips.utips import Utips

class MyTest(TestCase):
    def test_extract_zip_files(self):
        file_zip=Utips.unzip_file(self,"C:/Users/Nicolas/Desktop/test.zip")
        print('Validate')