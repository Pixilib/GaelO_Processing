import tempfile
import os
import shutil
import hashlib

from django.conf import settings

from zipfile import ZipFile


class Utips:
    def unzip_file(zip_file: str) -> None:
        """[Extract the zip files]

        Args:
            zip_file (str): [The zip link]

        Returns:
            dict : [The id of dicom series]
        """
       
        image_md5 = hashlib.md5(str(zip_file).encode())
        dicom_id = image_md5.hexdigest()
        os.mkdir(settings.STORAGE_DIR+'/dicom/dicom_serie_'+dicom_id)
        destination=settings.STORAGE_DIR+'/dicom/dicom_serie_'+dicom_id
        with ZipFile(zip_file) as my_zip:
            for member in my_zip.namelist():
                filename = os.path.basename(member)
                # skip directories
                if not filename:
                    continue
                # copy file (taken from zipfile's extract)
                source = my_zip.open(member)
                
                target = open(os.path.join(destination, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)
                dict={'id_dicom_serie':dicom_id}
            return dict