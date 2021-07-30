import os
import shutil
import tempfile
import hashlib
from zipfile import ZipFile

from django.conf import settings

from dicom_to_cnn.tools.pre_processing import series

from django.http import HttpResponse

def handle(request):
    method = request.method 
    if(method=='POST'):
        zip_file=request.read()
        filename=get_dicom_zip(zip_file)
        return HttpResponse(status=200)

def get_dicom_zip(zip_file):
    """[Get dicom series zip, unzip files and create nifti]

    Args:
        zip_file ([byte]): [content of zip file]

    Returns:
        [id_image]: [The id of the nifti image created]
    """
    data_path = settings.STORAGE_DIR
    destination=tempfile.mkdtemp(prefix='dicom_zip_')
    file = open(destination+'/dicom.zip', 'wb')
    file.write(zip_file)
    file.close()
    
    #unzip_file and save dicom series
    image_md5 = hashlib.md5(str(file.name).encode())
    dicom_id = image_md5.hexdigest()
    os.mkdir(settings.STORAGE_DIR+'/dicom/dicom_serie_'+dicom_id)
    destination=settings.STORAGE_DIR+'/dicom/dicom_serie_'+dicom_id
    with ZipFile(file.name) as my_zip:
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

    #create and save nifti  
        
    nifti=series.get_series_object(destination)        
    nifti_str=str(nifti)
    nifti_str=nifti_str[1:44]
    if nifti_str=='dicom_to_cnn.model.reader.SeriesCT.SeriesCT':  
        nifti.get_instances_ordered()   
        nifti.get_numpy_array()
        image_md5 = hashlib.md5(str(nifti).encode())
        image_id = image_md5.hexdigest()
        img=nifti.export_nifti(data_path+'/image/image_'+image_id+'_CT.nii')
    if nifti_str=='dicom_to_cnn.model.reader.SeriesPT.SeriesPT':
        nifti.get_instances_ordered()   
        nifti.get_numpy_array()
        nifti.set_ExportType('suv')
        image_md5 = hashlib.md5(str(nifti).encode())
        image_id = image_md5.hexdigest()
        img=nifti.export_nifti(data_path+'/image/image_'+image_id+'_PT.nii')
    return image_id