import os
import urllib.request as request
import zipfile
from cnnclassifier import  logger
from cnnclassifier.utils.common import get_size
from cnnclassifier.entity.config_entity import DataIngestionConfig
from cnnclassifier.constants import *

class DataIngestion :
    def __init__(
    self ,
    config  : DataIngestionConfig ):
        self.config = config

    def Download_file (self) :
        if not os.path.exists(self.config.local_data_file) :
            filenames ,  headers = request.urlretrieve(
                url = self.config.source_URL ,
                filename= self.config.local_data_file ,
            )
        else :
            logger.info(f"files already exist of size{get_size(Path(self.config.local_data_file))}")

    def extract_zip_file (self) :
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file) as zip_ref :
            zip_ref.extractall(unzip_path)
