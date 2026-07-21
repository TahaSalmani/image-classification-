from bokeh.command.subcommand import Args
from box.exceptions import  BoxValueError
import  os
import  yaml
import  box
import json
import joblib
from box import ConfigBox
from pathlib import Path
from typing import Any
import  base64
from ensure import ensure_annotations
from cnnclassifier import  logger

@ensure_annotations
def read_yaml(path_to_yaml : Path) -> ConfigBox:
    """Reads  YAML file and returns
    Args :
    path_to_YAML (str) : Path like input

    raises :
        ValueError : if YAML is empty
        e : empty file

    Returns:
        configBox : configBox type
"""

    try :
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file : {path_to_yaml} loaded successfully")
            return  ConfigBox(content)
    except BoxValueError :
        raise  ValueError ("YAML is empty")
    except Exception as e :
        raise e

@ensure_annotations
def create_directories (path_to_dir : list , verbose = True) :
    """ Creates directories inside path_to_dir
    Args :
    path_to_dir (list) : list of paths to directories
    ignore_log(bool , optional) : ignore if multiple directories is to be created . Default to False

        """

    for path in path_to_dir :
        os.makedirs(path, exist_ok = True)
        if  verbose :
            logger.info(f"created directory in  : {path}")



@ensure_annotations
def save_json (path : Path , data : dict) :
    """ saves data to JSON file
    Args :
        path (Path) : path to JSON file
        data(dict) : data to save to JSON file

    """
    with open(path, 'w') as f:
        json.dump(data, f , indent=4)
        logger.info(f"json file  saved in  : {path}")

@ensure_annotations
def load_json (path : Path) ->  ConfigBox :
    """ loads JSON file
    Args :
        path (Path) : path to JSON file

    Returns:
        configBox : data as class attributes instead of dict


    """

    with open(path) as f:
        content = json.load(f)
        logger.info(f"json file : {path} loaded successfully")
        return ConfigBox(content)

@ensure_annotations
def save_binary (data:Any, path : Path) :
    """ saves data to binary file
    Args : data(Any) : data to save to binary file
    path (Path) : path to binary file
    """
    joblib.dump(value= data, filename=path)
    logger.info(f"binary file saved in  : {path}")

@ensure_annotations
def load_binary (path : Path) -> Any :
    """ loads binary file
    Args :
            path (Path) : path to binary file

    Returns:
        Any : object stored in binary file

    """
    data = joblib.load(path)
    logger.info(f"binary file loaded in  : {path}")
    return data

@ensure_annotations
def get_size (path : Path) -> str :
    """ returns size of file in kb
    Args :
        path (Path) : path to file

    Returns:
          str : size of file in kb

    """
    size_in_kb = round(os.path.getsize(path) /1024 )
    return f"{size_in_kb} kb"

def decodeimage (imgstring , filename ) :   ## base 64 to pic
    imgdata = base64.b64decode(imgstring)
    with open(filename, 'wb') as f:
        f.write(imgdata)
        f.close()
def encodeimageinbase64 (croppedimagepath) :   ## pic to base 64
    with open(croppedimagepath, 'rb') as f:
        return base64.b64encode(f.read())






