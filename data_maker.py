import sqlite3
import shutil
import os 
import sys
import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm
import random

LABEL_DB_FILE = 'label.db'

def select_from_db(db_file, table_name, columns, condition=None):
    """
    Select rows from the table specified by the table_name argument.
    The columns argument should be a comma-separated string of column names,
    and the condition argument should be a string specifying the condition
    for the SELECT statement, e.g. "age > 30".
    """
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    if condition:
        c.execute(f"SELECT {columns} FROM {table_name} WHERE {condition}")
    else:
        c.execute(f"SELECT {columns} FROM {table_name}")
    rows = c.fetchall()
    conn.close()
    return rows

def get_labeled_list(dataset_path: str):
    '''Get the labeled list
    Parameters
    --------------------------------
    dataset_path: str
        Path to the dataset
    '''
    try:
        labeled_list = select_from_db(os.path.join(dataset_path, LABEL_DB_FILE), 'label', 'subfolder, best_image', '1=1')
        logger.debug(f"Load labeled list")
    except:
        logger.error(f"Cannot load the labeled list from [{os.path.join(dataset_path, LABEL_DB_FILE)}]")
    
    labeled_dict = {sub[0]: sub[1] for sub in labeled_list}
    labeled_list = [sub[0] for sub in labeled_list]
    
    return labeled_list, labeled_dict


if __name__ == "__main__":
    dataset_path = r"\\10.99.0.12\pmish\电子光学事业部\Personal\DL_GROUP\datasets\functions\semaf\af_dataset"
    out_path = "./datasets/images_background"
    IMAGE_SIZE = (256, 256)
    
    _, labeled_dict = get_labeled_list(dataset_path)
    
    ref = cv2.imread("ref.bmp")
    
    for i, folder_name in tqdm(enumerate(labeled_dict.keys()), total=len(labeled_dict.keys()), ncols=80):
       
        focused_img_name = labeled_dict[folder_name]
        sub_folder = os.path.join(dataset_path, folder_name)
        focused_img_path = os.path.join(sub_folder, focused_img_name)
        
        focused_img = cv2.imdecode(np.fromfile(focused_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        focused_img = cv2.resize(focused_img, IMAGE_SIZE)
        
        candidates = [f for f in os.listdir(sub_folder) if (f.endswith(".bmp") and f != focused_img_name)]
                
        if len(candidates) > 5:
            samples = random.sample(candidates, 5)
        else:
            samples = candidates

        dst_p0_dir = os.path.join(out_path, folder_name, "sub0")
        dst_p1_dir = os.path.join(out_path, folder_name, "sub1")
        
        os.makedirs(dst_p0_dir, exist_ok=True)
        os.makedirs(dst_p1_dir, exist_ok=True)
        for sample in samples:
            
            sample_img_path = os.path.join(sub_folder, sample)
            sample_img = cv2.imdecode(np.fromfile(sample_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            sample_img = cv2.resize(sample_img, IMAGE_SIZE)

            p0 = focused_img
            p1 = sample_img
            
            cv2.imencode('.png', p0)[1].tofile(dst_p0_dir+ "/" + sample.replace(".bmp", ".png"))
            cv2.imencode('.png', p1)[1].tofile(dst_p1_dir+ "/" + sample.replace(".bmp", ".png"))
            cv2.imencode('.png', ref)[1].tofile(dst_p1_dir+ "/" + ("ref.png"))
            
            
            