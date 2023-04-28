import numpy as np
from PIL import Image
import os
import cv2
from tqdm import tqdm
import time
from siamese import Siamese
from loguru import logger
import sqlite3


LABEL_DB_FILE = "label.db"

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
    model = Siamese()
    ref_img = Image.open("./ref.bmp")
    ref_img = ref_img.resize((256, 256), Image.LANCZOS)
    
    test_path = r"D:\Data\af_dataset"
    
    cnt_img = 0
    dst_path = "./img/out"
    error_cnt = 0
    
    _, labeled_dict = get_labeled_list(test_path)
    
    dirs = [os.path.join(test_path, d) for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
    start = time.time()
    for dir in tqdm(dirs):
        imgs = [f for f in os.listdir(dir) if f.endswith(".bmp") or f.endswith(".png")]
        scores = []
        
        for img_name in imgs:
            img = Image.open(os.path.join(dir, img_name))
            img = img.resize((256, 256), Image.LANCZOS)
            probability = model.detect_image(ref_img, img).item()
            score = 1 - probability
            scores.append(score)
            cnt_img += 1
            
        idx = np.argmax(scores)
        if labeled_dict[os.path.basename(dir)] != imgs[idx]:
            error_cnt += 1
        src = cv2.imdecode(np.fromfile(os.path.join(dir, imgs[idx]), dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imencode('.bmp', src)[1].tofile(os.path.join(dst_path, os.path.basename(dir)+".bmp"))
    print("Time per img: ", round((time.time()-start)/cnt_img, 6))
    print(f"Error rate: {round(error_cnt/len(dirs)*100, 4)}%")