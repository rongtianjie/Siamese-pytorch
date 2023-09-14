import numpy as np
from PIL import Image, ImageFilter
import os
import cv2
from siamese import Siamese
from glob import glob
import time
from tqdm import tqdm
import shutil

if __name__ == '__main__':
    
    # path = r"D:\Data\af_dataset_full\20230706_52"
    # output_path = r"out"
    # os.makedirs(output_path, exist_ok=True)
    
    # model = Siamese(model_path = r"logs\loss_2023_08_23_13_13_20\best_epoch_weights.pth")
    
    # for f in glob(os.path.join(path, "*.bmp")):
    #     img = Image.open(f)
    #     img = img.resize((256, 256), Image.LANCZOS)
        
    #     ref_img = img.filter(ImageFilter.GaussianBlur(radius=5))
        
    #     start = time.time()
    #     probability = model.detect_image(ref_img, img).item()
    #     print("time: ", time.time()-start)
        
    #     img = np.array(img)
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.putText(img, str(1-probability), (20, 30), font, 1, 255, 2)
    #     cv2.imwrite(f.replace(path, output_path), img)
    
    path = r"D:\Data\af_dataset_full"
    output_path = r"out"
    model = Siamese(model_path = r"logs\best_epoch_weights.pth")
    
    for dir in tqdm(glob(os.path.join(path, "*"))):
        if os.path.isdir(dir):
            os.makedirs(os.path.join(output_path, os.path.basename(dir)), exist_ok=True)
            score_list = []
            for f in glob(os.path.join(dir, "*.bmp")):
                img_raw = Image.open(f)
                img = img_raw.resize((256, 256), Image.LANCZOS)
                ref_img = img.filter(ImageFilter.GaussianBlur(radius=5))
                
                probability = model.detect_image(ref_img, img).item()
                score_list.append(1-probability)
                img_raw = np.array(img_raw)
                cv2.putText(img_raw, str(round(1-probability, 10)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                cv2.imwrite(f.replace(dir, os.path.join(output_path, os.path.basename(dir))), img_raw)
                
            best_image = glob(os.path.join(dir, "*.bmp"))[np.argmax(score_list)]
            shutil.copy(best_image, os.path.join(output_path, os.path.basename(dir)+".bmp"))