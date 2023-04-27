import numpy as np
from PIL import Image
import os
import cv2
from tqdm import tqdm
import time
from siamese import Siamese

if __name__ == "__main__":
    model = Siamese()
    ref_img = Image.open("./ref.bmp")
    ref_img = ref_img.resize((256, 256), Image.LANCZOS)
    
    test_path = r"C:\RTJ\Dev\Playground\Siamese-pytorch\img\20230424_af_59"
    
    os.makedirs(test_path+"_score", exist_ok=True)
    imgs = [f for f in os.listdir(test_path) if f.endswith(".bmp") or f.endswith(".png")]
    dst_path = test_path+"_score"
    
    for img_name in imgs:
        img = Image.open(os.path.join(test_path, img_name))
        img = img.resize((256, 256), Image.LANCZOS)
        
        start = time.time()
        probability = model.detect_image(ref_img, img).item()
        print("time: ", time.time()-start)
        score = 1 - probability
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 255)
        src = cv2.imdecode(np.fromfile(os.path.join(test_path, img_name), dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.putText(src, str(round(score, 3)), (50, 50), font, 2, color, 2)
        cv2.imencode('.bmp', src)[1].tofile(os.path.join(dst_path, img_name))
    
    