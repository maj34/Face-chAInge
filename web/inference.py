import os
import sys
import glob
import fractions
import random
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from models.classifier import predict_age_gender_race
from models.parsing_model import BiSeNet
from util.face_detect_crop_multi import Face_detect_crop
from util.reverse2original import reverse2wholeimage
from util.norm import SpecificNorm


transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


def detection(img_path):
    MY_HOME_DIR = "."
    crop_size = 224
    app = Face_detect_crop(name='weights', root=MY_HOME_DIR)
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640))
    with torch.no_grad():
        target_image = cv2.imread(img_path)
        tmp = app.get(target_image, crop_size)
        if tmp:
            img_b_whole, b_mat_list, bboxes = tmp
        else:
            return []

    w = target_image.shape[1]
    h = target_image.shape[0]
    
    vx = max(w / 1200, h / 600)
    w = w / vx
    h = h / vx
    
    vx = w / target_image.shape[1]
    vy = h / target_image.shape[0]
    return [[x1*vx, y1*vy, (x2-x1)*vx, (y2-y1)*vy] for x1,y1,x2,y2,_ in bboxes]
    

def face_swap(img_path, user_click_boolean):
    MY_HOME_DIR = "."
    crop_size = 224
    model = create_model(MY_HOME_DIR)
    model.eval()
    app = Face_detect_crop(name='weights', root=MY_HOME_DIR)
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640))
    
    with torch.no_grad():
        target_image = cv2.imread(img_path)
        if len(user_click_boolean) == 0:
            splitted = img_path.split(".")
            cv2.imwrite((".".join(splitted[:-1]) if len(splitted) > 2 else splitted[0]) + "_result." + splitted[-1], target_image)
            return
        img_b_whole, b_mat_list_whole, _ = app.get(target_image, crop_size)
        
        
        user_click = [idx for idx, v in enumerate(user_click_boolean) if v]
        img_b_selected = [img_b_whole[i] for i in user_click if i < len(img_b_whole)]
        b_mat_list = [b_mat_list_whole[i] for i in user_click if i < len(b_mat_list_whole)]
        people_no = len(img_b_selected)
        
        
        age_max_size = {}
        selected_status = {"10F":set(),"20F":set(),"30F":set(),"40F":set(),"50F":set(),"10M":set(),"20M":set(),"30M":set(),"40M":set(),"50M":set()}
        target_id_nonorm_list = []
        cls_labels = predict_age_gender_race(MY_HOME_DIR, img_b_selected)
        for idx, cls in enumerate(cls_labels):
            age, gender, race = cls
            age = str(min(max(int(age[0]) * 10 , 10), 50))
            gender = gender[0]
            
            MAX_SIZE = 5
            status = -1
            while status < 0:
                status = random.randint(1, MAX_SIZE)
                if status in selected_status[age+gender] and len(selected_status[age+gender]) < MAX_SIZE:
                    status = -1
                else:
                    selected_status[age+gender].add(status)
                    break
                
            os.system(f"cp ./static/images/GAN/SRC_{age}_{gender}_{status}.jpg ./static/images/tmp/SRC_{str(idx).zfill(2)}.jpg")
        
        
        source_id_norm_list = []
        source_path = os.path.join(MY_HOME_DIR, 'static/images/tmp/SRC_*')
        source_images_path = sorted(glob.glob(source_path))
        for idx, source_image in enumerate(source_images_path):
            if idx >= people_no:
                break
            person = cv2.imread(source_image)
            person_align_crop, _, _ = app.get(person, crop_size, threshold=0.01)
            img_pil = Image.fromarray(cv2.cvtColor(person_align_crop[0],cv2.COLOR_BGR2RGB))
            img_trans = transformer_Arcface(img_pil)
            img_id = img_trans.view(-1, img_trans.shape[0], img_trans.shape[1], img_trans.shape[2]).cpu()
            latent_id = F.normalize(model.netArc(F.interpolate(img_id, scale_factor=0.5)), p=2, dim=1)
            source_id_norm_list.append(latent_id.clone())
            
        
        result = []
        matrix = []
        original = []
        img_b_tensor = [_totensor(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))[None,...].cpu() for img in img_b_selected]
        for idx in range(people_no):
            res = model(None, img_b_tensor[idx], source_id_norm_list[idx], None, True)[0]
            result.append(res)
            matrix.append(b_mat_list[idx])
            original.append(img_b_tensor[idx])
        
        
        net = BiSeNet(n_classes=19).cpu()
        net.load_state_dict(torch.load(os.path.join(MY_HOME_DIR, 'weights/bisenet.pth'), map_location=torch.device('cpu')))
        net.eval()
        final_img = reverse2wholeimage(original, result, matrix, crop_size, target_image, net, SpecificNorm())
        
        
        splitted = img_path.split(".")
        cv2.imwrite((".".join(splitted[:-1]) if len(splitted) > 2 else splitted[0]) + "_result." + splitted[-1], final_img)
        
        os.system("rm ./static/images/tmp/*.jpg")
        
        
