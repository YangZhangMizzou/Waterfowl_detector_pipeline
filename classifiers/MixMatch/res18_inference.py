import os
import glob
import cv2
from torchvision import transforms,utils
import torch
import models.wideresnet as wmodels
from PIL import Image
import numpy as np
import dataset.waterfowl as dataset
import torch.nn as nn
import torch.nn.functional as F
import random

import sys
import torchvision.models as models
from tqdm import tqdm
import json

category_dict = {
    "0": "American Widgeon_Female",
    "1": "American Widgeon_Male",
    "2": "Canada Goose",
    "3": "Canvasback_Male",
    "4": "Coot",
    "5": "Gadwall",
    "6": "Green-winged teal",
    "7": "Mallard Female",
    "8": "Mallard Male",
    "9": "Not a bird",
    "10": "Pelican",
    "11": "Pintail_Female",
    "12": "Pintail_Male",
    "13": "Ring-necked duck Female",
    "14": "Ring-necked duck Male",
    "15": "Scaup_Male",
    "16": "Shoveler_Female",
    "17": "Shoveler_Male",
    "18": "Snow",
    "19": "Unknown",
    "20": "White-fronted Goose"
}
category_dict_reverse = {v:k for k,v in category_dict.items()}

test_transform = transforms.Compose([
    dataset.ToTensor(),
])
device = torch.device('cuda')

def window_jittering(box,mega_image,num_box=5):
    # h,w,c = mega_image.shape
    w, h = mega_image.size
    x1,y1,x2,y2 = box
    jittering_box = []
    jittering_box.append([int(x1),int(y1),int(x2),int(y2)])
    for _ in range(num_box-1):
        x1_random = random.random()*0.4-0.1
        x2_random = random.random()*0.4-0.1
        y1_random = random.random()*0.4-0.1
        y2_random = random.random()*0.4-0.1
        x1_jitter = int(max(x1-(x2-x1)*x1_random,0))
        x2_jitter = int(min(x2+(x2-x1)*x2_random,w))
        y1_jitter = int(max(y1-(y2-y1)*y1_random,0))
        y2_jitter = int(min(y2+(y2-y1)*y2_random,h))
        jittering_box.append([x1_jitter,y1_jitter,x2_jitter,y2_jitter])
    return jittering_box

def predict_methods(box,category,mega_image,method = 'baseline'):
    if category == 'Snow/Ross Goose' or category == 'Snow/Ross Goose (blue)':
        category = 'Snow'
    elif category not in category_dict.values():
        category = 'Unknown'
    bird_crop = prepare_data_res18(mega_image,box)
    out = res18_model(bird_crop)
    out = torch.topk(out,1, dim=1)[1].squeeze().data
    pred_cate = category_dict[str(out.cpu().numpy())]
    if pred_cate == category:
        counter.append(1)
    else:
        counter.append(0)

def get_model_res18(checkpoint_dir):
    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc =  nn.Sequential(resnet18.fc,nn.Dropout(p = 0.2),nn.Linear(in_features=1000, out_features=21, bias=True))
    resnet18.load_state_dict(torch.load(checkpoint_dir,map_location=device))
    resnet18 = resnet18.eval()
    resnet18.to(device)
    resnet18.eval()
    return resnet18
   

def prepare_data_res18(mega_image,box):
    test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((128,128)),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    [x1,y1,x2,y2] = box
    cropped_image = mega_image[y1:y2,x1:x2]
    inputs = test_transform(cropped_image).unsqueeze(0).to(device)
    return inputs

root_dir = '/home/yang/semi_classification/datasets/Bird_I_Waterfowl_SpeciesClassification/test'
image_dirs = glob.glob(root_dir+'/*.jpg')
checkpoint_dir = '/home/yang/semi_classification/MixMatch-pytorch-master/Res18_Bird_I/model.pth'
res18_model = get_model_res18(checkpoint_dir)
counter = []
for image_dir in image_dirs:
    txt_dir = image_dir.replace('.jpg','_class.txt')
    mega_image = cv2.imread(image_dir)
    mega_image = cv2.cvtColor(mega_image,cv2.COLOR_BGR2RGB)
    with open(txt_dir,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            [_,category,x1,y1,x2,y2] = line.split(',')
            box = [int(x1),int(y1),int(x2),int(y2)]
            predict_methods(box,category,mega_image)
print(sum(counter)/len(counter))