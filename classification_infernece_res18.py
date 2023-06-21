import sys
import torch.nn as nn
import cv2
import torch
from torchvision import transforms, utils
import torchvision.models as models
import json
import glob
import os

def res18_classifier_inference(model_dir,category_index_dir,image_list,detection_root_dir,device):
    text_out_dir = detection_root_dir
    test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((128,128)),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    with open(category_index_dir,'r') as f:
        category_dict = json.load(f)
    category_list = list(category_dict.values())

    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc =  nn.Sequential(resnet18.fc,nn.Dropout(p = 0.2),nn.Linear(in_features=1000, out_features=len(category_dict), bias=True))
    resnet18.load_state_dict(torch.load(model_dir,map_location=device))
    resnet18 = resnet18.eval()
    resnet18.to(device)
    resnet18.eval()


    for image_dir in image_list:
        file_name = os.path.basename(image_dir)
        pd_dir = os.path.join(detection_root_dir,file_name.split('.')[0]+'.txt')
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        with open(pd_dir,'r') as f:
            data = f.readlines()
        pred_data = []
        for line in data:
            line = line.replace('\n','').split(',')
            coord = [int(i) for i in line[2:]]
            coord= [max(0,coord[0]),max(0,coord[1]),min(image.shape[1],coord[2]),min(image.shape[0],coord[3]),]
            cropped_image = image[coord[1]:coord[3],coord[0]:coord[2],:]
            inputs = test_transform(cropped_image).unsqueeze(0).to(device)
            out = resnet18(inputs)
            out = torch.topk(out,1, dim=1)[1].squeeze().data
            preds = category_list[out]
            pred_data.append([preds,line[1],line[2],line[3],line[4],line[5]])
        with open(os.path.join(text_out_dir,'{}.txt'.format(file_name.split('.')[0])),'w') as f:
            for line in pred_data:
                line = ','.join(line)
                f.writelines(line+'\n')
