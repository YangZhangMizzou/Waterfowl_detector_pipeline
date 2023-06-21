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

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

def predict_methods(box,category,mega_image,method = 'baseline'):
    if category == 'Snow/Ross Goose' or category == 'Snow/Ross Goose (blue)':
        category = 'Snow'
    elif category not in category_dict.values():
        category = 'Unknown'

    if method == 'baseline':
        bird_crop = prepare_data_mixmatch(mega_image,box)
        out = mixmatch_model(bird_crop)
        _, pred = out.topk(1, 1, True, True)
        pred_cate = category_dict[str(np.array(pred.cpu())[0][0])]
    

    elif method == 'voting':
        jittering_boxes = window_jittering(box,mega_image)
        predictions = []
        for box in jittering_boxes:
            bird_crop = prepare_data_mixmatch(mega_image,box)
            out = mixmatch_model(bird_crop)
            # out = F.softmax(out,dim=1)
            pred_prob, pred_class = out.topk(1, 1, True, True)
            pred_cate = category_dict[str(np.array(pred_class.cpu())[0][0])]
            predictions.append(pred_cate)

        pred_cate = max(set(predictions), key=predictions.count)

    elif method == 'prob_sum':
        jittering_boxes = window_jittering(box,mega_image,10)
        score_list = [0 for _ in range(21)]
        for box in jittering_boxes:
            bird_crop = prepare_data_mixmatch(mega_image,box)
            out = mixmatch_model(bird_crop)
            out = F.softmax(out,dim=1)
            # sigmoid = nn.Sigmoid()
            # out = sigmoid(out)
            pred_prob, pred_class = out.topk(5, 1, True, True)
            pred_prob = pred_prob.cpu().numpy()[0]
            pred_class = pred_class.cpu().numpy()[0]
            for i in range(len(pred_prob)):
                score_list[pred_class[i]] += pred_prob[i]
        pred_cate = category_dict[str(np.argmax(np.array(score_list)))]
            
    if pred_cate == category:
        counter.append(1)
    else:
        counter.append(0)


def create_model(ema=False):
        model = wmodels.WideResNet(num_classes=21)
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

def get_model_mixmatch(checkpoint_dir,ema=True):
    checkpoint = torch.load(checkpoint_dir)
    if ema:
        ema_model = create_model(ema=True)
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        ema_model.to(device)
        print('loaded ema_model')
        return ema_model.eval()
    else:      
        model = create_model() 
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        print('loaded model')
        return model.eval()

   
cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255
def normalize(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def prepare_data_mixmatch(mega_image,box):
    [x1,y1,x2,y2] = box
    bird_crop = mega_image.crop((x1, y1, x2, y2)).resize((32,32))
    bird_crops = np.array([np.array(bird_crop)])
    bird_crops = transpose(normalize(bird_crops))
    bird_crops = test_transform(bird_crops).to(device)
    return bird_crops


root_dir = '/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/example_images/Bird_I_Waterfowl_SpeciesClassification/test'
image_dirs = glob.glob(root_dir+'/*.jpg')
checkpoint_dir = '/home/yangzhang/robert_pipeline/Waterfowl_detector_pipeline/checkpoint/classifier/mixmatch/model_best.pth.tar'
mixmatch_model = get_model_mixmatch(checkpoint_dir)
counter = []
for image_dir in image_dirs:
    txt_dir = image_dir.replace('.jpg','_class.txt')
    mega_image = Image.open(image_dir)
    with open(txt_dir,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            [_,category,x1,y1,x2,y2] = line.split(',')
            box = [int(x1),int(y1),int(x2),int(y2)]
            predict_methods(box,category,mega_image,'prob_sum')
print(sum(counter)/len(counter))
