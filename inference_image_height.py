import cv2
import torch
import torchvision.transforms as transforms
import os
import warnings
import glob
import numpy as np
import time
import json
import sys
import pandas as pd
from args import *
from tools import py_cpu_nms,get_sub_image,filter_small_fp
import matplotlib.pyplot as plt
from tqdm import tqdm
from mAP_cal import mAp_calculate,plot_f1_score,plot_mAp
import shutil
from compare_and_draw import compare_draw

#re18 and mixmatch
from classification_infernece_res18 import res18_classifier_inference
from classifiers.MixMatch.mixmatch_classification import mixmatch_classifier_inference
from resnet_pytorch import ResNet
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets


warnings.filterwarnings("ignore")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)


def get_model_conf_threshold ():
    return args.det_conf

def read_csv_info(csv_dir):

    if csv_dir == '':
        return {}

    df = pd.read_csv(csv_dir, usecols=['image_name','height'])
    altitude_dict = {}
    for line in df.values.tolist():
        altitude_dict[line[0].split('.')[0]] = int(line[1])
    return altitude_dict


def get_image_height_model_3(image_name,altitude_dict):

    if altitude_dict == {}:
        return 15,0,30

    image_height = altitude_dict[image_name]
    if image_height<=30:
        return image_height,0,30
    elif image_height<=60:
        return image_height,1,60
    else:
        return image_height,2,90

def get_image_height_model_4(image_name,altitude_dict):

    if altitude_dict == {}:
        return 15,0,15

    image_height = altitude_dict[image_name]
    if  image_height<=15:
        return image_height,0,15
    elif image_height<=30:
        return image_height,1,30
    elif image_height<=60:
        return image_height,2,60
    else:
        return image_height,3,90

def prepare_retina_net(model_dir,kwargs):
    if (kwargs['device']!=torch.device('cuda')):
        print ('loading CPU mode')
        device = torch.device('cpu')
        net = torch.load(model_dir,map_location=device)
        net = net.module.to(device)
    else:
        device = torch.device('cuda')
        net = torch.load(model_dir)
        net.to(kwargs['device'])
        print('check net model_type',next(net.parameters()).device)
    return net

def prepare_yolo_net(model_dir):
    from models.common import DetectMultiBackend
    model = DetectMultiBackend(model_dir, device=device, dnn=False, data=os.path.join('configs','BirdA_all.yaml'), fp16=False)
    return model

def get_detectron_predictor(model_dir):
    #faster
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join('configs','COCO-Detection','faster_rcnn_R_50_FPN_1x.yaml'))
    cfg.MODEL.DEVICE = device_name
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32],[64],[128]]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    cfg.MODEL.WEIGHTS = model_dir
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.det_conf  # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)
    return predictor

def prepare_yolonas(model_dir):
    #yolonas
    import super_gradients

    if device_name == 'cuda':
        return super_gradients.training.models.get('yolo_nas_m',num_classes=1,checkpoint_path=model_dir).cuda()
        # return super_gradients.training.models.get("yolo_nas_m", pretrained_weights="coco").cuda()
    else:
        return super_gradients.training.models.get('yolo_nas_m',num_classes=1,checkpoint_path=model_dir)


def inference_mega_image_yolonas(image_list,model_root, image_out_dir,text_out_dir, visualize, altitude_dict,device,scaleByAltitude, defaultAltitude=[],**kwargs):

    

    record_list = []
    model_15 = os.path.join(model_root,'ckpt_best15m.pth')
    model_30 = os.path.join(model_root,'ckpt_best30m.pth')
    model_60 = os.path.join(model_root,'ckpt_best60m.pth')
    model_90 = os.path.join(model_root,'ckpt_best90m.pth')
    model_list = [model_15,model_30,model_60,model_90]
    net_list = []
    for model_dir in model_list:
        net_list.append(prepare_yolonas(model_dir))

    with tqdm(total = len(image_list)) as pbar:
        for idxs, image_dir in (enumerate(image_list)):
            pbar.update(1)
            start_time = time.time()
            image_name = os.path.split(image_dir)[-1]
            mega_image = cv2.imread(image_dir)
            ratio = 1
            bbox_list = []
            sub_image_list, coor_list = get_sub_image(mega_image, overlap=0.2, ratio=ratio)
            for index, sub_image in enumerate(sub_image_list):
                if scaleByAltitude:
                    image_taken_height,model_index,model_height = get_image_height_model_4(image_name.split('.')[0],altitude_dict)
                    ratio = round(model_height/image_taken_height, 2)
                    selected_model = net_list[model_index]
                sub_image_dir = './tmp.JPG'
                cv2.imwrite(sub_image_dir,sub_image)
                images_predictions = selected_model.predict(sub_image_dir)
                os.remove(sub_image_dir)
                image_prediction = next(iter(images_predictions))
                labels = image_prediction.prediction.labels
                confidences = image_prediction.prediction.confidence
                bboxes = image_prediction.prediction.bboxes_xyxy

                for i in range(len(labels)):
                    label = labels[i]
                    confidence = confidences[i]
                    bbox = bboxes[i]
                    bbox_list.append([coor_list[index][1]+bbox[0], coor_list[index][0]+bbox[1],coor_list[index][1]+bbox[2], coor_list[index][0]+bbox[3], confidence])
            if (len(bbox_list) != 0):
                bbox_list = filter_small_fp(bbox_list)
                bbox_list = np.asarray([box for box in bbox_list])
                box_idx = py_cpu_nms(bbox_list, 0.25)
                selected_bbox = bbox_list[box_idx]
                selected_bbox = sorted(selected_bbox,key = lambda x: x[4],reverse = True)

            else:
                selected_bbox = []

            txt_name = os.path.basename(image_dir).split('.')[0]+'.txt'
            with open(os.path.join(text_out_dir,txt_name), 'w') as f:
                if (len(selected_bbox) != 0):
                    for box in selected_bbox:
                        f.writelines('bird,{},{},{},{},{}\n'.format(
                            box[4], int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            try:
                re = read_LatLotAlt(image_dir)
            except:
                re = {'latitude':0.0,
                      'longitude':0.0,
                      'altitude':0.0}
            record_list.append([os.path.basename(image_dir),kwargs['date_list'][idxs],kwargs['location_list'][idxs],
                               str(image_taken_height),re['latitude'],re['longitude'],re['altitude'],round(time.time()-start_time,2)])
    return record_list

def inference_mega_image_Retinanet_KNN(image_list, model_root, image_out_dir,text_out_dir, visualize, altitude_dict,device,scaleByAltitude, defaultAltitude=[],**kwargs):
    #retinanetknn
    from retinanet_inference_ver3 import Retinanet_instance
    model_type = 'Bird_drone_KNN'
    if (model_type=='Bird_drone_KNN'):
        load_w_config = True
    else:
        load_w_config = False
    record_list = []

    model_15 = os.path.join(model_root,'final_model_alt_15.pkl')
    model_30 = os.path.join(model_root,'final_model_alt_30.pkl')
    model_60 = os.path.join(model_root,'final_model_alt_60.pkl')
    model_90 = os.path.join(model_root,'final_model_alt_90.pkl')
    model_list = [model_15,model_30,model_60,model_90]
    net_list = []
    for model_dir in model_list:
        net_list.append(Retinanet_instance(transform,model_type,model_dir,device,load_w_config,int(defaultAltitude[0])))

    # model = Retinanet_instance(transform,model_type,model_dir,device,load_w_config,int(defaultAltitude[0]))
    with tqdm(total = len(image_list)) as pbar:
        for idxs, image_dir in (enumerate(image_list)):
            pbar.update(1)
            start_time = time.time()
            image_name = os.path.split(image_dir)[-1]
            if scaleByAltitude:
                image_taken_height,model_index,model_height = get_image_height_model_4(image_name.split('.')[0],altitude_dict)
                ratio = round(model_height/image_taken_height, 2)
                selected_model = net_list[model_index]

            mega_image,bbox_list = selected_model.inference(image_dir,0.2,scaleByAltitude)
            txt_name = os.path.basename(image_dir).split('.')[0]+'.txt'
            num_bird = 0
            with open(os.path.join(text_out_dir,txt_name), 'w') as f:
                if (len(bbox_list) != 0):
                    for box in bbox_list:
                        f.writelines('bird,{},{},{},{},{}\n'.format(
                            box[4], int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            mega_image = cv2.cvtColor(mega_image,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(image_out_dir,os.path.basename(image_dir)), mega_image)
            try:
                re = read_LatLotAlt(image_dir)
            except:
                re = {'latitude':0.0,
                      'longitude':0.0,
                      'altitude':0.0}
            record_list.append([os.path.basename(image_dir),kwargs['date_list'][idxs],kwargs['location_list'][idxs],
                               str(image_taken_height),re['latitude'],re['longitude'],re['altitude'],round(time.time()-start_time,2)])
    return record_list

def inference_mega_image_Retinanet(image_list, model_root, image_out_dir,text_out_dir, visualize,altitude_dict, scaleByAltitude,  defaultAltitude=[],**kwargs):
    #retinanet
    from retinanet import RetinaNet
    from encoder import DataEncoder

    conf_thresh = get_model_conf_threshold()
    model_30 = os.path.join(model_root,'final_model_alt_30.pkl')
    model_60 = os.path.join(model_root,'final_model_alt_60.pkl')
    model_90 = os.path.join(model_root,'final_model_alt_90.pkl')
    model_list = [model_30,model_60,model_90]

    net_list = []
    for height_model_dir in model_list:
        net_list.append(prepare_retina_net(height_model_dir,kwargs))

    record_list = []
    with tqdm(total = len(image_list)) as pbar:
        for idxs, image_dir in (enumerate(image_list)):
            pbar.update(1)
            start_time = time.time()
            altitude = int(defaultAltitude[idxs])
            image_name = os.path.split(image_dir)[-1]

            image_taken_height,model_index,model_height = 60,1,60
            ratio = 1.0
            net = net_list[1]
            if scaleByAltitude:
                image_taken_height,model_index,model_height = get_image_height_model_3(image_name.split('.')[0],altitude_dict)
                ratio = round(model_height/image_taken_height, 2)
                net = net_list[model_index]
            encoder = DataEncoder(device)
            bbox_list = []
            mega_image = cv2.imread(image_dir)
            mega_image = cv2.cvtColor(mega_image, cv2.COLOR_BGR2RGB)
            sub_image_list, coor_list = get_sub_image(mega_image, overlap=0.2, ratio=ratio)
            for index, sub_image in enumerate(sub_image_list):
                with torch.no_grad():
                    inputs = transform(cv2.resize(sub_image, (512, 512), interpolation=cv2.INTER_AREA))
                    inputs = inputs.unsqueeze(0).to(kwargs['device'])
                    loc_preds, cls_preds = net(inputs)
                    boxes, labels, scores = encoder.decode(
                        loc_preds.data.squeeze(), cls_preds.data.squeeze(), 512, CLS_THRESH = conf_thresh,NMS_THRESH = 0.5)
                if (len(boxes.shape) != 1):
                    for idx in range(boxes.shape[0]):
                        x1, y1, x2, y2 = list(
                            boxes[idx].cpu().numpy())  # (x1,y1, x2,y2)
                        score = scores.cpu().numpy()[idx]
                        bbox_list.append([coor_list[index][1]+ratio*x1, coor_list[index][0]+ratio*y1,
                                         coor_list[index][1]+ratio*x2, coor_list[index][0]+ratio*y2, score])
            txt_name = os.path.basename(image_dir).split('.')[0]+'.txt'
            with open(os.path.join(text_out_dir,txt_name), 'w') as f:
                if (len(bbox_list) != 0):
                    bbox_list = np.asarray([box for box in bbox_list])
                    box_idx = py_cpu_nms(bbox_list, 0.25)
                    selected_bbox = bbox_list[box_idx]
                    selected_bbox = sorted(selected_bbox,key = lambda x: x[4],reverse = True)
                    for box in selected_bbox:
                        f.writelines('bird,{},{},{},{},{}\n'.format(
                            box[4], int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                        if (visualize):
                            cv2.putText(mega_image, str(round(box[4], 2)), (int(box[0]), int(
                                box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            cv2.rectangle(mega_image, (int(box[0]), int(
                                box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            mega_image = cv2.cvtColor(mega_image, cv2.COLOR_RGB2BGR)
            if (visualize):
                cv2.imwrite(os.path.join(image_out_dir,os.path.basename(image_dir)), mega_image)
            try:
                re = read_LatLotAlt(image_dir)
            except:
                re = {'latitude':0.0,
                      'longitude':0.0,
                      'altitude':0.0}
            record_list.append([os.path.basename(image_dir),kwargs['date_list'][idxs],kwargs['location_list'][idxs],
                           str(image_taken_height),re['latitude'],re['longitude'],re['altitude'],round(time.time()-start_time,2)])
    return record_list

def inference_mega_image_YOLO(image_list, model_root, image_out_dir,text_out_dir, visualize , altitude_dict, scaleByAltitude=False,  defaultAltitude=[],**kwargs):
    #yolo
    from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                               increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

    record_list = []

    model_15_dir = os.path.join(model_root,'15.pt')
    model_30_dir = os.path.join(model_root,'30.pt')
    model_60_dir = os.path.join(model_root,'60.pt')
    model_90_dir = os.path.join(model_root,'90.pt')

    model_15 = prepare_yolo_net(model_15_dir)
    model_30 = prepare_yolo_net(model_30_dir)
    model_60 = prepare_yolo_net(model_60_dir)
    model_90 = prepare_yolo_net(model_90_dir)
    model_list = [model_15,model_30,model_60,model_90]
    
    mega_imgae_id = 0
    bbox_id = 1
    all_annotations= []

    with tqdm(total = len(image_list)) as pbar:
        for idxs,image_dir in (enumerate(image_list)):
            pbar.update(1)
            start_time = time.time()
            bbox_list = []
            mega_imgae_id += 1
            mega_image  = cv2.imread(image_dir)
            image_name = os.path.split(image_dir)[-1]
            image_taken_height,model_index,model_height = get_image_height_model_4(image_name.split('.')[0],altitude_dict)
            ratio = 1.0
            model = model_list[model_index]

            sub_image_list,coor_list = get_sub_image(mega_image,overlap = 0.2,ratio = ratio)
            for index,sub_image in enumerate(sub_image_list):
                sub_image = cv2.resize(sub_image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
                im = np.expand_dims(sub_image, axis=0)
                im = np.transpose(im,(0,3,1,2))
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                pred = model(im, augment=False, visualize=False)
                pred = non_max_suppression(pred, 0.05, 0.2, None, False, max_det=1000)

                for i, det in enumerate(pred):
                    if len(det):
                        for *xyxy, conf, cls in reversed(det):
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                            x1,y1,x2,y2 = xywh[0]-0.5*xywh[2], xywh[1]-0.5*xywh[3],xywh[0]+0.5*xywh[2], xywh[1]+0.5*xywh[3]  # (x1,y1, x2,y2)
                            if conf.cpu().numpy() > args.det_conf:
                                bbox_list.append([coor_list[index][1]+ratio*x1, coor_list[index][0]+ratio*y1, coor_list[index][1]+ratio*x2, coor_list[index][0]+ratio*y2,conf.cpu().numpy()])

            txt_name = os.path.basename(image_dir).split('.')[0]+'.txt'
            with open(os.path.join(text_out_dir,txt_name), 'w') as f:
                if (len(bbox_list) != 0):
                    bbox_list = np.asarray([box for box in bbox_list])
                    box_idx = py_cpu_nms(bbox_list, 0.25)
                    selected_bbox = bbox_list[box_idx]
                    selected_bbox = sorted(selected_bbox,key = lambda x: x[4],reverse = True)
                    for box in selected_bbox:
                        f.writelines('bird,{},{},{},{},{}\n'.format(
                            box[4], int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                        if (visualize):
                            cv2.putText(mega_image, str(round(box[4], 2)), (int(box[0]), int(
                                box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            cv2.rectangle(mega_image, (int(box[0]), int(
                                box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            if (visualize):
                cv2.imwrite(os.path.join(image_out_dir,os.path.basename(image_dir)), mega_image)
            try:
                re = read_LatLotAlt(image_dir)
            except:
                re = {'latitude':0.0,
                      'longitude':0.0,
                      'altitude':0.0}
            record_list.append([os.path.basename(image_dir),kwargs['date_list'][idxs],kwargs['location_list'][idxs],
                           str(image_taken_height),re['latitude'],re['longitude'],re['altitude'],round(time.time()-start_time,2)])
    return record_list

def inference_mega_image_faster(image_list, model_root, image_out_dir,text_out_dir, visualize , altitude_dict, scaleByAltitude=False, defaultAltitude=[],**kwargs):
  
    
    
  
    record_list = []
    model_15_dir = os.path.join(model_root,'Bird_GIJ_15m','model_final.pth')
    model_30_dir = os.path.join(model_root,'Bird_GIJ_30m','model_final.pth')
    model_60_dir = os.path.join(model_root,'Bird_GIJ_60m','model_final.pth')
    model_90_dir = os.path.join(model_root,'Bird_GIJ_90m','model_final.pth')

    model_15 = get_detectron_predictor(model_15_dir)
    model_30 = get_detectron_predictor(model_30_dir)
    model_60 = get_detectron_predictor(model_60_dir)
    model_90 = get_detectron_predictor(model_90_dir)
    model_list = [model_15,model_30,model_60,model_90]

    # predictor = get_detectron_predictor(model_dir = model_dir)

    mega_imgae_id = 0
    bbox_id = 1
    all_annotations= []

    with tqdm(total = len(image_list)) as pbar:

        for idxs,image_dir in (enumerate(image_list)):
            pbar.update(1)
            start_time = time.time()
            bbox_list = []
            mega_imgae_id += 1
            mega_image  = cv2.imread(image_dir)
            ratio =  1
            sub_image_list,coor_list = get_sub_image(mega_image,overlap = 0.2,ratio = 1)

            image_name = os.path.split(image_dir)[-1]
            image_taken_height,model_index,model_height = get_image_height_model_4(image_name.split('.')[0],altitude_dict)
            ratio = 1.0
            predictor = model_list[model_index]

            for index,sub_image in enumerate(sub_image_list):
                inputs = cv2.resize(sub_image,(512,512),interpolation = cv2.INTER_AREA)
                outputs = predictor(inputs)
                boxes = outputs["instances"].to("cpu").get_fields()['pred_boxes'].tensor.numpy()
                score = outputs["instances"].to("cpu").get_fields()['scores'].numpy()
                labels = outputs["instances"].to("cpu").get_fields()['pred_classes'].numpy()
                if (len(boxes.shape)!=0):
                    for idx in range(boxes.shape[0]):
                      x1,y1,x2,y2 = boxes[idx][0], boxes[idx][1] ,boxes[idx][2] ,boxes[idx][3]  # (x1,y1, x2,y2)
                      bbox_list.append([coor_list[index][1]+ratio*x1, coor_list[index][0]+ratio*y1, coor_list[index][1]+ratio*x2, coor_list[index][0]+ratio*y2,score[idx],labels[idx]])

            txt_name = os.path.basename(image_dir).split('.')[0]+'.txt'
            with open(os.path.join(text_out_dir,txt_name), 'w') as f:
                if (len(bbox_list) != 0):
                    bbox_list = np.asarray([box for box in bbox_list])
                    box_idx = py_cpu_nms(bbox_list, 0.25)
                    # num_bird = len(box_idx)
                    selected_bbox = bbox_list[box_idx]
                    # print('Finished on {},\tfound {} birds'.format(
                    # os.path.basename(image_dir), len(selected_bbox)))
                    selected_bbox = sorted(selected_bbox,key = lambda x: x[4],reverse = True)
                    for box in selected_bbox:
                        f.writelines('bird,{},{},{},{},{}\n'.format(
                            box[4], int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                        if (visualize):
                            cv2.putText(mega_image, str(round(box[4], 2)), (int(box[0]), int(
                                box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            cv2.rectangle(mega_image, (int(box[0]), int(
                                box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            if (visualize):
                cv2.imwrite(os.path.join(image_out_dir,os.path.basename(image_dir)), mega_image)
            try:
                re = read_LatLotAlt(image_dir)
            except:
                re = {'latitude':0.0,
                      'longitude':0.0,
                      'altitude':0.0}
            record_list.append([os.path.basename(image_dir),kwargs['date_list'][idxs],kwargs['location_list'][idxs],
                           str(image_taken_height),re['latitude'],re['longitude'],re['altitude'],round(time.time()-start_time,2)])
    return record_list

def prepare_classifier(model_name,num_of_classes):

    if model_name == 'resnet':
        model = ResNet.from_pretrained('resnet18')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_of_classes)
        savedweight = os.path.join('checkpoint','classifier','last-resnet18-realbirds.pt')
        checkpoint = torch.load(savedweight)
        model.load_state_dict(checkpoint)
        model.eval()
        model = model.to(device_name)
        return model


def simple_str(s):
    return "".join(filter(str.isalnum,s.lower()))


def crop_bird_from_image(image_data,bbox):
    height, width, channels = image_data.shape
    return cv2.resize(image_data[max(bbox[1],0):min(bbox[3],height),max(bbox[0],0):min(bbox[2],width)], (128,128), interpolation = cv2.INTER_AREA)

def build_dataloader(testdir):
         
    pretrained_size  = (128,128)
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds  = [0.229, 0.224, 0.225]

    test_transforms = transforms.Compose([
                transforms.Resize(pretrained_size),
                transforms.CenterCrop(pretrained_size),
                transforms.ToTensor(),
                transforms.Normalize(mean = pretrained_means, std = pretrained_stds)
            ])

    test_iterator = torch.utils.data.DataLoader(
         datasets.ImageFolder(testdir, test_transforms),
         batch_size=1)
    return test_iterator

ss=["American Widgeon_Female","American Widgeon_Male","Canada Goose","Canvasback_Male","Coot","Gadwall","Green-winged teal","Mallard Female",
"Mallard Male","Pelican","Pintail_Female","Pintail_Male","Ring-necked duck Female","Ring-necked duck Male","Scaup_Male","Shoveler_Female",
"Shoveler_Male","Snow/Ross Goose","White-fronted Goose"]

def predict_classes(images_root,bbox_dir,select_model):
    num_of_classes = 19
    txt_files = glob.glob(os.path.join(bbox_dir,'*.txt'))
    model = prepare_classifier(select_model,num_of_classes)
    print('There are {} images to be classied...'.format(str(len(txt_files))))
    for txt_dir in txt_files:
        txt_name = os.path.split(txt_dir)[-1].split('.')[0]
        image_dir = os.path.join(images_root,txt_name+'.{}'.format(args.image_ext))
        if not os.path.exists(os.path.join(bbox_dir,'tmp')):
            os.makedirs(os.path.join(bbox_dir,'tmp'))
            tmp_image_dir =os.path.join(bbox_dir,'tmp','unknown')
            os.makedirs(tmp_image_dir)
            with open(txt_dir, "r") as f:
                lines = f.readlines()
                if lines != []:
                    index = 1
                    image_data = cv2.imread(image_dir)
                    for line in lines:
                        confidence = float(line.split(',')[-5])
                        bbox = [int(line.split(',')[-4]),int(line.split(',')[-3]),int(line.split(',')[-2]),int(line.split(',')[-1])] #x1y1x2y2
                        bird_crop = crop_bird_from_image(image_data,bbox)
                        cv2.imwrite(os.path.join(tmp_image_dir,'{}.JPG'.format(index)),bird_crop)
                        index += 1
                    test_iterator = build_dataloader(os.path.join(bbox_dir,'tmp'))
                    y_preds=[]
                    with torch.no_grad():
                        for (x, y) in test_iterator:
                            # x = x.to('cuda')
                            x = x.to(device_name)
                            y_pred = model(x)
                            output = (torch.max(torch.exp(y_pred), 1)[1]).data.cpu().numpy()
                            y_preds.extend(output)
                    preds=[ss[i] for i in y_preds]
                    with open(txt_dir, "w") as f:
                        for j in range(len(lines)):
                            f.write(lines[j].replace('bird',preds[j]))
            shutil.rmtree(os.path.join(bbox_dir,'tmp'))

if __name__ == '__main__':
    args = get_args()
    image_list = sorted(glob.glob(os.path.join(args.image_root,'*.{}'.format(args.image_ext))))
    image_name_list = [os.path.basename(i) for i in image_list]

    altitude_list = [args.image_altitude for _ in image_list]
    
    location_list = [args.image_location for _ in image_list]
    date_list = [args.image_date for _ in image_list]
    
    target_dir = args.out_dir
    image_out_dir = os.path.join(target_dir,'visualize-results')
    text_out_dir = os.path.join(target_dir,'detection-results')
    csv_out_dir = os.path.join(target_dir,'detection_summary.csv')
    print ('*'*30)
    # print ('Using model type: {}'.format(model_type))
    print ('Using device: {}'.format(device))
    print ('Image out dir: {}'.format(image_out_dir))
    print ('Texting out dir: {}'.format(text_out_dir))
    print ('Inferencing on Images:\n {}'.format(image_list))
    print ('Visualize on each image:\n {}'.format(args.visualize))
    print ('*'*30)
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(text_out_dir, exist_ok=True)
    record = []

    altitude_dict = read_csv_info(args.csv_root)

    if(args.det_model == 'retinanet'):
        model_dir = os.path.join('checkpoint','retinanet','retinanet')
        if args.model_dir != '':
            model_dir = args.model_dir
        record = inference_mega_image_Retinanet(
        image_list=image_list, model_root = model_dir, image_out_dir = image_out_dir,text_out_dir = text_out_dir,csv_out_dir = csv_out_dir,  altitude_dict = altitude_dict,
        scaleByAltitude=args.use_altitude, defaultAltitude=altitude_list,date_list = date_list,location_list =location_list, 
        visualize = args.visualize,device = device)
    if(args.det_model == 'retinanetknn'):
        model_dir = os.path.join('checkpoint','retinanet','retinanetknn')
        if args.model_dir != '':
            model_dir = args.model_dir
        record = inference_mega_image_Retinanet_KNN(
        image_list=image_list, model_root = model_dir, image_out_dir = image_out_dir,text_out_dir = text_out_dir,csv_out_dir = csv_out_dir,  altitude_dict = altitude_dict,
        scaleByAltitude=args.use_altitude, defaultAltitude=altitude_list,date_list = date_list,location_list =location_list, 
        visualize = args.visualize,device = device)
    if(args.det_model == 'yolo'):
        model_dir = os.path.join('checkpoint','yolo','height_varient')
        if args.model_dir != '':
            model_dir = args.model_dir
        record = inference_mega_image_YOLO(
        image_list=image_list, model_root = model_dir, image_out_dir = image_out_dir,text_out_dir = text_out_dir,csv_out_dir = csv_out_dir, altitude_dict = altitude_dict,
        scaleByAltitude=args.use_altitude,  defaultAltitude=altitude_list,date_list = date_list,location_list =location_list, 
        visualize = args.visualize,device = device)
    if(args.det_model == 'faster'):
        model_dir = os.path.join('checkpoint','faster','Model_Bird_GIJ_altitude_Zhenduo')
        if args.model_dir != '':
            model_dir = args.model_dir
        record = inference_mega_image_faster(
        image_list=image_list, model_root = model_dir, image_out_dir = image_out_dir,text_out_dir = text_out_dir,csv_out_dir = csv_out_dir, altitude_dict = altitude_dict,
        scaleByAltitude=args.use_altitude, defaultAltitude=altitude_list,date_list = date_list,location_list =location_list, 
        visualize = args.visualize,device = device)
    if(args.det_model == 'yolonas'):
        model_dir = os.path.join('checkpoint','yolonas','height_varient')
        record = inference_mega_image_yolonas(
        image_list=image_list, model_root = model_dir, image_out_dir = image_out_dir,text_out_dir = text_out_dir,csv_out_dir = csv_out_dir, altitude_dict = altitude_dict,
        scaleByAltitude=args.use_altitude, defaultAltitude=altitude_list,date_list = date_list,location_list =location_list, 
        visualize = args.visualize,device = device)

    if (args.cla_model != ''):
        print('predicting classes...')
        if args.cla_model== 'mixmatch':
            model_dir = os.path.join('checkpoint','classifier','mixmatch_nojitter','model_best.pth.tar')
            mixmatch_classifier_inference(model_dir,image_list,text_out_dir,device) 

        elif args.cla_model=='res18':
            model_dir = os.path.join('checkpoint','classifier','Res18_Bird_I','model.pth')
            category_index_dir = model_dir.replace('model.pth','category_index.json')
            res18_classifier_inference(model_dir,category_index_dir,image_list,text_out_dir,device)
            
        print('finsh predicting classes!')

    if (args.evaluate):
        if_cate = True
        if args.cla_model == '':
            if_cate = False
        precision, recall, sum_AP, mrec, mprec, area = mAp_calculate(image_name_list = image_name_list, 
                                                                    gt_txt_list=[os.path.splitext(i)[0]+'.txt' for i in image_list],
                                                                    pred_txt_list = [os.path.join(text_out_dir,i.replace(args.image_ext,'txt')) for i in image_name_list],
                                                                    iou_thresh=0.3, 
                                                                    )

        best_conf_thresh = plot_f1_score(precision, recall, 'general', text_out_dir, area, 'f1_score', color='r')
        plt.legend()
        plt.savefig(os.path.join(target_dir,'f1_score.jpg'))
        plt.figure()
        plot_mAp(precision, recall, mprec, mrec,  'general', area, 'mAp', color='r')
        plt.legend()
        plt.savefig(os.path.join(target_dir,'mAp.jpg'))
        print('Evaluation completed, proceed to wrap result')

        conf_thresh_dict = {
            'yolo':0.3,
            'retinanet':0.5,
            'retinanetknn':0.2,
            'faster':0.5,
        }

        # conf_thresh = conf_thresh_dict[args.det_model]

        record,precision,recall,f1_score,cate_precision,cate_recall,cate_f1_score,count_error = compare_draw(record,text_out_dir,args.image_root,args.image_ext,best_conf_thresh,0.3,if_cate)
        log = open(os.path.join(target_dir,'overall_performance.txt'),'w')
        log.write('The overall performance on all images is')
        log.write('\nThe precision will be'+str(precision))
        log.write('\nThe recall will be '+str(recall))
        log.write('\nThe f1 score will be '+str(f1_score))
        log.write('\nThe cate_precision will be'+str(cate_precision))
        log.write('\nThe cate_recall will be '+str(cate_recall))
        log.write('\nThe cate_f1 score will be '+str(cate_f1_score))
        log.write('\nThe count_error will be '+str(count_error))

        record = pd.DataFrame(record)
        record.to_csv(csv_out_dir,header = ['image_name','date','location','altitude','latitude_meta','longitude_meta','altitude_meta','time_spent(sec)','pred_num_birds','gt_num_birds','tp','fp','fn','precision','recall','f1-score','count_error'],index = True)

        print('Complete image drawing!')

    argparse_dict = vars(args)

    with open(os.path.join(target_dir,'configs.json'),'w') as f:
        json.dump(argparse_dict,f,indent=4)
