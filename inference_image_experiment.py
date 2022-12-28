# from black import E
import cv2
import torch
import torchvision.transforms as transforms
from retinanet import RetinaNet
from encoder import DataEncoder
import os
import warnings
import glob
import numpy as np
import time
import json
import sys
import pandas as pd
from args import *
from tools import py_cpu_nms,get_sub_image
import matplotlib.pyplot as plt
from tqdm import tqdm
from mAP_cal import mAp_calculate,plot_f1_score,plot_mAp
import shutil
from get_f1 import compare
import compare


from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode, time_sync

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from efficientnet_pytorch import EfficientNet
from resnet_pytorch import ResNet
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms
import collections
from absl import app, flags


# from utils_retina import read_LatLotAlt,get_GSD
# from WaterFowlTools.mAp import mAp_calculate,plot_f1_score,plot_mAp
# from WaterFowlTools.utils import py_cpu_nms, get_image_taking_conditions, get_sub_image

warnings.filterwarnings("ignore")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_conf_threshold = {'Bird_A':0.2,'Bird_B':0.2,'Bird_C':0.2,'Bird_D':0.2,'Bird_E':0.2,'Bird_drone':0.2}
model_extension = {'Bird_drone':{40:('_alt_30',30),90:('_alt_90',90)}}


def get_model_conf_threshold ():
    return 0.2

def get_model_extension(altitude_dict,image_dir,model_root_dir,defaultaltitude):
    altitude = altitude_dict[image_dir.split('/')[-1]]
    model_altitude = [30,60,90]
    model_gap_altitue = [abs(30-altitude),abs(60-altitude),abs(90-altitude)]
    model_height_select = model_altitude[model_gap_altitue.index(min(model_gap_altitue))]
    return model_root_dir+'/final_model_alt_{}.pkl'.format(str(model_height_select)),model_height_select,altitude

def get_detectron_predictor(model_dir):
    cfg = get_cfg()
    cfg.merge_from_file('./configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml')
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32],[64],[128]]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    cfg.MODEL.WEIGHTS = model_dir
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)
    return predictor

def read_csv_info(csv_dir):
    df = pd.read_csv(csv_dir)
    altitude_dict = {}
    image_list = []
    for line in df.values.tolist():
        altitude_dict[line[1]] = int(line[3])
        if line[4] == 'test' and abs(int(line[3])-15)<6:
            image_list.append(args.image_root+'/{}'.format(line[1]))
    return altitude_dict,image_list 

def inference_mega_image_Retinanet(image_list, model_root_dir, image_out_dir,text_out_dir, visualize,altitude_dict, scaleByAltitude=True, defaultAltitude=[],**kwargs):
    for idxs, image_dir in (enumerate(image_list)):
        conf_thresh = get_model_conf_threshold()
        model_dir,ref_altitude,image_altitude = get_model_extension(altitude_dict,image_dir,model_root_dir=model_root_dir,defaultaltitude=defaultAltitude[0])
        if (kwargs['device']!=torch.device('cuda')):
            print ('loading CPU mode')
            device = torch.device('cpu')
            net = torch.load(model_dir,map_location=device)
            net = net.module.to(device)
        else:
            device = torch.device('cuda')
            net = torch.load(model_dir)
        net.to(kwargs['device'])
        # print('check net mode',next(net.parameters()).device)
        encoder = DataEncoder(device)
        record = []
        start_time = time.time()
        altitude = int(defaultAltitude[idxs])
        if scaleByAltitude:
            # GSD,ref_GSD = get_GSD(altitude,camera_type='Pro2', ref_altitude=ref_altitude) # Mavic2 Pro GSD equations
            # ratio = 1.0*ref_GSD/GSD
            ratio = ref_altitude/image_altitude
        else:
            ratio = 1.0
        print ('Using {}m model for {}m image, Processing scale {}'.format(str(ref_altitude),str(image_altitude),str(ratio)))
        bbox_list = []
        mega_image = cv2.imread(image_dir)
        mega_image = cv2.cvtColor(mega_image, cv2.COLOR_BGR2RGB)
        sub_image_list, coor_list = get_sub_image(
            mega_image, overlap=0.2, ratio=ratio)
        for index, sub_image in enumerate(sub_image_list):
            with torch.no_grad():
                inputs = transform(cv2.resize(
                    sub_image, (512, 512), interpolation=cv2.INTER_AREA))
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
                num_bird = len(box_idx)
                selected_bbox = bbox_list[box_idx]
                print('Finished on {},\tfound {} birds'.format(
                os.path.basename(image_dir), len(selected_bbox)))
                selected_bbox = sorted(selected_bbox,key = lambda x: x[4],reverse = True)
                for box in selected_bbox:
                    f.writelines('bird {} {} {} {} {}\n'.format(
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
        record.append([os.path.basename(image_dir),kwargs['date_list'][idxs],kwargs['location_list'][idxs],
                       defaultAltitude[idxs],re['latitude'],re['longitude'],re['altitude'],
                       num_bird,round(time.time()-start_time,2)])
    record = pd.DataFrame(record)
    record.to_csv(kwargs['csv_out_dir'],header = ['image_name','date','location','altitude','latitude_meta','longitude_meta','altitude_meta','num_birds','time_spent(sec)'],index = True)
     
def inference_mega_image_YOLO(image_list, model_dir, image_out_dir,text_out_dir, visualize , scaleByAltitude=False, defaultAltitude=[],**kwargs):
    record = []
    device = select_device('')
    model = DetectMultiBackend(model_dir, device=device, dnn=False, data='./configs/BirdA_all.yaml', fp16=False)
    stride, names, pt = model.stride, model.names, model.pt

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
            ratio = 1.0
            sub_image_list,coor_list = get_sub_image(mega_image,overlap = 0.1,ratio = ratio)
            for index,sub_image in enumerate(sub_image_list):
                im = np.expand_dims(sub_image, axis=0)
                im = np.transpose(im,(0,3,1,2))
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                pred = model(im, augment=False, visualize=False)
                pred = non_max_suppression(pred, 0.05, 0.2, None, False, max_det=1000)

                # im0 = im
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                for i, det in enumerate(pred):
                    if len(det):
                        # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                            x1,y1,x2,y2 = xywh[0]-0.5*xywh[2], xywh[1]-0.5*xywh[3],xywh[0]+0.5*xywh[2], xywh[1]+0.5*xywh[3]  # (x1,y1, x2,y2)
                            if conf.cpu().numpy() > 0.3:
                                bbox_list.append([coor_list[index][1]+ratio*x1, coor_list[index][0]+ratio*y1, coor_list[index][1]+ratio*x2, coor_list[index][0]+ratio*y2,conf.cpu().numpy()])

            txt_name = os.path.basename(image_dir).split('.')[0]+'.txt'
            with open(os.path.join(text_out_dir,txt_name), 'w') as f:
                if (len(bbox_list) != 0):
                    bbox_list = np.asarray([box for box in bbox_list])
                    box_idx = py_cpu_nms(bbox_list, 0.25)
                    num_bird = len(box_idx)
                    selected_bbox = bbox_list[box_idx]
                    print('Finished on {},\tfound {} birds'.format(
                    os.path.basename(image_dir), len(selected_bbox)))
                    selected_bbox = sorted(selected_bbox,key = lambda x: x[4],reverse = True)
                    for box in selected_bbox:
                        f.writelines('bird {} {} {} {} {}\n'.format(
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
            record.append([os.path.basename(image_dir),kwargs['date_list'][idxs],kwargs['location_list'][idxs],
                           defaultAltitude[idxs],re['latitude'],re['longitude'],re['altitude'],
                           num_bird,round(time.time()-start_time,2)])
    record = pd.DataFrame(record)
    record.to_csv(kwargs['csv_out_dir'],header = ['image_name','date','location','altitude','latitude_meta','longitude_meta','altitude_meta','num_birds','time_spent(sec)'],index = True)

def inference_mega_image_faster(image_list, model_dir, image_out_dir,text_out_dir, visualize , scaleByAltitude=False, defaultAltitude=[],**kwargs):
    
    record = []
    predictor = get_detectron_predictor(model_dir = model_dir)

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
            sub_image_list,coor_list = get_sub_image(mega_image,overlap = 0.1,ratio = 1)

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
                    num_bird = len(box_idx)
                    selected_bbox = bbox_list[box_idx]
                    print('Finished on {},\tfound {} birds'.format(
                    os.path.basename(image_dir), len(selected_bbox)))
                    selected_bbox = sorted(selected_bbox,key = lambda x: x[4],reverse = True)
                    for box in selected_bbox:
                        f.writelines('bird {} {} {} {} {}\n'.format(
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
            record.append([os.path.basename(image_dir),kwargs['date_list'][idxs],kwargs['location_list'][idxs],
                           defaultAltitude[idxs],re['latitude'],re['longitude'],re['altitude'],
                           num_bird,round(time.time()-start_time,2)])
    record = pd.DataFrame(record)
    record.to_csv(kwargs['csv_out_dir'],header = ['image_name','date','location','altitude','latitude_meta','longitude_meta','altitude_meta','num_birds','time_spent(sec)'],index = True)

def prepare_classifier(model_name,num_of_classes):

    if model_name == 'resnet':
        model = ResNet.from_pretrained('resnet18')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_of_classes)
        savedweight = './checkpoint/classifier/resnet18-sklearn-sf-vr-last.pt'
    if  model_name == 'resnext':
        model = models.resnext50_32x4d(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_of_classes)
        savedweight = './checkpoint/classifier/resnext50-sklearn-last.pt'
    if  model_name == 'efficientnet':
        model = ResNet.from_pretrained('efficientnet-b3')
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_of_classes)

    checkpoint = torch.load(savedweight)
    model.load_state_dict(checkpoint)
    model.eval()
    model = model.to('cuda')
    return model

def crop_bird_from_image(image_data,bbox):
    height, width, channels = image_data.shape
    return image_data[max(bbox[1],0):min(bbox[3],height),max(bbox[0],0):min(bbox[2],width)]

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

ss=['Ring-necked_duck_Male', 'American_Widgeon_Female', 'Ring-necked_duck_Female', 'Canvasback_Male', 'Canvasback_Female', 
'Scaup_Male', 'Shoveler_Male', 'Not_a_bird', 'Shoveler_Female', 'Gadwall', 'Unknown', 'Mallard_Male', 'Pintail_Male', 
'Green-winged_teal', 'White-fronted_Goose', 'Snow/Ross_Goose_(blue)', 'Snow/Ross_Goose', 'Mallard_Female', 'Coot', 'Pelican', 
'American_Widgeon_Male', 'Canada_Goose']

def predict_classes(images_root,bbox_dir,select_model):
    num_of_classes = 22
    txt_files = glob.glob(bbox_dir+'/*.txt')
    model = prepare_classifier(select_model,num_of_classes)
    for txt_dir in txt_files:
        txt_name = txt_dir.split('/')[-1].split('.')[0]
        image_dir = images_root +'/'+ txt_name + '.{}'.format(args.image_ext)
        if not os.path.exists(bbox_dir+'/tmp'):
            os.makedirs(bbox_dir+'/tmp')
            tmp_image_dir = bbox_dir+'/tmp/unknown'
            os.makedirs(tmp_image_dir)
            with open(txt_dir, "r") as f:
                lines = f.readlines()
                if lines != []:
                    index = 1
                    image_data = cv2.imread(image_dir)
                    for line in lines:
                        confidence = float(line.split(' ')[-5])
                        bbox = [int(line.split(' ')[-4]),int(line.split(' ')[-3]),int(line.split(' ')[-2]),int(line.split(' ')[-1])] #x1y1x2y2
                        bird_crop = crop_bird_from_image(image_data,bbox)
                        cv2.imwrite(tmp_image_dir+'/{}.JPG'.format(index),bird_crop)
                        index += 1
                    test_iterator = build_dataloader(bbox_dir+'/tmp')
                    y_preds=[]
                    with torch.no_grad():
                        for (x, y) in tqdm(test_iterator):
                            x = x.to('cuda')
                            y_pred = model(x)
                            output = (torch.max(torch.exp(y_pred), 1)[1]).data.cpu().numpy()
                            y_preds.extend(output)
                    preds=[ss[i] for i in y_preds]
                    with open(txt_dir, "w") as f:
                        for j in range(len(lines)):
                            f.write(lines[j].replace('bird',preds[j]))
            shutil.rmtree(bbox_dir+'/tmp/')

if __name__ == '__main__':
    args = get_args()
    info_csv = args.image_root+'/image_info.csv'
    altitude_dict,image_list = read_csv_info(info_csv)
    # image_list = glob.glob(os.path.join(args.image_root,'*.{}'.format(args.image_ext)))
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
    # print ('Inferencing on Images:\n {}'.format(image_list))
    print ('Visualize on each image:\n {}'.format(args.visualize))
    print ('*'*30)
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(text_out_dir, exist_ok=True)

    if(args.det_model == 'retinanet'):
        model_dir = './checkpoint/retinanet/general'
        if args.model_dir != '':
            model_dir = args.model_dir
        inference_mega_image_Retinanet(
        image_list=image_list, model_root_dir = model_dir, image_out_dir = image_out_dir,text_out_dir = text_out_dir,csv_out_dir = csv_out_dir,
        scaleByAltitude=args.use_altitude, defaultAltitude=altitude_list ,altitude_dict = altitude_dict, date_list = date_list,location_list =location_list,
        visualize = args.visualize,device = device)
    if(args.det_model == 'yolo'):
        model_dir = './checkpoint/yolo/general/weights/best.pt'
        if args.model_dir != '':
            model_dir = args.model_dir
        inference_mega_image_YOLO(
        image_list=image_list, model_dir = model_dir, image_out_dir = image_out_dir,text_out_dir = text_out_dir,csv_out_dir = csv_out_dir,
        scaleByAltitude=args.use_altitude, defaultAltitude=altitude_list,date_list = date_list,location_list =location_list,
        visualize = args.visualize,device = device)
    if(args.det_model == 'faster'):
        model_dir = './checkpoint/faster/Fasterrcnn-Bird/model_final.pth'
        if args.model_dir != '':
            model_dir = args.model_dir
        inference_mega_image_faster(
        image_list=image_list, model_dir = model_dir, image_out_dir = image_out_dir,text_out_dir = text_out_dir,csv_out_dir = csv_out_dir,
        scaleByAltitude=args.use_altitude, defaultAltitude=altitude_list,date_list = date_list,location_list =location_list,
        visualize = args.visualize,device = device)

    if (args.cla_model != ''):
        print('predicting classes...')
        predict_classes(args.image_root,text_out_dir,args.cla_model)
    if (args.evaluate):
        precision, recall, sum_AP, mrec, mprec, area = mAp_calculate(image_name_list = image_name_list, 
                                                                    gt_txt_list=[os.path.splitext(i)[0]+'.txt' for i in image_list],
                                                                    pred_txt_list = [text_out_dir+'/'+os.path.splitext(i)[0]+'.txt' for i in image_name_list],
                                                                    iou_thresh=0.3, 
                                                                    )
        plot_f1_score(precision, recall, 'general', text_out_dir, area, 'f1_score', color='r')
        plt.legend()
        plt.savefig(os.path.join(target_dir,'f1_score.jpg'))
        plt.figure()
        plot_mAp(precision, recall, mprec, mrec,  'general', area, 'mAp', color='r')
        plt.legend()
        plt.savefig(os.path.join(target_dir,'mAp.jpg'))
        print('Evaluation completed, proceed to wrap result')



    argparse_dict = vars(args)
    with open(os.path.join(target_dir,'configs.json'),'w') as f:
        json.dump(argparse_dict,f,indent=4)