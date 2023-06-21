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
from tools import py_cpu_nms,get_sub_image
import matplotlib.pyplot as plt
from tqdm import tqdm
from mAP_cal import mAp_calculate,plot_f1_score,plot_mAp
import shutil
from compare_and_draw import compare_draw

#re18
from classification_infernece_res18 import res18_classifier_inference
from classifiers.MixMatch.mixmatch_classification import mixmatch_classifier_inference
from resnet_pytorch import ResNet
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets

#retinanet
from retinanet import RetinaNet
from encoder import DataEncoder

#retinanetknn
from retinanet_inference_ver3 import Retinanet_instance

#yolo
from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

#faster
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

#yolonas
import super_gradients


warnings.filterwarnings("ignore")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)


def get_model_conf_threshold ():
    return args.det_conf


    

def get_detectron_predictor(model_dir):
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join('configs','COCO-Detection','faster_rcnn_R_50_FPN_1x.yaml'))
    cfg.MODEL.DEVICE = device_name
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32],[64],[128]]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    cfg.MODEL.WEIGHTS = model_dir
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)
    return predictor

def choose_model_and_infer(image_dir,model_list):
    optimal_bbox_list = []
    best_count = 0
    for predictor in model_list:
        bbox_list,count = faster_inference(image_dir,predictor)
        if count > best_count:
            best_count = count
            optimal_bbox_list = bbox_list
    return optimal_bbox_list

def faster_inference(image_dir,predictor):
    bbox_list = []
    mega_image  = cv2.imread(image_dir)
    sub_image_list,coor_list = get_sub_image(mega_image,overlap = 0.2,ratio = 1)
    image_name = os.path.split(image_dir)[-1]
    ratio = 1.0

    for index,sub_image in enumerate(sub_image_list):
        inputs = sub_image
        outputs = predictor(inputs)
        boxes = outputs["instances"].to("cpu").get_fields()['pred_boxes'].tensor.numpy()
        score = outputs["instances"].to("cpu").get_fields()['scores'].numpy()
        if (len(boxes.shape)!=0):
            for idx in range(boxes.shape[0]):
              x1,y1,x2,y2 = boxes[idx][0], boxes[idx][1] ,boxes[idx][2] ,boxes[idx][3]
              bbox_list.append([coor_list[index][1]+ratio*x1, coor_list[index][0]+ratio*y1, coor_list[index][1]+ratio*x2, coor_list[index][0]+ratio*y2,score[idx]]) 
    if (len(bbox_list) != 0):
        bbox_list = np.asarray([box for box in bbox_list])
        box_idx = py_cpu_nms(bbox_list, 0.25)
        selected_bbox = bbox_list[box_idx]
        selected_bbox = sorted(selected_bbox,key = lambda x: x[4],reverse = True)
    else:
        return [],0
    return selected_bbox,np.mean(score)

def inference_mega_image_faster(image_list, model_root, image_out_dir,text_out_dir, visualize , scaleByAltitude=False, defaultAltitude=[],**kwargs):
    
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

    with tqdm(total = len(image_list)) as pbar:
        for idxs,image_dir in (enumerate(image_list)):
            pbar.update(1)
            start_time = time.time()
            selected_bbox = choose_model_and_infer(image_dir,model_list)
            txt_name = os.path.basename(image_dir).split('.')[0]+'.txt'
            with open(os.path.join(text_out_dir,txt_name), 'w') as f:
                for box in selected_bbox:
                    f.writelines('bird,{},{},{},{},{}\n'.format(box[4], int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                    if (visualize):
                        cv2.putText(mega_image, str(round(box[4], 2)), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.rectangle(mega_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            if (visualize):
                cv2.imwrite(os.path.join(image_out_dir,os.path.basename(image_dir)), mega_image)
            try:
                re = read_LatLotAlt(image_dir)
            except:
                re = {'latitude':0.0,
                      'longitude':0.0,
                      'altitude':0.0}
            record_list.append([os.path.basename(image_dir),kwargs['date_list'][idxs],kwargs['location_list'][idxs],
                           defaultAltitude[idxs],re['latitude'],re['longitude'],re['altitude'],round(time.time()-start_time,2)])
    return record_list

def prepare_yolonas(model_dir):
    if device_name == 'cuda':
        return super_gradients.training.models.get('yolo_nas_m',num_classes=1,checkpoint_path=model_dir).cuda()
    else:
        return super_gradients.training.models.get('yolo_nas_m',num_classes=1,checkpoint_path=model_dir)

def inference_mega_image_yolonas(image_list,model_root, image_out_dir,text_out_dir, visualize, altitude_dict,device,scaleByAltitude, defaultAltitude=[],**kwargs):

    record_list = []
    model_15 = os.path.join(model_root,'ckpt_best.pth')
    selected_model = prepare_yolonas(model_15)

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
                ratio = 1
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
            selected_bbox = []
            if (len(bbox_list) != 0):
                bbox_list = np.asarray([box for box in bbox_list])
                box_idx = py_cpu_nms(bbox_list, 0.25)
                selected_bbox = bbox_list[box_idx]
                selected_bbox = sorted(selected_bbox,key = lambda x: x[4],reverse = True)

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
                               defaultAltitude[idxs],re['latitude'],re['longitude'],re['altitude'],round(time.time()-start_time,2)])
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


ss=["American Widgeon_Female","American Widgeon_Male","Canada Goose","Canvasback_Male","Coot","Gadwall","Green-winged teal","Mallard Female",
"Mallard Male","Pelican","Pintail_Female","Pintail_Male","Ring-necked duck Female","Ring-necked duck Male","Scaup_Male","Shoveler_Female",
"Shoveler_Male","Snow/Ross Goose","White-fronted Goose"]


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
    print ('Inferencing on {} Images:\n'.format(len(image_list)))
    print ('Visualize on each image:\n'.format(args.visualize))
    print ('*'*30)
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(text_out_dir, exist_ok=True)
    record = []

    if(args.det_model == 'faster'):
        model_dir = os.path.join('checkpoint','faster','Model_Bird_GIJ_altitude_Zhenduo')
        if args.model_dir != '':
            model_dir = args.model_dir
        record = inference_mega_image_faster(
        image_list=image_list, model_root = model_dir, image_out_dir = image_out_dir,text_out_dir = text_out_dir,csv_out_dir = csv_out_dir,
        scaleByAltitude=args.use_altitude, defaultAltitude=altitude_list,date_list = date_list,location_list =location_list, 
        visualize = args.visualize,device = device)

    if(args.det_model == 'yolonas'):
        model_dir = os.path.join('checkpoint','yolonas','deepforest')
        record = inference_mega_image_yolonas(
        image_list=image_list, model_root = model_dir, image_out_dir = image_out_dir,text_out_dir = text_out_dir,csv_out_dir = csv_out_dir, altitude_dict = {},
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
            'faster':0.9,
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