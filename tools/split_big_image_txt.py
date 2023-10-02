import json
import os
import glob
import numpy as np 
import torchvision.transforms as transforms
import cv2

def IoU(true_box, pred_box):

    [xmin1, ymin1, xmax1, ymax1] = [int(true_box[0][0]),int(true_box[0][1]),int(true_box[1][0]),int(true_box[1][1])]
    [xmin2, ymin2, xmax2, ymax2] = [int(pred_box[0][0]),int(pred_box[0][1]),int(pred_box[1][0]),int(pred_box[1][1])]
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    xmin_inter = int(np.max([xmin1, xmin2]))
    xmax_inter = int(np.min([xmax1, xmax2]))
    ymin_inter = int(np.max([ymin1, ymin2]))
    ymax_inter = int(np.min([ymax1, ymax2]))
    if xmin_inter > xmax_inter or ymin_inter > ymax_inter:
        return 0
    area_inter = (xmax_inter - xmin_inter) * (ymax_inter - ymin_inter)
    # iou_num = float(area_inter) / (area1 + area2 - area_inter)
    # iou_tmp = float(area_inter) / area2
    # if iou_tmp < 0.2:
    #     return 0
    bbox_r = [xmin_inter-xmin1,ymin_inter-ymin1,xmax_inter-xmin1,ymax_inter-ymin1]
    return bbox_r

def check_and_make_dir(folder_dir):
    folder = os.path.exists(folder_dir)
    if not folder:
        os.makedirs(folder_dir)

def read_gps(image_name,info_dir):
    info_txt = open(info_dir,'r')
    for line in info_txt.readlines():
        img_name,_,coor_lat,coor_lont,altitude = line.split(' ')
        if (img_name in image_name):
            coor_lat = float(coor_lat.replace('(','').replace(',',''))
            coor_lont = float(coor_lont.replace('(','').replace(',',''))

            altitude = float(altitude.split(')')[0])
            return coor_lat, coor_lont, altitude

def read_gps_new(image_name):
  coor_lat = 39.761838833333336
  coor_lont = -93.23834030555555
  if image_name.split('m')[-1] == '.JPG':
    altitude = 208 + int(image_name.split('m')[0].split('_')[-1])
    print('with altitude')
  else:
    altitude = 208 + 90
  return coor_lat, coor_lont, altitude

def get_sub_image(mega_image,image_name,overlap=0.0,ratio=1):
    #mage_image: original image
    #ratio: ratio * 512 counter the different heights of image taken
    #return: list of sub image and list fo the upper left corner of sub image
    coor_list = []
    sub_image_list = []
    w,h,c = mega_image.shape
    size  = int(ratio*512)
    num_rows = int(w/int(size*(1-overlap)))
    num_cols = int(h/int(size*(1-overlap)))
    new_size = int(size*(1-overlap))
    for i in range(num_rows+1):
        if (i == num_rows):
            for j in range(num_cols+1):
                if (j==num_cols):
                    sub_image = mega_image[-size:,-size:,:]
                    coor_list.append([w-size,h-size])
                    sub_image_list.append (sub_image)
                else:
                    sub_image = mega_image[-size:,new_size*j:new_size*j+size,:]
                    coor_list.append([w-size,new_size*j])
                    sub_image_list.append (sub_image)
        else:
            for j in range(num_cols+1):
                if (j==num_cols):
                    sub_image = mega_image[new_size*i:new_size*i+size,-size:,:]
                    coor_list.append([new_size*i,h-size])
                    sub_image_list.append (sub_image)
                else:
                    sub_image = mega_image[new_size*i:new_size*i+size,new_size*j:new_size*j+size,:]
                    coor_list.append([new_size*i,new_size*j])
                    sub_image_list.append (sub_image)
    return sub_image_list,coor_list

def py_cpu_nms(dets, thresh):  
    """Pure Python NMS baseline.""" 
    dets = np.asarray(dets) 
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    scores = dets[:, 4]
  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  

    order = scores.argsort()[::-1]  
 
    keep = []  
    while order.size > 0:  
        i = order[0]  
        keep.append(i)  
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  
        inds = np.where(ovr <= thresh)[0]  
        order = order[inds + 1]  
  
    return keep

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

root_dir = '/home/yangzhang/waterfowl_datasets/drone_collection_dataset/test/Bird-I'
image_list = glob.glob(root_dir+'/*.JPG')
mega_imgae_dic = {}
mega_imgae_id = 1
bbox_id = 1
annotations= []

def line2bbox(line):
    parts = line.split(',')
    return [[int(parts[1]),int(parts[2])],[int(parts[3]),int(parts[4])]]

for image_dir in image_list:
    bbox_list = []
    mega_imgae_dic[mega_imgae_id] = image_dir.split('/')[-1]
    mega_imgae_id += 1
    mega_image  = cv2.imread(image_dir)
    coor_lat, coor_lont, altitude = read_gps_new(image_dir.split('/')[-1])
    absolute_altitude = 208 
    height_ratio = (altitude-absolute_altitude)
    ratio = 1
    print(image_dir) 
    sub_image_list,coor_list = get_sub_image(mega_image,image_dir.split('/')[-1],overlap = 0.1,ratio = ratio)
    sub_image_dir = root_dir+'/small_only/'
    check_and_make_dir(sub_image_dir)
    check_and_make_dir(sub_image_dir+'/images/')
    check_and_make_dir(sub_image_dir+'/labels/')

    txt_dir = image_dir.replace(image_dir.split('.')[-1],'txt')
    if os.path.exists(txt_dir):
        with open(txt_dir, 'r') as f:
            lines = f.readlines()
        for i in range(len(sub_image_list)):
            bbox_str = ''
            coor_y = int(coor_list[i][0])
            coor_x = int(coor_list[i][-1])
            image_p = [[coor_x,coor_y],[coor_x+512*ratio,coor_y+512*ratio]]
            for line in lines:
                bbox_item = line2bbox(line)
                bbox = IoU(image_p,bbox_item)
                if bbox != 0:
                    bbox_str += '0 {} {} {} {}\n'.format((bbox[0]+bbox[2])/1024,(bbox[1]+bbox[3])/1024,(bbox[2]-bbox[0])/512,(bbox[3]-bbox[1])/512)
            if True:
                cv2.imwrite (sub_image_dir+'/images/'+image_dir.split('/')[-1].split('.')[0]+'_'+str(coor_list[i][0])+'_'+str(coor_list[i][1])+'.JPG',cv2.resize(sub_image_list[i],(512,512),interpolation = cv2.INTER_AREA))
                with open(sub_image_dir+'/labels/'+image_dir.split('/')[-1].split('.')[0]+'_'+str(coor_list[i][0])+'_'+str(coor_list[i][1])+'.txt','w') as f:
                    f.write(bbox_str)
