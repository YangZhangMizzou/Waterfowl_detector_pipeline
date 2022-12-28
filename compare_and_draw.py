import json
import numpy as np
import pandas as pd
import glob
import cv2

def IoU(true_box, pred_box):

	[xmin1, ymin1, xmax1, ymax1] = [true_box[0],true_box[1],true_box[2],true_box[3]]
	[xmin2, ymin2, xmax2, ymax2] = [pred_box[0],pred_box[1],pred_box[2],pred_box[3]]
	area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
	area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
	xmin_inter = max(xmin1, xmin2)
	xmax_inter = min(xmax1, xmax2)
	ymin_inter = max(ymin1, ymin2)
	ymax_inter = min(ymax1, ymax2)
	if xmin_inter > xmax_inter or ymin_inter > ymax_inter:
	    return 0
	area_inter = (xmax_inter - xmin_inter) * (ymax_inter - ymin_inter)
	return float(area_inter) / (area1 + area2 - area_inter)

def draw_image(image_dir,output_dir,tp_list,fp_list,fn_list,tp_cate_list,cate = True):
	raw_image = cv2.imread(image_dir)
	for box in fn_list:
		cv2.ellipse(raw_image, [int((box[0]+box[2])/2),int((box[1]+box[3])/2)], (int(box[2]-box[0]),int(box[3]-box[1])),0, 0, 360, (0,0,255), 3)
		
	for box in fp_list:
		cv2.polylines(raw_image, np.array([[(int((box[0]+box[2])/2), box[1]), (box[0], box[3]), (box[2], box[3])]]), True, (0,0,255), 3)
	for box in tp_list:
		cv2.rectangle(raw_image, (box[0],box[1]), (box[2],box[3]), (0,0,255), 3)
		cv2.putText(raw_image, str(box[-1]), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
	if cate:
		for box in tp_cate_list:
			cv2.rectangle(raw_image, (box[0],box[1]), (box[2],box[3]), (255,0,0), 5)
			cv2.putText(raw_image, str(box[-1]), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
	save_dir = output_dir+'/'+image_dir.split('/')[-1].replace('.','_eval.')
	cv2.imwrite(save_dir,raw_image)

def calculate_precis_recall(true_bbox,pred_bbox,iou):
    fn = 0
    fp = 0
    tp = 0
    tp_cate = 0
    tp_list = []
    fp_list = []
    fn_list = []
    tp_cate_list = []
    # tp_class_list = []
    # tp_cate_class_list = []

    total_pred = len(pred_bbox)
    nneg = lambda x :max(0,x)
    if (len(true_bbox)*len(pred_bbox)==0):
        fn = len(true_bbox)
        fp = len(pred_bbox)
        tp = 0
    else:
        for t_bbox in true_bbox:
            iou_val = []
            positive = [] 
            for p_bbox in pred_bbox:
                iou_val.append(IoU(t_bbox,p_bbox))
                if IoU(t_bbox,p_bbox) > iou:
                	positive = p_bbox

            if sum(np.array(iou_val)>iou)==0:
                fn += 1
                fn_list.append(t_bbox)
            else :
                tp+=1
                taken = iou_val.index(max(iou_val))
                tp_list.append(pred_bbox[taken])
                if pred_bbox[taken][-1] == t_bbox[-1]:
                	tp_cate +=1
                	tp_cate_list.append(pred_bbox[taken])
                pred_bbox.remove(pred_bbox[taken])
    fp = total_pred-tp
    fp_list = pred_bbox
    return tp,fp,fn,tp_cate,tp_list,fp_list,fn_list,tp_cate_list

def read_box_from_gt_txt(txt_dir):
	bbox_list = []
	with open(txt_dir, "r") as f:
		lines = f.readlines()
		if lines != []:
			for line in lines:
				part = line.split(',')
				bbox_list.append([int(part[-4]),int(part[-3]),int(part[-2]),int(part[-1]),part[0]])
	return bbox_list

def read_box_from_pred_txt(txt_dir,thresh = 0.0):
	bbox_list = []
	with open(txt_dir, "r") as f:
		lines = f.readlines()
		if lines != []:
			for line in lines:
				part = line.split(',')
				if float(part[1]) < thresh:
					continue
				bbox_list.append([int(part[-4]),int(part[-3]),int(part[-2]),int(part[-1]),part[0]])
	return bbox_list

def compare_draw(record,prediction_dir,ground_truth_dir,image_type = 'JPG',threshhold = 0.5,iou = 0.3):
	predict_txt_list = sorted(glob.glob(prediction_dir+'/*.txt'))

	false_pred = []
	true_pred = []
	false_neg =[]
	precision_per_image =[]
	recall_per_image = []
	count_error_per_image = []
	image_name = []
	f1_score_per_image = []
	tp_cates = []

	for index in range(len(predict_txt_list)):

		gt_txt = ground_truth_dir +'/'+ predict_txt_list[index].split('/')[-1]
		gt_list = read_box_from_gt_txt(gt_txt)
		pred_list = read_box_from_pred_txt(predict_txt_list[index],threshhold)
		image_dir = gt_txt.replace('.txt','.{}'.format(image_type))
		tp,fp,fn,tp_cate,tp_list,fp_list,fn_list,tp_cate_list = calculate_precis_recall(gt_list,pred_list,iou)
		draw_image(image_dir,prediction_dir.replace('detection-results','visualize-results'),tp_list,fp_list,fn_list,tp_cate_list)

		false_pred.append(fp)
		true_pred.append(tp)
		false_neg.append(fn)
		tp_cates.append(tp_cate)
		
		precision_this_image = 0
		recall_this_image = 0
		f1_score_this_image = 0
		count_error_this_image = 0
		if tp != 0:	
			precision_this_image = round((1.0*tp)/(1.0*tp+1.0*fp),2)
			recall_this_image = round((1.0*tp)/(1.0*tp+1.0*fn),2)
			f1_score_this_image = round(2*precision_this_image*recall_this_image/(precision_this_image+recall_this_image),2)
		if tp+fn != 0:
			count_error_this_image = round(abs((1.0*fp-1.0*fn))/(1.0*tp+1.0*fn),2)
		record[index].extend([precision_this_image,recall_this_image,f1_score_this_image,len(gt_list),count_error_this_image])


	precision = (1.0*np.sum(true_pred))/(1.0*np.sum(true_pred)+1.0*np.sum(false_pred)) 
	recall = (1.0*np.sum(true_pred)/(1.0*(np.sum(true_pred)+np.sum(false_neg))))
	f1_score = 2*precision*recall/(precision+recall)
	count_error = abs(1.0*np.sum(false_pred)-1.0*np.sum(false_neg))/(1.0*np.sum(true_pred)+1.0*np.sum(false_neg)) 
	cate_precision = (1.0*np.sum(tp_cates))/(1.0*np.sum(true_pred)+1.0*np.sum(false_pred)) 
	cate_recall = (1.0*np.sum(tp_cates)/(1.0*(np.sum(true_pred)+np.sum(false_neg))))
	cate_f1_score = 2*cate_precision*cate_recall/(cate_precision+cate_recall)

	return record,round(precision,2),round(recall,2),round(f1_score,2),round(cate_precision,2),round(cate_recall,2),round(cate_f1_score,2),round(count_error,2)

