import json
import numpy as np
import pandas as pd


def IoU(true_box, pred_box):

	[xmin1, ymin1, xmax1, ymax1] = true_box
	[xmin2, ymin2, xmax2, ymax2] = pred_box
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

def calculate_precis_recall(true_bbox,pred_bbox,iou):
    fn = 0
    fp = 0
    tp = 0
    tp_cate = 0
    fp_cate = 0
    fn_cate = 0

    total_pred = len(pred_bbox)
    nneg = lambda x :max(0,x)
    # print(len(true_bbox))
    if (len(true_bbox)*len(pred_bbox)==0):
        fn = len(true_bbox)
        fp = len(pred_bbox)
        tp = 0
        print(fp)
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
            else :
                tp+=1
                if t_bbox[-1] == positive[-1]:
                	if t_bbox[-1] == 1:
                		tp_cate +=1
                else:
                	fp_cate += 1
                taken = iou_val.index(max(iou_val))
                pred_bbox.remove(pred_bbox[taken])
    fp = total_pred-tp
    return tp,fp,fn,tp_cate,fp_cate,fn_cate

def get_preds(txt_dir):
	bbox_list = []
	with open(txt_dir, "r") as f:
		lines = f.readlines()
	for line in lines:
		bbox_list.append(line.split(' ')[-5:])

def get_gts(txt_dir):
	bbox_list = []
	with open(txt_dir, "r") as f:
		lines = f.readlines()
	for line in lines:
		bbox_list.append(line.split(' ')[-4:])


def compare(preds_dir,gts_dir,results_dir,is_classification,threshhold = 0.3,iou = 0.3):

	preds = get_preds(preds_dir)
	gts = get_gts(gts_dir)
	total_predictions = len(preds)
	total_gts = len(gts)

	tp,fp,fn,tp_cate,fp_cate,fn_cate = calculate_precis_recall(preds,gts,iou)

	precision = (1.0*tp)/(1.0*total_predictions) 
	recall = (1.0*tp)/(1.0*total_gts)
	f1_score = 2*precision*recall/(precision+recall)

	count_error = 0
	if total_gts != 0:
		count_error = abs(total_predictions-total_gts)/total_gts

	columns = ['Class','Precision','Recall','F1-SCORE','Counting_error']
	dataframe.columns = pd.MultiIndex.from_tuples(columns)
	dataframe = pd.DataFrame({'Class':'Waterfowl','Precision':[precision],'Recall':[recall],'F1-SCORE':[f1_score],'count_error':[count_error]})
	dataframe.to_csv(results_dir+'/{}.csv',index=False,sep=',',columns = columns)

	# print ('The missing pred will be: '+str(empty_pred))
	# print ('The precision will be'+str(precision))
	# print ('The recall will be '+str(recall))
	# print ('The f1 score will be '+str(2*precision*recall/(precision+recall)))
	# print ('The cate_precision will be'+str(cate_precision))
	# print ('The cate_recall will be '+str(cate_recall))
	# print ('The cate_f1 score will be '+str(cate_f1_score))
	# print ('The cate_error will be '+str(np.sum(fp_cate_list)))
	# print(np.sum(true_pred),np.sum(false_pred))

	# log.write('\nThe precision will be'+str(precision))
	# log.write('\nThe recall will be '+str(recall))
	# log.write('\nThe f1 score will be '+str(2*precision*recall/(precision+recall)))
	# log.write('\nThe cate_precision will be'+str(cate_precision))
	# log.write('\nThe cate_recall will be '+str(cate_recall))
	# log.write('\nThe cate_f1 score will be '+str(cate_f1_score))
	# log.write('\nThe cate_error will be '+str(np.sum(fp_cate_list)))
	# log.close()
