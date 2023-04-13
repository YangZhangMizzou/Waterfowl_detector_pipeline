import glob
import numpy as np
import matplotlib.pyplot as plt
import os


def IoU2(pred_bbox, true_box):
    bb = pred_bbox[1:]
    bbgt = true_box
    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]),
          min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1
    if iw > 0 and ih > 0:
        # compute overlap (IoU) = area of intersection / area of union
        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
        ov = iw * ih / ua
        return ov
    return 0.0


def IoU(pred_box, true_box):
    [ymin1, xmin1, ymax1, xmax1] = true_box
    [s2, ymin2, xmin2, ymax2, xmax2] = pred_box
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    xmin_inter = max(xmin1, xmin2)
    xmax_inter = min(xmax1, xmax2)
    ymin_inter = max(ymin1, ymin2)
    ymax_inter = min(ymax1, ymax2)
    if xmin_inter > xmax_inter or ymin_inter > ymax_inter:
        return 0
    area_inter = 1.0*(xmax_inter - xmin_inter) * (ymax_inter - ymin_inter)
    return float(area_inter) / (area1 + area2 - area_inter)


def nms(pred_bbox, thres=0.0):
    for id1 in range(len(pred_bbox)):
        y1, x1, y1_2, x1_2, score1 = pred_bbox[id1]
        y1 = int(y1)
        x1 = int(x1)
        for id2 in range(len(pred_bbox)):
            if id1 == id2:
                continue
            y2, x2, y2_2, x2_2, score2 = pred_bbox[id2]
            y2 = int(y2)
            x2 = int(x2)
            distance = np.sqrt(np.power((x2-x1), 2)+np.power((y2-y1), 2))
            if distance <= 2*thres*np.sqrt(2):
                if score1 >= score2:
                    pred_bbox[id2][4] = -1
    out_bbox = []
    for bbox in pred_bbox:
        if (not bbox[4] == -1):
            out_bbox.append(bbox)
    return out_bbox


def match_GT(pred_bbox, gt_bbox, image_name=None, iou_thresh=0.3, confidence_thresh=0.3):
    iou_val = []
    for gt_item in gt_bbox:
        iou_val.append(IoU2(pred_bbox, gt_item))
    if sum(np.array(iou_val) > iou_thresh) == 0:
        fn = 1
        return 0
    else:
        tp = 1
        taken = iou_val.index(max(iou_val))
        gt_bbox.remove(gt_bbox[taken])
        return 1


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def mAp_calculate(image_name_list, gt_txt_list, pred_txt_list, iou_thresh):
    tp = 0
    fp = 0
    gt_dict = {}
    total_gt_box = 0
    for idx, gt_file in enumerate(gt_txt_list):
        gt_name = os.path.split(image_name_list[idx])[-1].split('.')[0]
        gt_bbox = []
        try:
            with open(gt_file, 'r') as f:
                gt_data = f.readlines()
        except:
            gt_data = []
        for line in gt_data:
            gt_bbox.append([float(i)
                           for i in line.replace('\n', '').split(',')[-4:]])
        gt_dict[gt_name] = gt_bbox
        total_gt_box += len(gt_bbox)
    pred_dict = {}
    total_pred_box = 0
    for idx, pred_file in enumerate(pred_txt_list):
        pred_name = os.path.split(image_name_list[idx])[-1].split('.')[0]
        with open(pred_file, 'r') as f:
            pred_data = f.readlines()
        pred_bbox = []
        for line in pred_data:
             pred_bbox.append([float(i)
                         for i in line.replace('\n', '').split(',')[1:]])
        pred_bbox.sort(reverse=True)
        pred_dict[pred_name] = pred_bbox
        total_pred_box += len(pred_bbox)
    ct = 0
    precision = []
    recall = []
    while True:
        pred_head = ([j[0] if j != [] else [-1, 0, 0, 0, 0]
                     for j in pred_dict.values()])
        selected_head = max(pred_head)
        if (selected_head == [-1, 0, 0, 0, 0]):
            break
        selected_index = pred_head.index(selected_head)
        image_name = list(pred_dict.keys())[selected_index]
        pred_dict[image_name].remove(selected_head)
        gt_bbox = gt_dict[image_name]
        re = match_GT(selected_head, gt_bbox,
                      image_name=image_name, iou_thresh=iou_thresh)
        if (re == 1):
            tp += 1
        else:
            fp += 1
        ct += 1
        precision.append(1.0*tp/(tp+fp))
        recall.append(1.0*tp/total_gt_box)
    area = 0
    sum_AP = 0
    ap, mrec, mprec = voc_ap(recall[:], precision[:])
    sum_AP += ap
    for i in range(1, len(precision)):
        area += (precision[i]+precision[i])/2*(recall[i]-recall[i-1])
    return precision, recall, sum_AP, mrec, mprec, area


def plot_mAp(precision, recall, mprec, mrec, dataset_name, area, label, color='r'):
    area = round(area*100, 2)
    plt.plot(recall, precision, label=label +
             ' mAp:{}'.format(area), color=color)
    plt.fill_between(mrec[:-1], mprec[:-1], color=color, alpha=0.2)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('{} mAp'.format(dataset_name))
    plt.xlabel('Recall')
    plt.ylabel('Precision')


def plot_f1_score(precision, recall, dataset_name, pred_dir, area, label, color):
    f1_list = []
    for i in range(len(precision)):
        f1 = 2*precision[i]*recall[i]/(precision[i]+recall[i]+0.00001)
        f1_list.append(f1)
    pred_bbox = []
    for pred_file in glob.glob(os.path.join(pred_dir,'*.txt')):
        with open(pred_file, 'r') as f:
            pred_data = f.readlines()
        for line in pred_data:
            pred_bbox.append([float(i)
                         for i in line.replace('\n', '').split(',')[1:]])
    pred_bbox.sort(reverse=True)
    score = [i[0] for i in pred_bbox]
    conf_thresh = pred_bbox[f1_list.index(max(f1_list))][0]
    plt.plot(score, f1_list, label=label+' f1:{} at conf thresh:{}'.format(
        round(max(f1_list), 2), round(conf_thresh, 2)), color=color)
    plt.title('{} F1 score'.format(dataset_name))
    plt.xlim(1.0, 0)
    plt.xlabel('confidence threshold')
    plt.ylabel('F1_score')
    return conf_thresh
