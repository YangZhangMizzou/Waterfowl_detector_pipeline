from encoder import DataEncoder,DataEncoder_fusion
import torch
import json
from WaterFowlTools.utils import py_cpu_nms, get_image_taking_conditions, get_sub_image
import cv2
from utils import read_LatLotAlt,get_GSD,filter_slice

model_conf_threshold = {'Bird_A':0.2,
                        'Bird_B':0.2,
                        'Bird_C':0.2,
                        'Bird_D':0.2,
                        'Bird_E':0.2,
                        'Bird_drone':0.2}
model_extension = {
        'Bird_drone':{40:('_alt_30',30),
                        75:('_alt_60',60),
                        90:('_alt_90',90)},
        'Bird_drone_KNN':{20:('_alt_15',15),
                        40:('_alt_30',30),
                        75:('_alt_60',60),
                        90:('_alt_90',90)}
                    }

def get_model_conf_threshold (model_type):
    if (model_type in model_conf_threshold):
        return model_conf_threshold[model_type]
    else:
        return 0.3
def get_model_extension(model_type,model_dir,altitude):
    if(model_type in model_extension):
        model_ext = model_extension[model_type]
        for altitude_thresh in model_ext:
            if (altitude_thresh>=altitude):
                ref_altitude = model_ext[altitude_thresh][1]
                model_dir = model_dir.replace('.pkl',model_ext[altitude_thresh][0]+'.pkl')
                return model_dir,ref_altitude
        model_dir = model_dir.replace('.pkl',model_ext[max(model_ext.keys())][0]+'.pkl')
        return model_dir,model_ext[max(model_ext.keys())][1]
    else:
        return model_dir,altitude

class Retinanet_instance():
    def __init__(self,input_transform,model_type,model_dir,device =torch.device('cuda'),load_w_config = True,altitude=15):
        self.transform = input_transform
        self.model_type = model_type
        self.load_w_config = load_w_config
        self.altitude = altitude
        self.model_dir,self.ref_altitude = get_model_extension(model_type,model_dir,altitude)
        self.device = device
        self.conf_threshold = get_model_conf_threshold(model_type)
        self.model = None
        self.encoder = None
        self.load_model()
    
    def load_model(self):
        print(self.model_dir)
        if (self.load_w_config):
            config_dir = self.model_dir.replace('.pkl','.json')
            with open(config_dir,'r') as f:
                cfg = json.load(f)
            print (cfg['KNN_anchors'])
            from retinanet_fusion import RetinaNet
            self.model = RetinaNet(num_classes=1,num_anchors=len(cfg['KNN_anchors']))
            self.encoder = DataEncoder_fusion(anchor_wh=cfg['KNN_anchors'],device = self.device)
            #self.model.load_state_dict(torch.load(self.model_dir))
        else:
            from retinanet import RetinaNet
            self.model = RetinaNet(num_classes=1)
            self.encoder = DataEncoder(self.device)
        self.model = torch.load(self.model_dir,map_location=self.device)
        self.model = self.model.module.to(self.device)
        self.model.eval()
        print('check net mode',next(self.model.parameters()).device)

    def inference(self,image_dir,slice_overlap,read_GPS = False,debug = True):
        mega_image = cv2.imread(image_dir)
        mega_image = cv2.cvtColor(mega_image, cv2.COLOR_BGR2RGB)
        if (read_GPS):
            try:
                altitude = read_LatLotAlt(image_dir)['altitude']
                print ('Reading altitude from Meta data of {}'.format(altitude))
            except:
                altitude = self.altitude
                print ('Meta data not available, use default altitude {}'.format(altitude))
        else:
            altitude = self.altitude
            print ('Using default altitude {}'.format(altitude))
        GSD,ref_GSD = get_GSD(altitude,camera_type='Pro2', ref_altitude=self.ref_altitude)
        ratio = 1.0*ref_GSD/GSD
        print('Image processing altitude: {} \t Processing scale {}'.format(altitude,ratio))
        sub_image_list, coor_list = get_sub_image(
            mega_image, overlap=slice_overlap, ratio=ratio)
        
        bbox_list = []
        for index, sub_image in enumerate(sub_image_list):
            sub_bbox_list = []
            with torch.no_grad():
                inputs = self.transform(cv2.resize(
                    sub_image, (512, 512), interpolation=cv2.INTER_AREA))
                inputs = inputs.unsqueeze(0).to(self.device)
                loc_preds, cls_preds = self.model(inputs)
                boxes, labels, scores = self.encoder.decode(
                    loc_preds.data.squeeze(), cls_preds.data.squeeze(), 512, CLS_THRESH = self.conf_threshold,NMS_THRESH = 0.25)
            if (len(boxes.shape) != 1):
                for idx in range(boxes.shape[0]):
                    x1, y1, x2, y2 = list(
                        boxes[idx].cpu().numpy())  # (x1,y1, x2,y2)
                    score = scores.cpu().numpy()[idx]
                    sub_bbox_list.append([x1,y1,x2,y2,score])
                #filter boxes that has overlapped region on sliced images

                sub_bbox_list = filter_slice(sub_bbox_list,coor_list[index],sub_image.shape[0],mega_image.shape[:2],dis = int(slice_overlap/2*512))
                
                for sub_box in sub_bbox_list:
                    x1,y1,x2,y2,score = sub_box
                    bbox_list.append([coor_list[index][1]+ratio*x1, coor_list[index][0]+ratio*y1,
                                     coor_list[index][1]+ratio*x2, coor_list[index][0]+ratio*y2, score])

        box_idx = py_cpu_nms(bbox_list, 0.25)
        bbox_list = [bbox_list[i] for i in box_idx]
        if (debug):
            w = sub_image_list[0].shape[0]
            for i,coor in enumerate(coor_list):
                cv2.rectangle(mega_image,(coor[1],coor[0]),(coor[1]+w,coor[0]+w),(i, 255-i, 0), 2)


        for box in bbox_list:
            cv2.putText(mega_image, str(round(box[4], 2)), (int(box[0]), int(
                            box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(mega_image, (int(box[0]), int(
                            box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        return mega_image,bbox_list

if __name__=='__main__':
    import torchvision.transforms as transforms
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
    model = Retinanet_instance(input_transform = transform,model_type = 'Bird_drone_KNN',
                            model_dir = '/home/robert/Models/Retinanet_inference_example/checkpoint/Bird_drone_KNN/final_model.pkl',
                            device =torch.device('cpu'),load_w_config = True,altitude=15)
    image_dir = '/home/robert/Data/drone_collection/Cloud_HarvestedCrop_15m_DJI_0251.jpg'
    re = model.inference(image_dir=image_dir,slice_overlap= 0.2)
    import matplotlib.pyplot as plt
    plt.imshow(re[0])
    plt.show()
