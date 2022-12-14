import json
import os
import cv2
import collections
from tqdm import tqdm
from absl import app, flags
 
from efficientnet_pytorch import EfficientNet
from resnet_pytorch import ResNet
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms


class Classifier:   
     def __init__(self, detectionJson, dataset, croppedsave, score, batch, outputfile,classifier):
        self.detectionJson=detectionJson
        self.dataset=dataset
        self.croppedsave=croppedsave
        self.score=score
        self.batch=batch
        self.outputfile=outputfile
        self.classfier=classifier
      
     
        
     def input_crop(self):
         with open (self.detectionJson, 'r') as w:
              data=json.load(w)
         
         imglist=[]
         predictedbbox_list=[]
         idx=0
         
         #check if directory exists:
         if os.path.exists(self.croppedsave+'/1')==False:
             os.mkdir(self.croppedsave+'/1')
         newdir=self.croppedsave+'/1'
         
         
         for i in range(len(data)):
             imglist.append(data[i]["image_name"])
         counter=dict(collections.Counter(imglist))
         
         confidscore=[]
         
         for jk in range(len(list(counter.keys()))):
             img=cv2.imread(self.dataset+'/'+list(counter.keys())[jk])
             
             for mm in range(idx, idx+counter[list(counter.keys())[jk]]):
                 
                 xmin=data[mm]["bbox"][0]
                 ymin=data[mm]["bbox"][1]
                 width=data[mm]["bbox"][2]
                 height=data[mm]["bbox"][3]
                 
                 if data[mm]["score"]>self.score:
                    predictedbbox_list.append([xmin,ymin,xmin+width,ymin+height])
                    croppedimage=img[ymin:ymin+height, xmin:xmin+width]
                    cv2.imwrite(newdir+'/'+list(counter.keys())[jk].split('.')[0]+'_'+str(mm)+'.'+list(counter.keys())[jk].split('.')[1], croppedimage)
                    confidscore.append(data[mm]["score"])
                    
             idx=idx+counter[list(counter.keys())[jk]]
                    
         return confidscore
        
        
     def test_evalaute(self, savedweight):
         
         number_class=22
         
         pretrained_size  = (128,128)
         pretrained_means = [0.485, 0.456, 0.406]
         pretrained_stds  = [0.229, 0.224, 0.225]
             
         
         ss=['Ring-necked duck Male', 'American Widgeon_Female', 'Ring-necked duck Female', 'Canvasback_Male', 
             'Canvasback_Female', 'Scaup_Male', 'Shoveler_Male', 'Not a bird', 'Shoveler_Female', 'Gadwall', 'Unknown', 'Mallard Male', 'Pintail_Male', 'Green-winged teal', 
             'White-fronted Goose', 'Snow/Ross Goose (blue)', 'Snow/Ross Goose', 'Mallard Female', 'Coot', 'Pelican', 'American Widgeon_Male', 
             'Canada Goose']
         
         
         test_transforms = transforms.Compose([
         				transforms.Resize(pretrained_size),
         				transforms.CenterCrop(pretrained_size),
         				transforms.ToTensor(),
         				transforms.Normalize(mean = pretrained_means, std = pretrained_stds)
         			])
         testdir=self.croppedsave
         
         test_iterator = torch.utils.data.DataLoader(
                 datasets.ImageFolder(testdir, test_transforms),
                 batch_size=self.batch)
         
         model_list=['resnet18', 'resnet50', 'efficientnet-b3', 'efficientnet-b5']
         
         if self.classfier == '1'  or self.classfier == '2':
             model = ResNet.from_pretrained(model_list[self.classfier-1])
             num_ftrs = model.fc.in_features
             model.fc = nn.Linear(num_ftrs, number_class)
         elif self.classfier == '4'  or self.classfier == '5':
             model = ResNet.from_pretrained(model_list[self.classfier-2])
             num_ftrs = model._fc.in_features
             model._fc = nn.Linear(num_ftrs, number_class)
         else:
             model = models.resnext50_32x4d(pretrained=True)
             num_ftrs = model.fc.in_features
             model.fc = nn.Linear(num_ftrs, number_class)
             
         checkpoint = torch.load(savedweight)
         
         model.load_state_dict(checkpoint)
         model.eval()
         model = model.to('cuda')
         
         
         y_preds=[]
         
         with torch.no_grad():
             for (x, y) in tqdm(test_iterator):
                 x = x.to('cuda')
                 y_pred = model(x)
                 output = (torch.max(torch.exp(y_pred), 1)[1]).data.cpu().numpy()
                 y_preds.extend(output)
                 
         preds=[ss[i] for i in y_preds]
         
         prdidx=[int(ks.split('.')[0].split('_')[len(ks.split('.')[0].split('_'))-1]) for ks in os.listdir(self.croppedsave+'/1')]
         
         prdidx.sort()
         
         return preds, prdidx
         
         
     def save_output(self, pred, predindex, Bigimgdir, confiscore):
         
         if os.path.exists(Bigimgdir)==False:
             os.mkdir(Bigimgdir)
         
         with open (self.detectionJson, 'r') as w:
              data=json.load(w)
              
         with open (self.outputfile, 'w') as f:

            for i in range(len(predindex)):
              ssz=str(data[predindex[i]]["image_name"])+","+str(data[predindex[i]]["bbox"])+","+pred[i]+","+str(i)
              f.write(ssz)
              f.write('\n')
        
         singleimg=[data[rr]["image_name"] for rr in predindex]
         
         counter=dict(collections.Counter(singleimg))
         
         with open (self.outputfile, 'r') as f2:
              Lines=f2.readlines()
              
         Bigimgidx=1
         
         idx1=0
         
         #Create a 
         
         singimg=[]
         
         dirc=[]
         
         
         print(Lines[796])
         
         for imgidx in range(len(confiscore)):
             
             temp=[]
             
             species=Lines[imgidx].strip().split(',')[5]
             
             xmin=int(Lines[imgidx].strip().split(',')[1][1:])
             #print(xmin)
             ymin=int(Lines[imgidx].strip().split(',')[2])
             #print(ymin)
             xmax=int(int(Lines[imgidx].strip().split(',')[3])+xmin)
             #print(xmax)
             #print(Lines[imgidx].strip().split(',')[4][1:-1])
             ymax=int(int(Lines[imgidx].strip().split(',')[4][1:-1])+ymin)
             
             score=confiscore[imgidx]
             
             tmp=[species, score, xmin, ymin, xmax, ymax]
             
             singimg.append(tmp)
             
             
             if imgidx+1<=len(confiscore)-1:
                if Lines[imgidx+1].strip().split(',')[0] != Lines[imgidx].strip().split(',')[0]:
                    #print(Lines[imgidx+1].strip().split(',')[0])
                    Bigimgidx+=1
                    dirc.append(singimg)
                    singimg=[]
        
         dirc.append(singimg)
        
             
         for jk in range(Bigimgidx):
             
             with open (Bigimgdir+'/'+str(jk)+'.txt', 'w') as f1:
                # print(len(dirc[jk]))
                 for miao in range(len(dirc[jk])):
                 #    print(str(dirc[jk][miao][0]))
                     f1.write(str(dirc[jk][miao][0])+' '+str(dirc[jk][miao][1])+' '+str(dirc[jk][miao][2])+' '+str(dirc[jk][miao][3])+' '+str(dirc[jk][miao][4])+' '+str(dirc[jk][miao][5]))
                     f1.write('\n')
              
            
              
flags.DEFINE_string('detection','/home/shiqi/Data/ShiqiWang2021/Processed/Bird_J_shiqi/bird.json', 'The input json file for zhenduo detection result')

flags.DEFINE_string('dataset', '/home/shiqi/Data/ShiqiWang2021/Processed/Bird_J', 'The big image dataset')

flags.DEFINE_string('croppedsave', '/home/shiqi/Data/ShiqiWang2021/Processed/Robert_train_test/Zhenduo', 'cropped file your want to save')

flags.DEFINE_float('score', 0.9, 'confidence score for detection result')

flags.DEFINE_integer('batch', 1, 'the batch size for the evaluation')

flags.DEFINE_string('outputfile', '/home/shiqi/Data/ShiqiWang2021/Processed/Robert_train_test/output>90.txt', 'result file for zhendup')

flags.DEFINE_string('singlebigimg', '/home/shiqi/Data/ShiqiWang2021/Processed/Robert_train_test/Detectedscore09', 'Each Big Images')

#flags.DEFINE_string('textfile', '/home/shiqi/Data/ShiqiWang2021/Processed/Bird_B', 'The text file diretory')

flags.DEFINE_string('classifier', 1, 'What classifier do u want')
        
            
#weight="/home/shiqi/Data/ShiqiWang2021/Processed/pruned_image+labels/resnet18-sklearn-sf-vr-last.pt" 

weight=""

FLAGS1=flags.FLAGS 

if FLAGS1.classifier == 1:
   weight="./checkpoint/classifier/resnet18-sklearn-sf-vr-last.pt" 
elif  FLAGS1.classifier == 2:
   weight="./checkpoint/classifier/resnet50-sklearn-last.pt" 
elif  FLAGS1.classifier == 3:
   weight="./checkpoint/classifier/resnext50-sklearn-last.pt"  
elif  FLAGS1.classifier == 4:
   weight="./checkpoint/classifier/efficientnet-b3-sklearn-last.pt" 
elif  FLAGS1.classifier == 5:
   weight="./checkpoint/classifier/efficientnet-b5-sklearn-last.pt" 


def main(argv):     
         
    asj=Classifier(FLAGS1.detection, FLAGS1.dataset, FLAGS1.croppedsave, FLAGS1.score, FLAGS1.batch, FLAGS1.outputfile, FLAGS1.classifier)
    
    bigimg=asj.input_crop()

    pred, predindex=asj.test_evalaute(weight)

    asj.save_output(pred, predindex, FLAGS1.singlebigimg, bigimg)

app.run(main)