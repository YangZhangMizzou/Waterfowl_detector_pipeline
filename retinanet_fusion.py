import torch
import torch.nn as nn

from fpn import FPN101, FPN50
from torch.autograd import Variable


class RetinaNet(nn.Module):
    
    def __init__(self, num_classes=20,num_anchors=5):
        super(RetinaNet, self).__init__()
        self.num_anchors = num_anchors
        self.fpn = FPN50()
        self.num_classes = num_classes
        self.p4_layer = self._fusion_head(128)
        self.p3_layer = self._fusion_head(128)
        self.loc_head = self._make_head(self.num_anchors*4)
        self.cls_head = self._make_head(self.num_anchors*self.num_classes)
        self.up_sample_layer = self.up_sample()
        self.down_sample_layer = self.down_sample()

    def forward(self, x):
        fms = self.fpn(x)
        fms = fms[:2]
        #fms = [fms]#one layer
        p4 = self.p4_layer(fms[1])
        #p4 = nn.functional.interpolate(p4,scale_factor=2, mode='bicubic', align_corners=True)
        p3 = self.p3_layer(fms[0])
        p3 = self.down_sample_layer(p3)
        #print (p3.shape,p4.shape)
        fusion = torch.cat([p3,p4],1)
        loc_pred = self.loc_head(fusion).permute(0,2,3,1).contiguous().view(x.size(0),-1,4)                 # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
        cls_pred = self.cls_head(fusion).permute(0,2,3,1).contiguous().view(x.size(0),-1,self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
        return loc_pred,cls_pred
    def up_sample(self):
        up = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1)
        return up
    def down_sample(self):
        layer = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        return layer
    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)
    def _fusion_head(self,out_planes):
        layers = []
        for _ in range(2):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)
    
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

def test():
    img = cv2.imread(path_img, 1)
    net = RetinaNet()
    loc_preds, cls_preds = net(Variable(torch.randn(2,3,224,224)))
    print(loc_preds.size())
    print(cls_preds.size())
import numpy as np
from PIL import Image
def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


if __name__ == '__main__':
    import cv2
    import torchvision.transforms as transforms
    net  = RetinaNet(2,3)
    model_dir = '/home/robert/Models/Retinanet_inference_example/checkpoint/drone_collection_KNN_15m/'
    print (net._modules.keys(),net.fpn._modules)
    #net.load_state_dict(torch.load(model_dir+'final_model_dict.pkl'))
    net = torch.load(model_dir+'final_model.pkl')
    torch.save(net.state_dict(), model_dir+'extra_dict.pkl')
    print (net.state_dict().keys())