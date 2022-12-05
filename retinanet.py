import torch
import torch.nn as nn

from fpn import FPN101, FPN50
from torch.autograd import Variable


class RetinaNet(nn.Module):
    num_anchors = 9
    
    def __init__(self, num_classes=20):
        super(RetinaNet, self).__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors*4)
        self.cls_head = self._make_head(self.num_anchors*self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        #fms = [fms]#one layer
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)                 # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds,1), torch.cat(cls_preds,1)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


import numpy as np
from PIL import Image
def img_transform(img_in, transform):
    """B*C*H*W
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
    img_path = '/home/zt253/data/UnionData/mdc/mar2019_clipped_MODOC1214_0015GSD_LINE03B0501.png'
    img = cv2.imread(img_path,1)
    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img_input = img_transform(img, transform)
    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())


    def farward_hook(module, input, output):
        fmap_block.append(output)
    fmap_block = list()
    grad_block = list()
    net  = RetinaNet(2)
    print (net._modules.keys(),net.fpn._modules)
    net.fpn.toplayer2.register_forward_hook(farward_hook)
    net.fpn.toplayer2.register_backward_hook(backward_hook)

    output = net(img_input)
    