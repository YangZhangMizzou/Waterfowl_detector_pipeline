import numpy as np
from PIL import Image
import glob
from tqdm import tqdm

import torchvision
import torch

category_dict = {
    "0": "American Widgeon_Female",
    "1": "American Widgeon_Male",
    "2": "Canada Goose",
    "3": "Canvasback_Male",
    "4": "Coot",
    "5": "Gadwall",
    "6": "Green-winged teal",
    "7": "Mallard Female",
    "8": "Mallard Male",
    "9": "Not a bird",
    "10": "Pelican",
    "11": "Pintail_Female",
    "12": "Pintail_Male",
    "13": "Ring-necked duck Female",
    "14": "Ring-necked duck Male",
    "15": "Scaup_Male",
    "16": "Shoveler_Female",
    "17": "Shoveler_Male",
    "18": "Snow",
    "19": "Unknown",
    "20": "White-fronted Goose"
}
category_dict = {v:k for k,v in category_dict.items()}

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_waterfowl_dataset(root,img_size,transform_train=None, transform_val=None):

    train_dir = '{}/{}'.format(root,'train')
    test_dir = '{}/{}'.format(root,'test')
    unlabel_dir = '{}/{}'.format(root,'unlabel')


    train_image_list,train_label_list,val_image_list,val_label_list,test_image_list,test_label_list,unlabel_image_list = train_val_split(train_dir,unlabel_dir,test_dir)
    train_labeled_dataset = CIFAR10_labeled(root,train_image_list, train_label_list, img_size, train=True, transform=transform_train,download=False)
    train_unlabeled_dataset = CIFAR10_unlabeled(root,unlabel_image_list, img_size, train=True, transform=TransformTwice(transform_train),download=False)
    val_dataset = CIFAR10_labeled(root,val_image_list, val_label_list, img_size, train=False, transform=transform_val,download=False)
    test_dataset = CIFAR10_labeled(root,test_image_list, test_label_list, img_size, train=False, transform=transform_val,download=False)

    print (f"#Labeled: {len(train_label_list)} #Unlabeled: {len(unlabel_image_list)} #Val: {len(val_label_list)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset
    

def train_val_split(train_dir,unlabel_dir,test_dir):

    #train+val
    image_list = glob.glob(train_dir+'/*/*_0_decoy.JPG')
    np.random.shuffle(image_list)
    # train_image_list = image_list[0:int(0.8*len(image_list))]
    train_image_list = image_list
    train_label_list = [image_dir.split('/')[-2] for image_dir in train_image_list]

    val_image_list = glob.glob(test_dir+'/*/*_0_decoy.JPG')
    np.random.shuffle(val_image_list)
    val_label_list = [image_dir.split('/')[-2] for image_dir in val_image_list]

    #unlabel
    unlabel_image_list = glob.glob(unlabel_dir+'/*.JPG')

    #test
    test_image_list = glob.glob(test_dir+'/*/*.JPG')
    np.random.shuffle(test_image_list)
    test_label_list = [image_dir.split('/')[-2] for image_dir in test_image_list]

    return train_image_list,train_label_list,val_image_list,val_label_list,test_image_list,test_label_list,unlabel_image_list

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalize(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x

class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()

class GaussianNoise(object):
    """Add gaussian noise to the image.
    """
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x

class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x

class CIFAR10_labeled():

    def __init__(self,root, image_list, label_list, img_size, train=True,transform=None, target_transform=None,download=False):
        image_data = []
        label_trans = []
        self.img_size = img_size
        with tqdm(total=len(image_list)) as pbar:
            for i in range(len(image_list)):
                sub_image = Image.open(image_list[i])
                sub_image = sub_image.resize((self.img_size,self.img_size))
                image_data.append(np.asarray(sub_image))
                label_trans.append(int(category_dict[label_list[i]]))
                pbar.update(1)

        self.transform = transform
        self.target_transform = target_transform
        self.data = np.array(image_data)
        self.targets = np.array(label_trans)
        self.data = transpose(normalize(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)


class CIFAR10_unlabeled():

    def __init__(self,root, image_list, img_size, train=True,transform=None, target_transform=None,download=False):
        image_data = []
        self.img_size = img_size
        with tqdm(total=len(image_list)) as pbar:
            for image_dir in image_list:
                sub_image = Image.open(image_dir)
                sub_image = sub_image.resize((self.img_size,self.img_size))
                image_data.append(np.asarray(sub_image))
                pbar.update(1)

        self.data = np.array(image_data)
        self.targets = np.array([-1 for i in range(len(image_list))])
        self.data = transpose(normalize(self.data))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)