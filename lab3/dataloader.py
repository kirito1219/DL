import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision import transforms

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
#         print("> Found %d images..." % (len(self.img_name)))
        MEAN = [108.64628601 / 255, 75.86886597 / 255, 54.34005737 / 255]
        STD = [70.53946096 / 255, 51.71475228 / 255, 43.03428563 / 255]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=512,
                scale=(1 / 1.15, 1.15),
                ratio=(0.7561, 1.3225)
            ),
            transforms.RandomAffine(
                degrees=(-180, 180),
                translate=(40 / 448, 40 / 448),
                scale=None,
                shear=None
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(tuple(MEAN), tuple(STD)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(tuple(MEAN), tuple(STD))
        ])
        self.transfrom = None
        if mode == 'train':
            self.transform = train_transform
        else:
            self.transform = test_transform

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        
        path = os.path.join('.', self.root, self.img_name[index] + '.jpeg')
        img = Image.open(path)
        img = self.transform(img)
        label = self.label[index]
        
        return img, label
