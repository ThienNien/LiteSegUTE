import os
from numpy.core.fromnumeric import searchsorted
import torch
from torch.utils import data
from tqdm import tqdm
from torchvision import transforms
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
# from ex import BiSeNet
import numpy as np
from PIL import Image
from glob import glob


class Image_loader(Dataset):
    color_encoding = [
         ('Bird',(165,42,42)),
 ('Ground Animal',(0,192,0)),
 ('Curb',(196,196,196)),
 ('Fence',(190,153,153)),
 ('Guard Rail',(180,165,180)),
 ('Barrier',(90,120,150)),
 ('Wall',(102,102,156)),
 ('Bike Lane',(128,64,255)),
 ('Crosswalk - Plain',(140,140,200)),
 ('Curb Cut',(170,170,170)),
 ('Parking',(250,170,160)),
 ('Pedestrian Area',(96,96,96)),
 ('Rail_Track',(230,150,140)),
 ('Road',(128,64,128)),
 ('Service Lane',(110,110,110)),
 ('Sidewalk',(244,35,232)),
 ('Bridge',(150,100,100)),
 ('Building',(70,70,70)),
 ('Tunnel',(150,120,90)),
 ('Person',(220,20,60)),
 ('Bicyclist',(255,0,0)),
 ('Motorcyclist',(255,0,100)),
 ('Other Rider',(255,0,200)),
 ('Lane_Marking-Crosswalk',(200,128,128)),
 ('Lane_Marking-General',(255,255,255)),
 ('Mountain',(64,170,64)),
 ('Sand',(230,160,50)),
 ('Sky',(70,130,180)),
 ('Snow',(190,255,255)),
 ('Terrain',(152,251,152)),
 ('Vegetation',(107,142,35)),
 ('Water',(0,170,30)),
 ('Banner',(255,255,128)),
 ('Bench',(250,0,30)),
 ('BikeRack',(100,140,180)),
 ('Billboard',(220,220,220)),
 ('Catch Basin',(220,128,128)),
 ('CCTVCamera',(222,40,40)),
 ('FireHydrant',(100,170,30)),
 ('Junction Box',(40,40,40)),
 ('Mailbox',(33,33,33)),
 ('Manhole',(100,128,160)),
 ('Phone Booth',(142,0,0)),
 ('Pothole',(70,100,150)),
 ('Street Light',(210,170,100)),
 ('Pole',(153,153,153)),
 ('Traffic Sign Frame',(128,128,128)),
 ('Utility Pole',(0,0,80)),
 ('Traffic Light',(250,170,30)),
 ('Traffic Sign (Back)',(192,192,192)),
 ('Traffic Sign (Front)',(220,220,0)),
 ('Trash Can',(140,140,20)),
 ('Bicycle',(119,11,32)),
 ('Boat',(150,0,255)),
 ('Bus',(0,60,100)),
 ('Car',(0,0,142)),
 ('Caravan',(0,0,90)),
 ('Motorcycle',(0,0,230)),
 ('OnRails',(0,80,100)),
 ('Other Vehicle',(128,64,64)),
 ('Trailer',(0,0,110)),
 ('Truck',(0,0,70)),
 ('Wheeled Slow',(0,0,192)),
 ('CarMount',(32,32,32)),
 ('EgoVehicle',(120,10,10)),
 ('Unlabeled',(0,0,0))
        ]
 
    
      
    def __init__(self, num_classes=66,mode='train'):
        self.num_classes = num_classes
        self.mode=mode
        #Normalization
        self.normalize = transforms.Compose([
            transforms.Resize((360,640)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ]) ##imagenet norm

        self.DATA_PATH = os.path.join(os.getcwd())
        self.train_path, self.val_path, self.test_path = [os.path.join(self.DATA_PATH, x) for x in ['image/train/images','image/val/images','image/test/images']]

        if self.mode == 'train':
            self.data_files = self.get_files(self.train_path)
            self.label_files = [self.get_label_file(f, 'image/train/images', 'image/train/val_label') for f in self.data_files]
        elif self.mode == 'val':
            self.data_files = self.get_files(self.val_path)
            self.label_files = [self.get_label_file(f, 'image/iangevalid/images', 'image/val/val_label') for f in self.data_files]
        elif self.mode == 'test':
            self.data_files = self.get_files(self.test_path)
            self.label_files = [self.get_label_file(f, 'image/test/images', 'image/test/val_label') for f in self.data_files]
        else: 
            raise RuntimeError("Unexpected dataset mode."
                                "Supported modes: train, val, test")

    def __len__(self):
        return len(self.data_files)

    def get_files(self, data_folder):
        #
        return  glob("{}/*.{}".format(data_folder, 'jpg'))

    def get_label_file(self, data_path, data_dir, label_dir):
        #
        data_path = data_path.replace(data_dir, label_dir)
        frame, ext = data_path.split('.')
        ext='png'
        return "{}.{}".format(frame, ext)

    def image_loader(self, data_path, label_path):
        data = Image.open(data_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        return data, label.resize((640,360))

    def label_decode_cross_entropy(self, label):
        """
        Convert label image to matrix classes for apply cross entropy loss. 
        Return semantic index, label in enumemap of H x W x class
        """
        semantic_map = np.zeros(label.shape[:-1])
        #Fill all value with 0 - defaul
        semantic_map.fill(self.num_classes - 1) #self.num_classes - 1
        #Fill the pixel with correct class
        for class_index, color_info in enumerate(self.color_encoding):
            color = np.array(color_info[1])
            # print(color.shape)
            # print(label.shape)
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            semantic_map[class_map] = class_index
            # print(semantic_map)
        return semantic_map

    def __getitem__(self, index):
        """
            Args:
            - index (``int``): index of the item in the dataset
            Returns:
            A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
            of the image.
        """
        data_path, label_path = self.data_files[index], self.label_files[index]
        img, label = self.image_loader(data_path, label_path)
        # img.show()
        # label.show()
        # Normalize image
        img = self.normalize(img)
        # Convert label for cross entropy
        label = np.array(label)
        label = self.label_decode_cross_entropy(label)
        # print(label)
        label = torch.from_numpy(label).long()

        return img, label


# datasest = UIT()