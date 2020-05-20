from PIL import Image
import pandas as pd
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

from configs import DATA_PATH

# DATA_PATH = '/gdata1/luyn/CD-FSL/data'  # Change to the dataset path
print(DATA_PATH)

CropDisease_path = DATA_PATH +'/CropDiseases'
EuroSAT_path = DATA_PATH +'/EuroSAT/2750'
ISIC_path = DATA_PATH + '/ISIC'
ChestX_path = DATA_PATH + '/chestX'
miniImagenet_train_path = DATA_PATH + '/miniImagenet'
save_path = DATA_PATH + '/processed'
ImageSize=256


def miniImagenet_train():
    db = ImageFolder(miniImagenet_train_path)
    temps = dict()
    for c in db.classes:
        temps[c] = []
    for i, (im, label) in enumerate(db):
        im = im.resize((ImageSize, ImageSize), resample=Image.BILINEAR)
        im = np.array(im)
        temps[db.classes[label]].append(im)
        if i % 100 == 0:
            print(i, i * 1.0 / len(db))
    for c in temps.keys():
        temps[c] = np.array(temps[c])
    np.save(save_path + '/' + 'miniImagenet_train.npy', temps)


def CropDiseases():
    db = ImageFolder(CropDisease_path + "/dataset/train/")
    temps = dict()
    for c in db.classes:
        temps[c] = []
    for i, (im, label) in enumerate(db):
        im = im.resize((ImageSize, ImageSize), resample=Image.BILINEAR)
        im = np.array(im)
        temps[db.classes[label]].append(im)
        if i % 100 == 0:
            print(i, i * 1.0 / len(db))
    for c in temps.keys():
        temps[c] = np.array(temps[c])
    np.save(save_path+ '/' + 'CropDisease.npy', temps)


def EuroSAT():
    db = ImageFolder(EuroSAT_path)
    temps = dict()
    for c in db.classes:
        temps[c] = []
    for i, (im, label) in enumerate(db):
        im = im.resize((ImageSize, ImageSize), resample=Image.BILINEAR)
        im = np.array(im)
        temps[db.classes[label]].append(im)
        if i % 100 == 0:
            print(i, i * 1.0 / len(db))
    for c in temps.keys():
        temps[c] = np.array(temps[c])
    np.save(save_path+ '/' + 'EuroSAT.npy', temps)


def ISIC():
    db = ISIC_db
    temps = dict()

    for i, (im, label) in enumerate(db):
        im = im.resize((ImageSize, ImageSize), resample=Image.BILINEAR)
        im = np.array(im)
        if label not in temps.keys():
            temps[label] = []
        temps[label].append(im)
        if i % 100 == 0:
            print(i, i * 1.0 / len(db))
    for c in temps.keys():
        temps[c] = np.array(temps[c])
    np.save(save_path + '/' + 'ISIC.npy', temps)


def ChesX():
    db = ChesX_db
    temps = dict()

    for i, (im, label) in enumerate(db):
        im = im.resize((ImageSize, ImageSize), resample=Image.BILINEAR)
        im = np.array(im)
        if label not in temps.keys():
            temps[label] = []
        temps[label].append(im)
        if i % 100 == 0:
            print(i, i * 1.0 / len(db))
    for c in temps.keys():
        temps[c] = np.array(temps[c])
    np.save(save_path + '/' + 'ChesX.npy', temps)


class ISIC_db(Dataset):
    def __init__(self,
                 csv_path= ISIC_path + "/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv",
                 image_path =  ISIC_path + "/ISIC2018_Task3_Training_Input/"):

        self.img_path = image_path
        self.csv_path = csv_path

        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name = np.asarray(self.data_info.iloc[:, 0])

        self.labels = np.asarray(self.data_info.iloc[:, 1:])
        self.labels = (self.labels!=0).argmax(axis=1)
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]
        # Open image
        img = Image.open(self.img_path +  single_image_name + ".jpg")

        single_image_label = self.labels[index]

        return (img, single_image_label)

    def __len__(self):
        return self.data_len


class ChesX_db(Dataset):
    def __init__(self, csv_path=ChestX_path+"/Data_Entry_2017.csv", \
        image_path = ChestX_path+"/images/"):

        self.img_path = image_path
        self.csv_path = csv_path
        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]

        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}

        labels_set = []

        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name  = []
        self.labels = []

        for name, label in zip(self.image_name_all,self.labels_all):
            label = label.split("|")

            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in self.used_labels:
                self.labels.append(self.labels_maps[label[0]])
                self.image_name.append(name)

        self.data_len = len(self.image_name)

        self.image_name = np.asarray(self.image_name)
        self.labels = np.asarray(self.labels)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]

        # Open image
        img_as_img = Image.open(self.img_path +  single_image_name).resize((ImageSize, ImageSize)).convert('RGB')

        single_image_label = self.labels[index]

        return (img_as_img, single_image_label)

    def __len__(self):
        return self.data_len


if __name__ == '__main__':
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    miniImagenet_train()
    CropDiseases()
    EuroSAT()
    ISIC()
    ChesX()
