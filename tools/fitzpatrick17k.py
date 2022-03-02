import torch
import pandas as pd 
import os
import imageio as im
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
from torchvision import transforms


class FitzpatrickDataset(Dataset):
    def __init__(self,df_images,image_dir_path,target='label',transform = None):
        '''
        Args:
        df_images: pandas dataframe with label as one hot encode
        image_dir_path: path to directory with all the images
        '''



        self.df = df_images
        one_hot_aux = pd.get_dummies(self.df[target])
        self.df = self.df.join(one_hot_aux)
        self.image_dir_path =  image_dir_path
        self.transform = transform
        self.label = self.df.drop(['md5hash','fitzpatrick','label','nine_partition_label','three_partition_label','url','url_alphanum'],axis = 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir_path,self.df.iloc[idx]['md5hash']+'.jpg')

        img = im.imread(img_name)

        annotations = self.df.iloc[idx]

        sample = {'image': img, 'label': self.label.iloc[idx].values}

        if self.transform:
            sample['image'] = self.transform(sample['image'])


        return sample