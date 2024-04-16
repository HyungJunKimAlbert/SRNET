import os, cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from util import *
from PIL import Image

class SRdataset(Dataset):
    """
        Parameters:
            path (List): Image File Paths List 
        Returns
            Input Sample (np.array): Blurred Image
            Label Sample (np.array): Original Image
    """
    def __init__(self, paths, transform=False):
        self.paths = paths        
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        path = self.paths[idx]
        img = np.array(Image.open(path))
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32) 
        inp = cv2.GaussianBlur(img, (11, 11), 0)    

        if self.transform:
            input_sample = Image.fromarray(inp.squeeze())
            label_sample = Image.fromarray(img.squeeze())

            input_sample = self.transform(input_sample)
            label_sample = self.transform(label_sample)
            
        input_sample, label_sample = torch.tensor(inp, dtype=torch.float32), torch.tensor(img, dtype=torch.float32)

        return input_sample,label_sample
