import os, math, random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity 

def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)



def fix_seed(SEED=42):
    os.environ['SEED'] = str(SEED)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(SEED)

def psnr(label, outputs, max_val=1.):
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    img_diff = outputs - label
    rmse = math.sqrt(np.mean((img_diff)**2))
    if rmse == 0: # label과 output이 완전히 일치하는 경우
        return 100
    else:
        psnr = 20 * math.log10(max_val/rmse)
        return psnr


def ssim(label, outputs):
    label_tmp = label.squeeze(1).cpu().detach().numpy()
    outputs_tmp = outputs.squeeze(1).cpu().detach().numpy()
    ssim_value = structural_similarity(label_tmp, outputs_tmp, min=0, max=1)

    return ssim_value


def loss_plot(train_losses, valid_losses, dst_path):
    plt.clf()
    # Loss Plot
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(dst_path + '/loss_plot.png')  # Loss 그래프를 이미지 파일로 저장
    plt.show()

def psnr_plot(train_psnr, valid_psnr, dst_path):
    plt.clf()
    # AUC Plot
    plt.plot(train_psnr, label='Training PSNR')
    plt.plot(valid_psnr, label='Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('Training and Validation PSNR')
    plt.legend()
    plt.savefig(dst_path + '/psnr_plot.png')  # Loss 그래프를 이미지 파일로 저장
    plt.show()


def ssim_plot(train_ssim, valid_ssim, dst_path):
    plt.clf()
    # AUC Plot
    plt.plot(train_ssim, label='Training SSIM')
    plt.plot(valid_ssim, label='Validation SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Training and Validation SSIM')
    plt.legend()
    plt.savefig(dst_path + '/ssim_plot.png')  # Loss 그래프를 이미지 파일로 저장
    plt.show()


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth'):
        self.counter = 0            # 현재까지 기다린 epoch
        self.patience = patience    # 최대 유지 epoch
        self.verbose = verbose      # print 여부
        self.best_score = None      # best 성능
        self.early_stop = False     # 학습 중단 여부
        self.delta = delta          # 개선 최소변화량
        self.path = path            # 모델 경로
    
    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss >= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation Loss Decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)




import os, tqdm
import pydicom
import numpy as np
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut


def convert_image(df):
    dst_path = "/home/hjkim/projects/local_dev/dental_SRNet/data/tiff"

    for idx, values in tqdm.tqdm(enumerate(df.iterrows())):
        path, dir, file_name = df.path[idx], df.dir[idx], df.file_name[idx]
            
        dicom_path = os.path.join(path, dir, file_name)
        tiff_file_name = os.path.join(dst_path, f"raw_{idx+1}.tiff")

        # convert dicom to tiff
        dicom = pydicom.dcmread(dicom_path)
        data_array = apply_voi_lut(dicom.pixel_array, dicom)    # Preprocessing
        # Scaling (0~1)
        # min_value, max_value = np.min(data_array), np.max(data_array)
        min_value, max_value =  204.80720180045, 3071.0
        scaled_image_array = ((data_array - min_value) / (max_value - min_value))
        scaled_image_array = np.clip(scaled_image_array, 0, 1)

        pil_image = Image.fromarray(scaled_image_array)
        pil_image.save(tiff_file_name)   # save tiff image  
