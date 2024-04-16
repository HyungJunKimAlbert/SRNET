import os, warnings
import numpy as np
import imageio

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim as optim
from torchvision.utils import save_image
from torchvision import transforms

from model import *
from datasets import SRdataset
from util import *
from losses import PerceptualLoss_VGG, PerceptualLoss_ResNet, ssim_loss
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

def train_one_epoch(model, data_dl, optimizer, criterion, device):
    model.train()
    running_loss, running_psnr, running_ssim = 0.0, 0.0, 0.0

    for batch_idx, (image, label) in enumerate(data_dl):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        batch_psnr = psnr(label, outputs)
        batch_ssim = ssim(label, outputs)
        
        running_loss += loss.item()
        running_psnr += batch_psnr
        running_ssim += batch_ssim

    final_loss = running_loss / (batch_idx+1)
    final_psnr = running_psnr / (batch_idx+1)
    final_ssim = running_ssim / (batch_idx+1)

    return final_loss, final_psnr, final_ssim


def valid_one_epoch(model, data_dl, criterion, device):
    model.eval()
    running_loss, running_psnr, running_ssim = 0.0, 0.0, 0.0

    with torch.no_grad():

        for batch_idx, (image, label) in enumerate(data_dl):
            image = image.to(device)
            label = label.to(device)

            outputs = model(image)
            loss = criterion(outputs, label)

            batch_psnr = psnr(label, outputs)
            batch_ssim = ssim(label, outputs)

            running_loss += loss.item()
            running_psnr += batch_psnr
            running_ssim += batch_ssim

        final_loss = running_loss / (batch_idx+1)
        final_psnr = running_psnr / (batch_idx+1)
        final_ssim = running_ssim / (batch_idx+1)

    return final_loss, final_psnr, final_ssim


def valid_test(model, data_dl, criterion, device, output_dir="/home/hjkim/projects/local_dev/dental_SRNet/result"):
    model.eval()  # 모델을 평가 모드로 설정
    os.makedirs(output_dir, exist_ok=True)  # 결과를 저장할 디렉토리 생성
    running_loss, running_psnr, running_ssim = 0.0, 0.0, 0.0
    min_value, max_value =  204.80720180045, 3071.0

    with torch.no_grad():

        for batch_idx, (image, label) in enumerate(data_dl):
            image = image.to(device)
            label = label.to(device)
            outputs = model(image)
            loss = criterion(outputs, label)

            batch_psnr = psnr(label, outputs)
            batch_ssim = ssim(label, outputs)

            running_loss += loss.item()
            running_psnr += batch_psnr
            running_ssim += batch_ssim

            # Save results
            image = image.detach().cpu()  
            outputs = outputs.detach().cpu() 
            label = label.detach().cpu() 

            for i in range(outputs.size(0)):
                denorm_image = ( (image[i][0,:,:] * (max_value - min_value)) + min_value )
                denorm_output = ( (outputs[i][0,:,:] * (max_value - min_value)) + min_value )
                denorm_label = ( (label[i][0,:,:] * (max_value - min_value)) + min_value )
                
                imageio.imwrite(os.path.join(output_dir, f'input_{batch_idx * test_dl.batch_size + i}.tiff'), denorm_image)
                imageio.imwrite(os.path.join(output_dir, f'output_{batch_idx * test_dl.batch_size + i}.tiff'), denorm_output)
                imageio.imwrite(os.path.join(output_dir, f'label_{batch_idx * test_dl.batch_size + i}.tiff'), denorm_label)
                # label[i][:,:,0].numpy().save(os.path.join(output_dir, f'label_{batch_idx * test_dl.batch_size + i}.tiff'))

        final_loss = running_loss / (batch_idx+1)
        final_psnr = running_psnr / (batch_idx+1)
        final_ssim = running_ssim / (batch_idx+1)


    return final_loss, final_psnr, final_ssim

def split_dataset(file_path = "/home/hjkim/projects/local_dev/dental_SRNet/data/tiff"):
       
    # Total sample
    file_list = os.listdir(file_path)
    num_sample = len(file_list)
    total_indices = list(range(num_sample))

    val_test_ratio = 0.3  

    # train과 validation, test 데이터셋을 위한 인덱스 추출
    train_indices, temp_indices = train_test_split(total_indices, test_size=val_test_ratio, random_state=42)    # 7 : 3
    val_test_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42) # 5 : 5

    train_data = [os.path.join(file_path, file_list[idx]) for idx in train_indices]
    valid_data = [os.path.join(file_path, file_list[idx]) for idx in val_test_indices]
    test_data = [os.path.join(file_path, file_list[idx]) for idx in test_indices]
    print(f"TRAIN: {len(train_data)}, VALID: {len(valid_data)}, TEST: {len(test_data)}")

    return train_data, valid_data, test_data


if __name__ == "__main__":
    
    # Environment
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TORCH_USE_CUDA_DSA"] = '1'

    # Options 
    EPOCHS = 200
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    CHECKPOINT_PATH = "/home/hjkim/projects/local_dev/dental_SRNet/checkpoint/best_model.pth"
    PLOT_PATH = "/home/hjkim/projects/local_dev/dental_SRNet/checkpoint"

    fix_seed(42)
    device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {device} ")
    # model = SRCNN().to(device)
    model = ConvUNet(in_channels=1, out_channels=1).to(device)
    # model.apply(initialize_weights)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # Augmentation
    transform = {'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),  
        transforms.RandomVerticalFlip(),  
        transforms.RandomRotation(15),    
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.ToTensor(),             
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
    ])
    }

    # Import Dataset
    train_data, valid_data, test_data = split_dataset()

    train_ds = SRdataset(train_data)    # transform=transform['train']
    valid_ds = SRdataset(valid_data)
    test_ds = SRdataset(test_data)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)

    print(f"TRAIN: {len(train_dl)}, VALID: {len(valid_dl)}, TEST: {len(test_dl)}")
    train_loss, train_psnr, train_ssim = [], [], []
    valid_loss, valid_psnr, valid_ssim = [], [], []
    early_stopping = EarlyStopping(patience=5, verbose=True, path=CHECKPOINT_PATH)

    # Training 
    for epoch in range(EPOCHS):
        # Train
        train_epoch_loss, train_epoch_psnr, train_epoch_ssim = train_one_epoch(model, train_dl, optimizer, criterion, device)
        
        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)
        train_ssim.append(train_epoch_ssim)

        # Valid
        valid_epoch_loss, valid_epoch_psnr, valid_epoch_ssim = valid_one_epoch(model, valid_dl, criterion, device)
        
        valid_loss.append(valid_epoch_loss)
        valid_psnr.append(valid_epoch_psnr)
        valid_ssim.append(valid_epoch_ssim)

        early_stopping(valid_epoch_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(f"Epoch: {epoch+1}, Train Loss: {train_epoch_loss:.6f}, Train PSNR: {train_epoch_psnr:.6f}, Train SSIM: {train_epoch_ssim:.6f}, Valid Loss: {valid_epoch_loss:.6f}, Valid PSNR: {valid_epoch_psnr:.6f}, Valid SSIM: {valid_epoch_ssim:.6f}")

    # Save Plot
    loss_plot(train_loss, valid_loss, PLOT_PATH)
    psnr_plot(train_psnr, valid_psnr, PLOT_PATH)
    ssim_plot(train_ssim, valid_ssim, PLOT_PATH)

    # Test 
    # model = SRCNN().to(device)
    model = ConvUNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    test_epoch_loss, test_epoch_psnr, test_ssim = valid_test(model, test_dl, criterion, device)
    print("FINAL")

    print(f"Test Loss: {test_epoch_loss:.6f}, Test PSNR: {test_epoch_psnr:.6f}, Test SSIM: {test_ssim:.6f}")
