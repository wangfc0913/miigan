import re
import csv
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from models.networks import TextureDetector

file_path = "../checkpoints/MIIGAN_VEDAI_512/"   # path


def process_loss_log():

    with open(file_path + 'loss_log.txt', 'r') as file:
        lines = file.readlines()

    ssim_lines = [line for line in lines if "SSIM" in line]

    data = []
    for line in ssim_lines:
        match = re.search(
            r'epoch:(\d+).*?\bSSIM:([-+]?\d*\.\d+|\d+).*?MSSIM:([-+]?\d*\.\d+|\d+).*?L1:([-+]?\d*\.\d+|\d+).*?PSNR:([-+]?\d*\.\d+|\d+).*?LPIPS:([-+]?\d*\.\d+|\d+)',
            line)
        if match:
            epoch = match.group(1)
            ssim = match.group(2)
            mssim = match.group(3)
            l1 = match.group(4)
            psnr = match.group(5)
            lpips = match.group(6)
            data.append((epoch, ssim, mssim, l1, psnr, lpips))

    with open(file_path + 'output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'SSIM', 'MSSIM', 'L1', 'PSNR', 'LPIPS'])  
        writer.writerows(data)  


def plot_metrix():
    df = pd.read_csv(file_path + 'output.csv')

    epoch = df['epoch']
    ssim = df['SSIM']
    mssim = df['MSSIM']
    l1 = df['L1']
    psnr = df['PSNR']
    lpips = df['LPIPS']

    window_size = 11
    order = 3

    # ssim_smooth = savgol_filter(ssim, window_size, order)
    # mssim_smooth = savgol_filter(mssim, window_size, order)
    # psnr_smooth = savgol_filter(psnr, window_size, order)
    # lpips_smooth = savgol_filter(lpips, window_size, order)

    ssim_smooth = ssim
    mssim_smooth = mssim
    l1_smooth = l1
    psnr_smooth = psnr
    lpips_smooth = lpips

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 2, 1)
    plt.plot(epoch, ssim_smooth, label='SSIM', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM over Epochs')
    plt.legend(loc='lower right')
    plt.grid(True)

    max_ssim_index = np.argmax(ssim_smooth)
    max_ssim = ssim_smooth[max_ssim_index]
    plt.plot(epoch[max_ssim_index], max_ssim, 'o', color='orange')
    plt.text(epoch[max_ssim_index], max_ssim, f'Epoch: {max_ssim_index+1}, Max Value: {max_ssim:.3f}', ha='right')

    plt.subplot(3, 2, 2)
    plt.plot(epoch, mssim_smooth, label='MSSIM', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSSIM')
    plt.title('MSSIM over Epochs')
    plt.legend(loc='lower right')
    plt.grid(True)

    max_mssim_index = np.argmax(mssim_smooth)
    max_mssim = mssim_smooth[max_mssim_index]
    plt.plot(epoch[max_mssim_index], max_mssim, 'o', color='orange')
    plt.text(epoch[max_mssim_index], max_mssim, f'Epoch: {max_mssim_index+1}, Max Value: {max_mssim:.3f}', ha='right')

    plt.subplot(3, 2, 3)
    plt.plot(epoch, l1_smooth, label='L1', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('L1')
    plt.title('L1 over Epochs')
    plt.legend(loc='upper right')
    plt.grid(True)

    min_l1_index = np.argmin(l1_smooth)
    min_l1 = l1_smooth[min_l1_index]
    plt.plot(epoch[min_l1_index], min_l1, 'o', color='orange')
    plt.text(epoch[min_l1_index], min_l1, f'Epoch: {min_l1_index+1}, Max Value: {min_l1:.3f}', ha='right')

    plt.subplot(3, 2, 4)
    plt.plot(epoch, psnr_smooth, label='PSNR', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('PSNR over Epochs')
    plt.legend(loc='lower right')
    plt.grid(True)

    max_psnr_index = np.argmax(psnr_smooth)
    max_psnr = psnr_smooth[max_psnr_index]
    plt.plot(epoch[max_psnr_index], max_psnr, 'o', color='orange')
    plt.text(epoch[max_psnr_index], max_psnr, f'Epoch: {max_psnr_index+1}, Max Value: {max_psnr:.3f}', ha='right')

    plt.subplot(3, 2, 5)
    plt.plot(epoch, lpips_smooth, label='LPIPS', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('LPIPS')
    plt.title('LPIPS over Epochs')
    plt.legend()
    plt.grid(True)

    min_lpips_index = np.argmin(lpips_smooth)
    min_lpips = lpips_smooth[min_lpips_index]
    plt.plot(epoch[min_lpips_index], min_lpips, 'o', color='orange')
    plt.text(epoch[min_lpips_index], min_lpips, f'Epoch: {min_lpips_index+1}, Min Value: {min_lpips:.3f}', ha='right')

    plt.tight_layout()

    plt.savefig(file_path + "matrix.png")
    plt.show()


if __name__ == '__main__':
    process_loss_log()
    plot_metrix()
