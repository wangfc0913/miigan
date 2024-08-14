import re
import csv
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from models.networks import TextureDetector

file_path = "../checkpoints/imgan_vedai/"


def process_loss_log():

    # 打开 loss_log.txt 文件
    with open(file_path + 'loss_log.txt', 'r') as file:
        lines = file.readlines()

    # 匹配包含 "SSIM" 的行
    ssim_lines = [line for line in lines if "SSIM" in line]

    # 从每一行中解析出 epoch、SSIM、MSSIM、PSNR、LPIPS 的值
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

    # 将数据写入 CSV 文件
    with open(file_path + 'output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'SSIM', 'MSSIM', 'L1', 'PSNR', 'LPIPS'])  # 写入列名
        writer.writerows(data)  # 写入数据


def plot_metrix():
    # 从 CSV 文件读取数据
    df = pd.read_csv(file_path + 'output.csv')

    # 提取数据
    epoch = df['epoch']
    ssim = df['SSIM']
    mssim = df['MSSIM']
    l1 = df['L1']
    psnr = df['PSNR']
    lpips = df['LPIPS']

    # 使用Savitzky-Golay平滑滤波器对数据进行平滑处理
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

    # 创建新的图形并设置图形大小
    plt.figure(figsize=(10, 8))

    # 绘制SSIM曲线图
    plt.subplot(3, 2, 1)
    plt.plot(epoch, ssim_smooth, label='SSIM', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM over Epochs')
    plt.legend(loc='lower right')
    plt.grid(True)

    # 找到SSIM曲线的最大值并标出
    max_ssim_index = np.argmax(ssim_smooth)
    max_ssim = ssim_smooth[max_ssim_index]
    plt.plot(epoch[max_ssim_index], max_ssim, 'o', color='orange')
    plt.text(epoch[max_ssim_index], max_ssim, f'Epoch: {max_ssim_index+1}, Max Value: {max_ssim:.3f}', ha='right')

    # 绘制MSSIM曲线图
    plt.subplot(3, 2, 2)
    plt.plot(epoch, mssim_smooth, label='MSSIM', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSSIM')
    plt.title('MSSIM over Epochs')
    plt.legend(loc='lower right')
    plt.grid(True)

    # 找到MSSIM曲线的最大值并标出
    max_mssim_index = np.argmax(mssim_smooth)
    max_mssim = mssim_smooth[max_mssim_index]
    plt.plot(epoch[max_mssim_index], max_mssim, 'o', color='orange')
    plt.text(epoch[max_mssim_index], max_mssim, f'Epoch: {max_mssim_index+1}, Max Value: {max_mssim:.3f}', ha='right')

    # 绘制L1曲线图
    plt.subplot(3, 2, 3)
    plt.plot(epoch, l1_smooth, label='L1', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('L1')
    plt.title('L1 over Epochs')
    plt.legend(loc='upper right')
    plt.grid(True)

    # 找到L1曲线的最大值并标出
    min_l1_index = np.argmin(l1_smooth)
    min_l1 = l1_smooth[min_l1_index]
    plt.plot(epoch[min_l1_index], min_l1, 'o', color='orange')
    plt.text(epoch[min_l1_index], min_l1, f'Epoch: {min_l1_index+1}, Max Value: {min_l1:.3f}', ha='right')

    # 绘制PSNR曲线图
    plt.subplot(3, 2, 4)
    plt.plot(epoch, psnr_smooth, label='PSNR', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('PSNR over Epochs')
    plt.legend(loc='lower right')
    plt.grid(True)

    # 找到PSNR曲线的最大值并标出
    max_psnr_index = np.argmax(psnr_smooth)
    max_psnr = psnr_smooth[max_psnr_index]
    plt.plot(epoch[max_psnr_index], max_psnr, 'o', color='orange')
    plt.text(epoch[max_psnr_index], max_psnr, f'Epoch: {max_psnr_index+1}, Max Value: {max_psnr:.3f}', ha='right')

    # 绘制LPIPS曲线图
    plt.subplot(3, 2, 5)
    plt.plot(epoch, lpips_smooth, label='LPIPS', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('LPIPS')
    plt.title('LPIPS over Epochs')
    plt.legend()
    plt.grid(True)

    # 找到LPIPS曲线的最大值并标出
    min_lpips_index = np.argmin(lpips_smooth)
    min_lpips = lpips_smooth[min_lpips_index]
    plt.plot(epoch[min_lpips_index], min_lpips, 'o', color='orange')
    plt.text(epoch[min_lpips_index], min_lpips, f'Epoch: {min_lpips_index+1}, Min Value: {min_lpips:.3f}', ha='right')

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.savefig(file_path + "matrix.png")
    plt.show()


def test_texture():
    # 加载图像
    image = Image.open("./data/0001_rgb.tiff")  # 替换为您自己的图像路径
    # 图像预处理
    preprocess = transforms.Compose([
        transforms.ToTensor()
    ])
    image = preprocess(image).unsqueeze(0)  # 添加 batch 维度

    # create model
    model = TextureDetector()
    output = model(image)

    # 可视化结果
    plt.imshow(output.squeeze(0).permute(1, 2, 0).detach().numpy(), cmap='gray')
    plt.title('Texture Detected')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    process_loss_log()
    plot_metrix()
