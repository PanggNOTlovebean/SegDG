from PIL import Image
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob 

origion_folder = '/data/Processed_data_nii'
site_list = ['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL']
for site_dir in site_list:
    # 输入文件夹路径和输出文件夹路径
    input_folder = os.path.join(origion_folder, site_dir)
    output_folder = os.path.join('data', f'{site_dir}')

    output_images_folder = os.path.join(output_folder, 'images')
    output_masks_folder = os.path.join(output_folder, 'masks')
    
    # 确保输出文件夹存在
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)
    if not os.path.exists(output_masks_folder):
        os.makedirs(output_masks_folder)
    

    # 循环遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件扩展名是否为".nii"或".nii.gz"
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            # 读取NII格式图像为NumPy数组
            nii_path = os.path.join(input_folder, filename)
            itk_img=sitk.ReadImage(nii_path)
            img_array = sitk.GetArrayFromImage(itk_img)
            print(img_array.shape)
            # 转换为PIL图像并保存为JPG格式
            for i in range(img_array.shape[0]):
                img_scaled = (img_array[i] - np.min(img_array[i])) / (np.max(img_array[i]) - np.min(img_array[i])) * 255
                img_uint8 = img_scaled.astype(np.uint8)
                img_slice = Image.fromarray(img_uint8, mode='L')
                jpg_filename = f"{os.path.splitext(filename)[0]}_slice{i+1}.jpg"
                if 'egm' in filename:
                    jpg_path = output_masks_folder
                else:
                    jpg_path = output_images_folder
                img_slice.save(os.path.join(jpg_path, jpg_filename))
