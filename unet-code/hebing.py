from PIL import Image
import os
import numpy as np
import tifffile as tiff

# 设置输入和输出文件夹路径
input_folder_path_1to3 = 'I:/Data/2023newdata/ESdata/huafen/srganyuce/mul1-3/train608/test'
input_folder_path_3to5 = 'I:/Data/2023newdata/ESdata/huafen/srganyuce/mul3-5/train608/test'
output_folder_path_merged = 'I:/Data/2023newdata/ESdata/huafen/srganyuce/srganhecheng/test'

# 获取第一个文件夹中所有tif文件
tif_files_1to3 = os.listdir(input_folder_path_1to3)

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder_path_merged):
    os.makedirs(output_folder_path_merged)

# 遍历第一个文件夹中的tif文件
for file_name in tif_files_1to3:
    # 读取第一个文件夹中的tif文件
    file_path_1to3 = os.path.join(input_folder_path_1to3, file_name)
    multiband_img_1to3 = Image.open(file_path_1to3)
    
    # 构建第四和第五通道（取第二个文件夹中的2到3通道）
    file_path_3to5 = os.path.join(input_folder_path_3to5, file_name)
    multiband_img_3to5 = Image.open(file_path_3to5)
    fourth_fifth_channels = multiband_img_3to5.split()[1:3]  # 取第二到第三通道作为第四和第五通道
    
    # 将图像转换为numpy数组
    multiband_img_1to3_array = np.array(multiband_img_1to3)
    fourth_channel_array = np.array(fourth_fifth_channels[0])
    fifth_channel_array = np.array(fourth_fifth_channels[1])
    
    # 合并为五通道图像
    merged_image_array = np.dstack((multiband_img_1to3_array, fourth_channel_array, fifth_channel_array))
    
    # 将numpy数组转换回Pillow的Image对象
    merged_image = Image.fromarray(merged_image_array, 'RGB')
    
    # 保存合并后的五通道图像
    output_file_name = os.path.splitext(file_name)[0] + '_merged.tif'
    output_file_path = os.path.join(output_folder_path_merged, output_file_name)
    merged_image.save(output_file_path)

# 如果需要的话，你还可以使用tifffile库来重新打开并保存合并后的五通道图像
# 注意：tifffile库中的保存函数可以直接保存具有多个通道的图像
# merged_image_np = np.array(merged_image)
# tiff.imwrite(output_file_path, merged_image_np)
