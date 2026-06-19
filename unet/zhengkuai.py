# import gdal
from osgeo import gdal
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import losses
import datetime
import math
import sys
import matplotlib.pyplot as plt
import cv2

#修改路径#
#''''''路径#
TifPath = r"G:\\df\\ENDING2\\3_2.tif"#整幅原始图像
ModelPath = r"model-new\\unet_model-3.hdf5"#训练好的模型
ResultPath = r"G:\\df\\ENDING2\\3_2Result.tif"#预测整幅标签
ROIResultPath = r"G:\\df\\ENDING2\\3_2ResultROI.tif"#整幅ROI去除背景
#  读取tif数据集
def readTif(fileName, xoff = 0, yoff = 0, data_width = 0, data_height = 0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize 
    #  栅格矩阵的行数
    height = dataset.RasterYSize 
    #  波段数
    bands = dataset.RasterCount 
    #  获取数据
    if(data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    #  获取仿射矩阵信息
    geotrans = dataset.GetGeoTransform()
    #  获取投影信息
    proj = dataset.GetProjection()
    return width, height, bands, data, geotrans, proj

#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape

    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

#  tif裁剪（tif像素数据，裁剪边长）
def TifCroppingArray(img, SideLength):
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int((img.shape[0] - SideLength * 2) / (256 - SideLength * 2))
    #  行上图像块数目
    RowNum = int((img.shape[1] - SideLength * 2) / (256 - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (256 - SideLength * 2) : i * (256 - SideLength * 2) + 256,
                          j * (256 - SideLength * 2) : j * (256 - SideLength * 2) + 256]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[i * (256 - SideLength * 2) : i * (256 - SideLength * 2) + 256,
                      (img.shape[1] - 256) : img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - 256) : img.shape[0],
                      j * (256-SideLength*2) : j * (256 - SideLength * 2) + 256]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[(img.shape[0] - 256) : img.shape[0],
                  (img.shape[1] - 256) : img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength * 2) % (256 - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength * 2) % (256 - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver

#  标签可视化，即为第n类赋上n值
def labelVisualize(img):
    img_out = np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #  为第n类赋上n值
            img_out[i][j] = np.argmax(img[i][j])
    return img_out

#  对测试图片进行归一化，并使其维度上和训练图片保持一致
def testGenerator(TifArray):
    for i in range(len(TifArray)):
        for j in range(len(TifArray[0])):
            img = TifArray[i][j]
            #  归一化
            img = img / 255.0
            #  在不改变数据内容情况下，改变shape
            img = np.reshape(img,(1,)+img.shape)
            yield img

#  获得结果矩阵
def Result(shape, TifArray, npyfile, num_class, RepetitiveLength, RowOver, ColumnOver):
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0  
    for i,item in enumerate(npyfile):
        img = labelVisualize(item)
        img = img.astype(np.uint8)
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if(i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 256 - RepetitiveLength, 0 : 256-RepetitiveLength] = img[0 : 256 - RepetitiveLength, 0 : 256 - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                #  原来错误的
                #result[shape[0] - ColumnOver : shape[0], 0 : 256 - RepetitiveLength] = img[0 : ColumnOver, 0 : 256 - RepetitiveLength]
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0 : 256 - RepetitiveLength] = img[256 - ColumnOver - RepetitiveLength : 256, 0 : 256 - RepetitiveLength]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       0:256-RepetitiveLength] = img[RepetitiveLength : 256 - RepetitiveLength, 0 : 256 - RepetitiveLength]   
        #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif(i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 256 - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0 : 256 - RepetitiveLength, 256 -  RowOver: 256]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0], shape[1] - RowOver : shape[1]] = img[256 - ColumnOver : 256, 256 - RowOver : 256]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       shape[1] - RowOver : shape[1]] = img[RepetitiveLength : 256 - RepetitiveLength, 256 - RowOver : 256]   
            #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 256 - RepetitiveLength,
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[0 : 256 - RepetitiveLength, RepetitiveLength : 256 - RepetitiveLength]         
            #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0],
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[256 - ColumnOver : 256, RepetitiveLength : 256 - RepetitiveLength]
            else:
                result[j * (256 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       (i - j * len(TifArray[0])) * (256 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (256 - 2 * RepetitiveLength) + RepetitiveLength,
                       ] = img[RepetitiveLength : 256 - RepetitiveLength, RepetitiveLength : 256 - RepetitiveLength]
    return result

area_perc = 0.5

#TifPath = sys.argv[1]
#ModelPath = sys.argv[2]
#ResultPath = sys.argv[3]
#area_perc = float(sys.argv[4])
RepetitiveLength = int((1 - math.sqrt(area_perc)) * 256 / 2)

#  记录测试消耗时间
testtime = []
#  获取当前时间
starttime = datetime.datetime.now()

im_width, im_height, im_bands, im_data, im_geotrans, im_proj = readTif(TifPath)
im_data = im_data.swapaxes(1, 0)
im_data = im_data.swapaxes(1, 2)

TifArray, RowOver, ColumnOver = TifCroppingArray(im_data, RepetitiveLength)
endtime = datetime.datetime.now()
text = "读取tif并裁剪预处理完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
print(text)
testtime.append(text)

model = load_model(ModelPath)
testGene = testGenerator(TifArray)
results = model.predict_generator(testGene,
                                  len(TifArray) * len(TifArray[0]),
                                  verbose = 1)
endtime = datetime.datetime.now()
text = "模型预测完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
print(text)
testtime.append(text)

#保存结果
result_shape = (im_data.shape[0], im_data.shape[1])
result_data = Result(result_shape, TifArray, results, 2, RepetitiveLength, RowOver, ColumnOver)
writeTiff(result_data, im_geotrans, im_proj, ResultPath)
endtime = datetime.datetime.now()
text = "结果拼接完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
print(text)
testtime.append(text)

time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
with open('timelog_%s.txt'%time, 'w') as f:
    for i in range(len(testtime)):
        f.write(testtime[i])
        f.write("\r\n")

###保存ROI####

# RepetitiveLength and area percentage setup
RepetitiveLength = int((1 - math.sqrt(0.5)) * 256 / 2)  # Example, adjust area percentage

# 记录测试消耗时间
testtime = []
starttime = datetime.datetime.now()

# 使用 GDAL 读取图像
dataset = gdal.Open(TifPath)  # 使用 GDAL 打开原始 TIFF 文件
num_bands = dataset.RasterCount
print(f"Number of bands: {num_bands}")

# 读取图像的各个通道
A = np.stack([dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(num_bands)], axis=-1)

# 如果是五通道图像，您可以选择提取前三个通道作为 RGB 或继续处理其他通道
if num_bands >= 3:
    R = A[:, :, 0]  # 红色通道
    G = A[:, :, 1]  # 绿色通道
    B = A[:, :, 2]  # 蓝色通道
else:
    raise ValueError("图像通道数不足，无法提取 RGB 通道")

# 读取标签图像（假设为灰度图像）
label = cv2.imread(ResultPath, cv2.IMREAD_GRAYSCALE)

# 标签掩膜处理，按位与操作
if label.max() > 1:
    label = label // 255  # 将标签归一化为 0 或 1

# 根据标签生成新的图像
IMGR = R * label
IMGG = G * label
IMGB = B * label

# 合成最终图像
IMG = np.stack([IMGR, IMGG, IMGB], axis=2).astype(np.uint8)

# 保存结果到 ROIResultPath
# Create a GDAL dataset for saving the result
driver = gdal.GetDriverByName('GTiff')
out_dataset = driver.Create(ROIResultPath, dataset.RasterXSize, dataset.RasterYSize, 3, gdal.GDT_Byte)

# Set geo-referencing information
out_dataset.SetGeoTransform(dataset.GetGeoTransform())
out_dataset.SetProjection(dataset.GetProjection())

# Write the bands to the new TIFF
out_dataset.GetRasterBand(1).WriteArray(IMGR)
out_dataset.GetRasterBand(2).WriteArray(IMGG)
out_dataset.GetRasterBand(3).WriteArray(IMGB)

# Close the dataset
out_dataset = None

# 显示结果
plt.imshow(IMG)
plt.title('ROI')
plt.axis('off')  # 不显示坐标轴
plt.tight_layout()
plt.show()

# 记录处理时间
endtime = datetime.datetime.now()
text = "结果保存完毕,目前耗时间: " + str((endtime - starttime).seconds) + "s"
print(text)
testtime.append(text)

# 保存时间日志
time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
with open('timelog_%s.txt' % time, 'w') as f:
    for i in range(len(testtime)):
        f.write(testtime[i])
        f.write("\r\n")
