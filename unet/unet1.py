import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Model.unet import unet_with_transformer
#from Model.seg_hrnet import seg_hrnet
from dataProcess import trainGenerator, color_dict
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import datetime
import xlwt
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"   #//这里是自己的GPU编号，我是"0",原文是"8".

'''
数据集相关参数
'''
#  训练数据图像路径
train_image_path = "D:\\zjsrdata\\srganhecheng\\train - 副本"
#  训练数据标签路径
train_label_path = "D:\\zjsrdata\\label\\train"
#  验证数据图像路径
validation_image_path = "D:\\zjsrdata\\srganhecheng\\test"
#  验证数据标签路径
validation_label_path = "D:\\zjsrdata\\label\\test"

'''
模型相关参数
'''
#  批大小
batch_size =1
#  类的数目(包括背景)
classNum = 2
#  模型输入图像大小
input_size = (608,608,5)
#  训练模型的迭代总轮数
epochs = 120
#  初始学习率
learning_rate = 1e-5
#  预训练模型地址
premodel_path = None #None
#  训练模型保存地址
model_path = "model-tf\\unet_modelmul120newkuozengsrgan.hdf5"
# model_path = "model-new\\unet_modelrgb.hdf5"
#  训练数据数目
train_num = len(os.listdir(train_image_path))
#  验证数据数目
validation_num = len(os.listdir(validation_image_path))

#  训练集每个epoch有多少个batch_size
steps_per_epoch = train_num / batch_size
#  验证集每个epoch有多少个batch_size
validation_steps = validation_num / batch_size
#  标签的颜色字典,用于onehot编码
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)


#  得到一个生成器，以batch_size的速率生成训练数据
train_Generator = trainGenerator(batch_size,
                                 train_image_path, 
                                 train_label_path,
                                 classNum ,
                                 colorDict_GRAY,
                                 input_size)

#  得到一个生成器，以batch_size的速率生成验证数据
validation_data = trainGenerator(batch_size,
                                 validation_image_path,
                                 validation_label_path,
                                 classNum,
                                 colorDict_GRAY,
                                 input_size)
#  定义模型
model = unet_with_transformer(pretrained_weights = premodel_path, 
             input_size = input_size, 
             classNum = classNum, 
             learning_rate = learning_rate)
#  打印模型结构

model.summary()
#  回调函数
#  val_loss连续10轮没有下降则停止训练
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 100)
#  当3个epoch过去而val_loss不下降时，学习率减半
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, verbose = 1)
model_checkpoint = ModelCheckpoint(model_path,
                                   monitor = 'loss',
                                   verbose = 1,# 日志显示模式:0->安静模式,1->进度条,2->每轮一行
                                   save_best_only = True)

#  获取当前时间
start_time = datetime.datetime.now()

#  模型训练
history = model.fit_generator(train_Generator,
                    steps_per_epoch = steps_per_epoch,
                    epochs = epochs,
                    callbacks = [early_stopping, model_checkpoint, model_checkpoint],
                    validation_data = validation_data,
                    validation_steps = validation_steps)

