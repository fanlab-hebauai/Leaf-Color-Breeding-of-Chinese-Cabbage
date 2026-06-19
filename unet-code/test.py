from Model.seg_unet import unet
from dataProcess import testGenerator, saveResult, color_dict
import os
import tensorflow as tf

# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#训练模型保存地址
model_path=r"G:/df/model-new/unet_model-5.hdf5"
#  测试数据路径
test_iamge_path = r"G:/df/5/YINGSHE/test"

#  结果保存路径
save_path = r"DATAnew/3/savetest、"
#  测试数据数目
test_num = len(os.listdir(test_iamge_path))
#  类的数目(包括背景)
classNum = 2
#  模型输入图像大小
input_size = (608, 608,5)
#  生成图像大小
output_size = (608,608)
#  训练数据标签路径
train_label_path =r"G:/df/5/train_label"
#  标签的颜色字典
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)

model = unet(model_path)

testGene = testGenerator(test_iamge_path, input_size)

#  预测值的Numpy数组
results = model.predict_generator(testGene,
                                  test_num,
                                  verbose = 1)

#  保存结果
saveResult(test_iamge_path, save_path, results, colorDict_GRAY, output_size)