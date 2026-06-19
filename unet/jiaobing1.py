import numpy as np
import cv2
import os

def color_dict(label_folder, class_num):
    """获取颜色字典"""
    color_set = set()
    for img_name in os.listdir(label_folder):
        img_path = os.path.join(label_folder, img_name)
        img = cv2.imread(img_path).astype(np.uint32)
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        encoded = img[:, :, 0] * 1000000 + img[:, :, 1] * 1000 + img[:, :, 2]
        color_set.update(np.unique(encoded))
        if len(color_set) >= class_num:
            break
    color_list = sorted(color_set)
    color_dict_bgr = []
    for val in color_list:
        color_str = str(val).rjust(9, '0')
        color_bgr = [int(color_str[:3]), int(color_str[3:6]), int(color_str[6:])]
        color_dict_bgr.append(color_bgr)
    color_dict_bgr = np.array(color_dict_bgr, dtype=np.uint8)
    color_gray = cv2.cvtColor(color_dict_bgr.reshape((-1, 1, 3)), cv2.COLOR_BGR2GRAY)
    return color_dict_bgr, color_gray

def ConfusionMatrix(num_class, pred, label):
    mask = (label >= 0) & (label < num_class)
    combined = num_class * label[mask] + pred[mask]
    count = np.bincount(combined, minlength=num_class**2)
    return count.reshape((num_class, num_class))

def OverallAccuracy(conf_matrix):
    return np.diag(conf_matrix).sum() / conf_matrix.sum()

def Precision(conf_matrix):
    return np.diag(conf_matrix) / np.maximum(conf_matrix.sum(axis=0), 1e-6)

def Recall(conf_matrix):
    return np.diag(conf_matrix) / np.maximum(conf_matrix.sum(axis=1), 1e-6)

def F1Score(conf_matrix):
    prec = Precision(conf_matrix)
    rec = Recall(conf_matrix)
    return 2 * prec * rec / np.maximum(prec + rec, 1e-6)

def IntersectionOverUnion(conf_matrix):
    intersection = np.diag(conf_matrix)
    union = conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - intersection
    return intersection / np.maximum(union, 1e-6)

def MeanIntersectionOverUnion(conf_matrix):
    return np.nanmean(IntersectionOverUnion(conf_matrix))

def Frequency_Weighted_Intersection_over_Union(conf_matrix):
    freq = conf_matrix.sum(axis=1) / conf_matrix.sum()
    iou = IntersectionOverUnion(conf_matrix)
    return (freq * iou).sum()

# ---------- 主流程 ----------
LabelPath = r"G:\df\3\train_label"
PredictPath = r"DATAnew\3\savetrain"
class_num = 2

color_dict_bgr, color_dict_gray = color_dict(LabelPath, class_num)

label_files = sorted(os.listdir(LabelPath))
predict_files = sorted(os.listdir(PredictPath))
assert len(label_files) == len(predict_files), "预测和标签数量不一致"

# 获取图像大小
sample_img = cv2.imread(os.path.join(LabelPath, label_files[0]), 0)
H, W = sample_img.shape
num_images = len(label_files)

label_all = np.zeros((num_images, H, W), dtype=np.uint8)
predict_all = np.zeros((num_images, H, W), dtype=np.uint8)

for i, (label_name, predict_name) in enumerate(zip(label_files, predict_files)):
    label = cv2.imread(os.path.join(LabelPath, label_name), cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(os.path.join(PredictPath, predict_name), cv2.IMREAD_GRAYSCALE)
    if label is None or pred is None:
        continue
    label_all[i] = label
    predict_all[i] = pred

# 灰度值映射为类别编号
for idx, gray_val in enumerate(color_dict_gray.flatten()):
    label_all[label_all == gray_val] = idx
    predict_all[predict_all == gray_val] = idx

# 展平
label_flat = label_all.flatten()
predict_flat = predict_all.flatten()

# 计算指标
conf_mat = ConfusionMatrix(class_num, predict_flat, label_flat)
precision = Precision(conf_mat)
recall = Recall(conf_mat)
f1 = F1Score(conf_mat)
iou = IntersectionOverUnion(conf_mat)
miou = MeanIntersectionOverUnion(conf_mat)
fw_iou = Frequency_Weighted_Intersection_over_Union(conf_mat)
oa = OverallAccuracy(conf_mat)

# 打印输出
print("类别标签 (灰度值):", color_dict_gray.flatten())
print("混淆矩阵:\n", conf_mat)
print("精确度 (Precision):", precision)
print("召回率 (Recall):", recall)
print("F1 分数:", f1)
print("整体精度 (OA):", oa)
print("IoU:", iou)
print("平均 IoU (mIoU):", miou)
print("频权 IoU (FWIoU):", fw_iou)
