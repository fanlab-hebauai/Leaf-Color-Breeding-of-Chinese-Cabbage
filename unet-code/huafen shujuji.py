import os
import random
import shutil
import time

def copyFile(fileDir, origion_path1, class_name):
    name = class_name
    path = origion_path1
    image_list = os.listdir(fileDir)
    image_number = len(image_list)
    train_number = int(image_number * train_rate)
    test_number = int(image_number * test_rate)
    val_number = image_number - train_number - test_number

    print(f"Class: {name}, Total images: {image_number}, Train: {train_number}, Test: {test_number}, Val: {val_number}")
    
    train_sample = random.sample(image_list, train_number)
    test_sample = random.sample(list(set(image_list) - set(train_sample)), test_number)
    val_sample = list(set(image_list) - set(train_sample) - set(test_sample))

    sample = [train_sample, test_sample, val_sample]
    
    for k in range(len(save_dir)):
        try:
            if os.path.isdir(save_dir[k]) and os.path.isdir(save_dir1[k]):
                for name in sample[k]:
                    name1 = name.split(".")[0] + '.tif'
                    src_image_path = os.path.join(fileDir, name)
                    src_label_path = os.path.join(path, name1)

                    if os.path.exists(src_image_path) and os.path.exists(src_label_path):
                        shutil.copy(src_image_path, os.path.join(save_dir[k], name))
                        shutil.copy(src_label_path, os.path.join(save_dir1[k], name1))
                    else:
                        print(f"Warning: Image or Label missing for {name}")
            else:
                os.makedirs(save_dir[k], exist_ok=True)
                os.makedirs(save_dir1[k], exist_ok=True)
                for name in sample[k]:
                    name1 = name.split(".")[0] + '.tif'
                    shutil.copy(os.path.join(fileDir, name), os.path.join(save_dir[k], name))
                    shutil.copy(os.path.join(path, name1), os.path.join(save_dir1[k], name1))
        except Exception as e:
            print(f"Error copying files for class {name}: {e}")

if __name__ == '__main__':
    time_start = time.time()

    origion_path = 'J:/df/mutispectralunet/DATAnew/1/3/'
    origion_path1 = 'J:/df/mutispectralunet/DATAnew/1/label/'

    save_train_dir = 'J:/df/3/train/'
    save_test_dir = 'J:/df/3/test/'
    save_val_dir = 'J:/df/3/val/'
    save_train_dir1 = 'J:/df/3/train_label/'
    save_test_dir1 = 'J:/df/3/test_label/'
    save_val_dir1 = 'J:/df/3/val_label/'
    save_dir = [save_train_dir, save_test_dir, save_val_dir]
    save_dir1 = [save_train_dir1, save_test_dir1, save_val_dir1]

    train_rate = 0.8
    test_rate = 0.1

    file_list = os.listdir(origion_path)
    num_classes = len(file_list)

    for i in range(num_classes):
        class_name = file_list[i]
        copyFile(origion_path, origion_path1, class_name)
        print(f"Finished copying class: {class_name}")

    print('数据集划分完毕!')
    time_end = time.time()
    print(f'---------------\n训练集和测试集划分共耗时: {time_end - time_start}秒')
