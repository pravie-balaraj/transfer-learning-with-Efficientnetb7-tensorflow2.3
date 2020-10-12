import shutil
import glob
import os
import random

original_imagenet_raw_images = 'D:\\imagenet_images\\original_images'
original_imagenet_processed_images = 'D:\\imagenet_images\\dataset'

original_imagenet_processed_images_train = original_imagenet_processed_images + 'train\\'
original_imagenet_processed_images_val = original_imagenet_processed_images + 'val\\'
original_imagenet_processed_images_test = original_imagenet_processed_images + 'test\\'

class_folders = os.listdir(original_imagenet_raw_images)

for fol in class_folders:
    os.makedirs(original_imagenet_processed_images_train+fol,  exist_ok=True)
    
for fol in class_folders:
    os.makedirs(original_imagenet_processed_images_val+fol, exist_ok=True)
    
for fol in class_folders:
    os.makedirs(original_imagenet_processed_images_test+fol, exist_ok=True)
    

train_portion = 0.6
test_val_ratio = 0.5

for folder in os.scandir(original_imagenet_raw_images):
    all_files = glob.glob(folder.path+'\\*')
    train = random.sample(all_files, int(len(all_files) * train_portion))
    for files in train:
        image_name = files.split('\\')[-1].split('.')[0]
        file_name = image_name+'.jpg'
        shutil.move(files, original_imagenet_processed_images_train + folder.name + '\\' + file_name)

    all_files = glob.glob(folder.path+'\\*')
    val = random.sample(all_files, int(len(all_files) * test_val_ratio))
    for files in val:
        image_name = files.split('\\')[-1].split('.')[0]
        file_name = image_name+'.jpg'
        shutil.move(files, original_imagenet_processed_images_val + folder.name + '\\' + file_name)

    all_files = glob.glob(folder.path+'\\*')
    for files in all_files:
        image_name = files.split('\\')[-1].split('.')[0]
        file_name = image_name+'.jpg'
        shutil.move(files, original_imagenet_processed_images_test + folder.name + '\\' + file_name)