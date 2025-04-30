# dataset of vehicles for training model
import os
import cv2
import shutil
import torch
from sklearn.model_selection import train_test_split

# create folder for train, validation and test data
data_path = "dataset/vehicles_8/training_yolov5"
if os.path.isdir(data_path):
    shutil.rmtree(data_path)
for folder in ["images", "labels"]:
    os.makedirs(os.path.join(data_path, folder))

# create list of file paths of images and labels
file_path = "dataset/vehicles_8/train/images/train.txt"
file = open(file_path)
data_paths = [path.replace("\n", "") for path in file.readlines()]
file.close()

image_paths = ["dataset/vehicles_8/{}".format(path.replace("../", "")) for path in data_paths]
label_paths = [path.replace("/images/", "/labels/").replace(".jpg", ".txt") for path in image_paths]

# split the dataset into train (6000), validation (2000), and test data (218)
train_images, temp_images, train_labels, temp_labels = train_test_split(image_paths, label_paths, train_size=6000, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, train_size=2000, random_state=42)

split_folders = ["train", "val", "test"]
for folder in split_folders:
    if not os.path.isdir(os.path.join(data_path, "images", folder)):
        os.makedirs(os.path.join(data_path, "images", folder))
    if not os.path.isdir(os.path.join(data_path, "labels", folder)):
        os.makedirs(os.path.join(data_path, "labels", folder))


# copy the dataset into according folders
def copy_files(path_list, destination_folder):
    for path in path_list:
        shutil.copy2(path, destination_folder)

copy_files(train_images, os.path.join(data_path, "images/train/"))
copy_files(val_images, os.path.join(data_path, "images/val"))
copy_files(test_images, os.path.join(data_path, "images/test/"))
copy_files(train_labels, os.path.join(data_path, "labels/train/"))
copy_files(val_labels, os.path.join(data_path, "labels/val/"))
copy_files(test_labels, os.path.join(data_path, "labels/test/"))



# only for checking cuda available, verifying if bounding box matches the pattern
'''
print(torch.cuda.is_available())

test_image_path = "dataset/vehicles_8/train/images/highway_3610_2020-08-26.jpg"
test_label_path = "dataset/vehicles_8/train/labels/highway_3610_2020-08-26.txt"
label_file = open(test_label_path)
label = [line.replace("\n", "") for line in label_file.readlines()]
label_file.close()
image = cv2.imread(test_image_path)
width, height, _ = image.shape
for bbox in label:
    bbox_split = bbox.split(" ")
    xmin = (float(bbox_split[1]) - float(bbox_split[3]) / 2) * width
    ymin = (float(bbox_split[2]) - float(bbox_split[4]) / 2) * height
    xmax = (float(bbox_split[1]) + float(bbox_split[3]) / 2) * width
    ymax = (float(bbox_split[2]) + float(bbox_split[4]) / 2) * height

    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0))
cv2.imshow("test", image)
cv2.waitKey(0)
'''