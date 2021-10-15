# ** COMP309 Project **
# Johniel Bocacao
# 300490028

###
# Load data
###

import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor

base_dir = "traindata"
train_list = []
y_train = []
for class_type in os.listdir(base_dir):
    if class_type == ".DS_Store": continue
    subdir = os.path.join(base_dir, class_type)
    for filename in os.listdir(subdir):
        if filename == ".DS_Store": continue
        filepath = os.path.join(subdir, filename)
        image = ToTensor()(Image.open(filepath)).unsqueeze(0)
        if list(image.size())[1:] == [3, 300, 300]:
            train_list.append(image)
            y_train.append(class_type)
        else:
            print(filepath, "is not size 300x300, image is:", image.size())

X_train = torch.Tensor((len(train_list), 3, 300, 300))
torch.cat(train_list, out=X_train)
print(X_train.size())
print(len(y_train))

###
# Preprocessing
###

X_train_proc = torch.zeros((len(train_list), 6, 300, 300))
red = X_train[:, 0, :, :]
green = X_train[:, 1, :, :]
blue = X_train[:, 2, :, :]
X_train_proc[:, 0:3, :, :] = X_train
X_train_proc[:, 3, :, :] = (red - green) / 2 + 0.5
X_train_proc[:, 4, :, :] = (red - blue) / 2 + 0.5
X_train_proc[:, 5, :, :] = (green - blue) / 2 + 0.5