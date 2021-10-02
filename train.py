# ** COMP309 Project **
# Johniel Bocacao
# 300490028

###
# Step 0: Load data
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
# Step 1: Conduct exploratory data analysis
###
