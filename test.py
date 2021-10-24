import os
import tensorflow as tf
import numpy as np
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

base_dir = "testdata"
X_list = []
y = []
throwaway = []

for class_type in sorted(os.listdir(base_dir)):
    if class_type == ".DS_Store": continue
    subdir = os.path.join(base_dir, class_type)
    for filename in sorted(os.listdir(subdir)):
        if filename == ".DS_Store": continue
        filepath = os.path.join(subdir, filename)
        image = imread(filepath)
        if image.shape == (300, 300, 3):
            X_list.append(image/255.0)
            y.append(class_type)
        else:
            print(filepath, "is not size 300x300, image is:", image.shape)
            throwaway.append(image)

assert y.index('cherry') < y.index('strawberry') < y.index('tomato'), "Class labels have not been loaded correctly," \
                                                                      "ensure that the cherry, strawberry and tomato " \
                                                                      "subdirectories are in alphabetical order."
# Wrangle X, y datasets
X = np.stack(X_list, axis=0)
X_sm = tf.image.resize(X, [100, 100])
labeller = LabelEncoder().fit(y)
y_proc = labeller.transform(y)
