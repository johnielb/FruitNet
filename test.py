import os
import tensorflow as tf
import numpy as np
from matplotlib.image import imread
from sklearn.preprocessing import LabelEncoder
import sys

if __name__ == "__main__":
    args = sys.argv
    if len(args) == 2:
        model_file = args[1]
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
                if image.ndim == 2:
                    image = np.stack((image,)*3, axis=-1)
                image = image[:, :, :3]/255.0
                if image.shape[0] != image.shape[1]:
                    image = tf.image.central_crop(image, central_fraction=1)
                image = tf.image.resize(image, [100, 100]).numpy()
                X_list.append(image)
                y.append(class_type)

        assert y.index('cherry') < y.index('strawberry') < y.index('tomato'), \
            "Class labels have not been loaded correctly," \
            "ensure that the cherry, strawberry and tomato " \
            "subdirectories are in alphabetical order."

        # Wrangle X, y datasets
        X = np.stack(X_list, axis=0)
        labeller = LabelEncoder().fit(y)
        y_proc = labeller.transform(y)

        model = tf.keras.models.load_model(model_file)
        model.evaluate(X, y_proc)

        predictions = np.argmax(model.predict(X), axis=1)
        cm = tf.math.confusion_matrix(y_proc, predictions)
    else:
        print("usage: test.py [model]")
        print("e.g. : test.py model.h5")
