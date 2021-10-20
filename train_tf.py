import numpy as np
import os
from matplotlib.image import imread
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import datasets, layers, models, optimizers

base_dir = "traindata"
X = []
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
            X.append(image/255.0)
            y.append(class_type)
        else:
            print(filepath, "is not size 300x300, image is:", image.shape)
            throwaway.append(image)

X = np.stack(X, axis=0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

# X_train_200 = tf.image.resize(X_train, [200, 200])
# X_val_200 = tf.image.resize(X_val, [200, 200])

X_train_100 = tf.image.resize(X_train, [100, 100])
X_val_100 = tf.image.resize(X_val, [100, 100])

labeller = LabelEncoder().fit(y_train)
y_train_proc = labeller.transform(y_train)
y_val_proc = labeller.transform(y_val)

mlp = models.Sequential()
mlp.add(layers.Flatten())
mlp.add(layers.Dense(250))
mlp.add(layers.Dense(50))
mlp.add(layers.Dense(3))

mlp.compile(optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

history0 = mlp.fit(X_train_100, y_train_proc, epochs=20, batch_size=5, verbose=1,
                  validation_data=(X_val_100, y_val_proc))

cnn1 = models.Sequential()
cnn1.add(layers.Conv2D(24, 5, 1, activation='relu', input_shape=(100, 100, 3)))
cnn1.add(layers.MaxPooling2D((2, 2)))
cnn1.add(layers.Conv2D(4, 5, 1, activation='relu'))
cnn1.add(layers.MaxPooling2D((2, 2)))
cnn1.add(layers.Flatten())
cnn1.add(layers.Dense(64, activation='relu'))
cnn1.add(layers.Dense(3))

cnn1.compile(optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

history1 = cnn1.fit(X_train_100, y_train_proc, epochs=20, batch_size=5, verbose=1,
                    validation_data=(X_val_100, y_val_proc))
