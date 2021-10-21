import numpy as np
import os
from matplotlib import pyplot as plt
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

X_train_sm = tf.image.resize(X_train, [150, 150])
X_val_sm = tf.image.resize(X_val, [150, 150])

labeller = LabelEncoder().fit(y_train)
y_train_proc = labeller.transform(y_train)
y_val_proc = labeller.transform(y_val)

###
# Build models
###

mlp = models.Sequential()
mlp.add(layers.Flatten())
mlp.add(layers.Dense(256, activation="relu"))
mlp.add(layers.Dense(128, activation="relu"))
mlp.add(layers.Dense(3))

mlp.compile(optimizer=optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

history0 = mlp.fit(X_train_sm, y_train_proc, epochs=20, batch_size=10, verbose=1,
                   validation_data=(X_val_sm, y_val_proc))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(history0.history["loss"])
ax[0].plot(history0.history["val_loss"])
ax[0].set_title("Training and validation (orange) loss")
ax[1].plot(history0.history["accuracy"])
ax[1].plot(history0.history["val_accuracy"])
ax[1].set_title("Training and validation (orange) accuracy")

cnn1 = models.Sequential()
cnn1.add(layers.Conv2D(64, 5, activation='relu'))
cnn1.add(layers.MaxPooling2D((2, 2)))
cnn1.add(layers.Conv2D(64, 3, activation='relu'))
cnn1.add(layers.MaxPooling2D((2, 2)))
cnn1.add(layers.Conv2D(16, 3, activation='relu'))
cnn1.add(layers.MaxPooling2D((2, 2)))
cnn1.add(layers.Flatten())
cnn1.add(layers.Dense(256, activation='relu'))
cnn1.add(layers.Dense(64, activation='relu'))
cnn1.add(layers.Dense(3))
cnn1.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
history1 = cnn1.fit(X_train_sm, y_train_proc, epochs=25, batch_size=10, verbose=1,
                    validation_data=(X_val_sm, y_val_proc))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(history1.history["loss"])
ax[0].plot(history1.history["val_loss"])
ax[0].set_title("Training and validation (orange) loss")
ax[1].plot(history1.history["accuracy"])
ax[1].plot(history1.history["val_accuracy"])
ax[1].set_title("Training and validation (orange) accuracy")

X_train_half = tf.image.random_crop(X_train, [len(X_train), 150, 150, 3])
X_train_aug = tf.concat([X_train_sm,
                         tf.cast(X_train_half, tf.float32)
                         ], axis=0)
y_train_aug = np.concatenate((y_train_proc,y_train_proc), axis=0)

cnn2 = models.Sequential()
cnn2.add(layers.Conv2D(64, 5, activation='relu'))
cnn2.add(layers.MaxPooling2D((2, 2)))
cnn2.add(layers.Conv2D(64, 3, activation='relu'))
cnn2.add(layers.MaxPooling2D((2, 2)))
cnn2.add(layers.Conv2D(16, 3, activation='relu'))
cnn2.add(layers.MaxPooling2D((2, 2)))
cnn2.add(layers.Flatten())
cnn2.add(layers.Dense(256, activation='relu'))
cnn2.add(layers.Dense(64, activation='relu'))
cnn2.add(layers.Dense(3))
cnn2.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
history2 = cnn2.fit(X_train_aug, y_train_aug, epochs=25, batch_size=10, verbose=1,
                    validation_data=(X_val_sm, y_val_proc))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(history2.history["loss"])
ax[0].plot(history2.history["val_loss"])
ax[0].set_title("Training and validation (orange) loss")
ax[1].plot(history2.history["accuracy"])
ax[1].plot(history2.history["val_accuracy"])
ax[1].set_title("Training and validation (orange) accuracy")
