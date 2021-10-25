import time
import numpy as np
import os
import requests
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from matplotlib.image import imread
from matplotlib import style
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, optimizers, regularizers

style.use("seaborn")

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
        image = imread(filepath)[:, :, :3]/255.0
        if image.shape[0] != image.shape[1]:
            image = tf.image.central_crop(image, central_fraction=1)
        image = tf.image.resize(image, [100, 100]).numpy()
        X.append(image)
        y.append(class_type)

# Wrangle X, y datasets
X = np.stack(X, axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, stratify=y)

labeller = LabelEncoder().fit(y_train)
y_train_proc = labeller.transform(y_train)
y_test_proc = labeller.transform(y_test)

# Cross-validation
kfold = StratifiedKFold(n_splits=3, shuffle=True)
cv = []
for train, val in kfold.split(X_train, y_train):
    cv.append((X_train[train], X_train[val],
               y_train_proc[train], y_train_proc[val]))


###
# Build models
###


def cross_validated_fit(model, fold, title, with_training=False, **kwargs):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    fig.suptitle(title)
    loss = []
    acc = []
    for i, (X1, X2, y1, y2) in enumerate(fold):
        print("*****     Fold {}     *****".format(i + 1))
        es = EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.01, verbose=1)
        history = model.fit(X1, y1, verbose=2, validation_data=(X2, y2),
                            callbacks=[es],
                            **kwargs)
        score = model.evaluate(X2, y2)
        ax[0].plot(history.history["val_loss"], label="Validation {}".format(i + 1))
        ax[1].plot(history.history["val_accuracy"], label="Validation {}".format(i + 1))
        if with_training:
            ax[0].plot(history.history["loss"], '--', label="Training {}".format(i + 1))
            ax[1].plot(history.history["accuracy"], '--', label="Training {}".format(i + 1))
        loss.append(score[0])
        acc.append(score[1])
    print("Training loss:", loss, " = ", np.mean(loss))
    print("Training acc:", acc, " = ", np.mean(acc))
    ax[0].set_title("Loss")
    ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("Epoch")
        box = a.get_position()
        a.set_position([box.x0, box.y0 + box.height * 0.03,
                        box.width, box.height * 0.97])

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.7, 0), ncol=3)


start = time.time()
mlp = models.Sequential()
mlp.add(layers.Flatten())
mlp.add(layers.Dense(256, activation="relu"))
mlp.add(layers.Dense(128, activation="relu"))
mlp.add(layers.Dense(64, activation="relu"))
mlp.add(layers.Dense(3))
mlp.compile(optimizer=optimizers.Adam(learning_rate=0.00001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
cross_validated_fit(mlp, cv, "MLP baseline", epochs=20, batch_size=50)
print(time.time()-start, "seconds to train MLP")
mlp.evaluate(X_test, y_test_proc)
mlp.save("mlp.h5")

# Tune minibatch size
for m in [5, 10, 50, 100, 500, len(cv[0][0])]:
    print("************ Minibatch = {} ************".format(m))
    cnn1 = models.Sequential()
    cnn1.add(layers.Conv2D(64, 5, activation='relu'))
    cnn1.add(layers.MaxPooling2D((2, 2)))
    cnn1.add(layers.Conv2D(64, 3, activation='relu'))
    cnn1.add(layers.MaxPooling2D((2, 2)))
    cnn1.add(layers.Flatten())
    cnn1.add(layers.Dense(256, activation='relu'))
    cnn1.add(layers.Dense(64, activation='relu'))
    cnn1.add(layers.Dense(3))
    cnn1.compile(optimizer=optimizers.Adam(learning_rate=0.00005),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    cross_validated_fit(cnn1, cv, "CNN, batch size {}".format(m), epochs=10, batch_size=m)
    cnn1.evaluate(X_test, y_test_proc)


###
# Test augmentation
###
def create_delta_channels(data):
    result = np.empty((data.shape[0], data.shape[1], data.shape[2], 3))
    result[:, :, :, 0:3] = data
    red = result[:, :, :, 0]
    green = result[:, :, :, 1]
    blue = result[:, :, :, 2]
    result[:, :, :, 1] = (red - green) / 2 + 0.5
    # result[:, :, :, 2] = (red - blue) / 2 + 0.5
    # result[:, :, :, 2] = (green - blue) / 2 + 0.5
    return result


X_train_aug = tf.concat([X_train,
                         tf.image.rot90(X_train, k=1),
                         tf.image.rot90(X_train, k=2),
                         tf.image.rot90(X_train, k=3),
                         tf.image.flip_up_down(X_train),
                         tf.image.flip_left_right(X_train),
                         ], axis=0).numpy()
y_train_aug = np.concatenate((y_train_proc, y_train_proc, y_train_proc, y_train_proc, y_train_proc, y_train_proc),
                             axis=0)
kfold_aug = StratifiedKFold(n_splits=3, shuffle=True)
cv_aug = []
for train, val in kfold_aug.split(X_train_aug, y_train_aug):
    cv_aug.append((X_train_aug[train], X_train_aug[val], y_train_aug[train], y_train_aug[val]))

for l in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
    print("************** Learning rate = {} **************".format(l))
    cnn_adam = models.Sequential()
    cnn_adam.add(layers.Conv2D(64, 3, activation='relu'))
    cnn_adam.add(layers.MaxPooling2D((2, 2)))
    cnn_adam.add(layers.Conv2D(64, 3, activation='relu'))
    cnn_adam.add(layers.MaxPooling2D((2, 2)))
    cnn_adam.add(layers.Conv2D(64, 3, activation='relu'))
    cnn_adam.add(layers.MaxPooling2D((2, 2)))
    cnn_adam.add(layers.Flatten())
    cnn_adam.add(layers.Dense(256, activation='relu'))
    cnn_adam.add(layers.Dense(64, activation='relu'))
    cnn_adam.add(layers.Dense(3))
    cnn_adam.compile(optimizer=optimizers.Adam(learning_rate=l),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
    cross_validated_fit(cnn_adam, cv, "CNN optimised by Adam, lr={}".format(l), with_training=True, epochs=20,
                        batch_size=50)
    cnn_adam.evaluate(X_test, y_test_proc)

for l in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
    print("************** Learning rate = {} **************".format(l))
    cnn_rp = models.Sequential()
    cnn_rp.add(layers.Conv2D(64, 3, activation='relu'))
    cnn_rp.add(layers.MaxPooling2D((2, 2)))
    cnn_rp.add(layers.Conv2D(64, 3, activation='relu'))
    cnn_rp.add(layers.MaxPooling2D((2, 2)))
    cnn_rp.add(layers.Conv2D(64, 3, activation='relu'))
    cnn_rp.add(layers.MaxPooling2D((2, 2)))
    cnn_rp.add(layers.Flatten())
    cnn_rp.add(layers.Dense(256, activation='relu'))
    cnn_rp.add(layers.Dense(64, activation='relu'))
    cnn_rp.add(layers.Dense(3))
    cnn_rp.compile(optimizer=optimizers.RMSprop(learning_rate=l),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    cross_validated_fit(cnn_rp, cv, "CNN optimised by RMSprop, lr={}".format(l), with_training=True, epochs=20,
                        batch_size=50)
    cnn_rp.evaluate(X_test, y_test_proc)

for l in [0.01, 0.05, 0.1]:
    for p in [0, 0.5, 0.9]:
        print("************** Learning rate = {}, momentum = {} **************".format(l, p))
        cnn_sgd = models.Sequential()
        cnn_sgd.add(layers.Conv2D(64, 3, activation='relu'))
        cnn_sgd.add(layers.MaxPooling2D((2, 2)))
        cnn_sgd.add(layers.Conv2D(64, 3, activation='relu'))
        cnn_sgd.add(layers.MaxPooling2D((2, 2)))
        cnn_sgd.add(layers.Conv2D(64, 3, activation='relu'))
        cnn_sgd.add(layers.MaxPooling2D((2, 2)))
        cnn_sgd.add(layers.Flatten())
        cnn_sgd.add(layers.Dense(256, activation='relu'))
        cnn_sgd.add(layers.Dense(64, activation='relu'))
        cnn_sgd.add(layers.Dense(3))
        cnn_sgd.compile(optimizer=optimizers.SGD(learning_rate=l, momentum=p),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
        cross_validated_fit(cnn_sgd, cv, "CNN optimised by SGD, lr={}, p={}".format(l, p), with_training=True,
                            epochs=20, batch_size=50)
        cnn_sgd.evaluate(X_test, y_test_proc)

cnn_adam1 = models.Sequential()
cnn_adam1.add(layers.Conv2D(64, 3, activation='relu'))
cnn_adam1.add(layers.MaxPooling2D((2, 2)))
cnn_adam1.add(layers.Conv2D(64, 3, activation='relu'))
cnn_adam1.add(layers.MaxPooling2D((2, 2)))
cnn_adam1.add(layers.Conv2D(64, 3, activation='relu'))
cnn_adam1.add(layers.MaxPooling2D((2, 2)))
cnn_adam1.add(layers.Flatten())
cnn_adam1.add(layers.Dense(256, activation='relu'))
cnn_adam1.add(layers.Dense(64, activation='relu'))
cnn_adam1.add(layers.Dense(3))
cnn_adam1.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
cross_validated_fit(cnn_adam1, cv_aug, "Final Adam CNN", epochs=10, batch_size=50)
cnn_adam1.evaluate(X_test, y_test_proc)
predictions1 = np.argmax(cnn_adam1.predict(X_test), axis=1)
cm1 = tf.math.confusion_matrix(y_test_proc, predictions1)

X_train_rg = tf.image.resize(create_delta_channels(X_train), [100, 100])
X_test_rg = tf.image.resize(create_delta_channels(X_test), [100, 100])
X_train_aug_rg = tf.concat([X_train_rg,
                            tf.image.rot90(X_train_rg, k=1),
                            tf.image.rot90(X_train_rg, k=2),
                            tf.image.rot90(X_train_rg, k=3)
                            ], axis=0).numpy()
y_train_aug = np.concatenate((y_train_proc, y_train_proc, y_train_proc, y_train_proc), axis=0)
kfold_aug = StratifiedKFold(n_splits=3, shuffle=True)
cv_aug_rg = []
for train, val in kfold_aug.split(X_train_aug_rg, y_train_aug):
    cv_aug_rg.append((X_train_aug_rg[train], X_train_aug_rg[val], y_train_aug[train], y_train_aug[val]))

cnn_adam2 = models.Sequential()
cnn_adam2.add(layers.Conv2D(64, 3, activation='relu'))
cnn_adam2.add(layers.MaxPooling2D((2, 2)))
cnn_adam2.add(layers.Conv2D(64, 3, activation='relu'))
cnn_adam2.add(layers.MaxPooling2D((2, 2)))
cnn_adam2.add(layers.Conv2D(64, 3, activation='relu'))
cnn_adam2.add(layers.MaxPooling2D((2, 2)))
cnn_adam2.add(layers.Flatten())
cnn_adam2.add(layers.Dense(256, activation='relu'))
cnn_adam2.add(layers.Dense(64, activation='relu'))
cnn_adam2.add(layers.Dense(3))
cnn_adam2.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
cross_validated_fit(cnn_adam2, cv_aug_rg, "Final Adam CNN with RG", epochs=10, batch_size=50)
cnn_adam2.evaluate(X_test_rg, y_test_proc)
predictions2 = np.argmax(cnn_adam2.predict(X_test_rg), axis=1)
cm2 = tf.math.confusion_matrix(y_test_proc, predictions2)

for l in [0, 1e-6, 1e-5, 1e-4]:
    for d in [0, 0.25, 0.5]:
        print("******** Regn: lambda={}, dropout={} *********".format(l, d))
        cnn_reg = models.Sequential()
        cnn_reg.add(layers.Conv2D(64, 3, activation='relu',
                                  kernel_regularizer=regularizers.l2(l)))
        cnn_reg.add(layers.MaxPooling2D((2, 2)))
        cnn_reg.add(layers.Dropout(d))
        cnn_reg.add(layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(l)))
        cnn_reg.add(layers.MaxPooling2D((2, 2)))
        cnn_reg.add(layers.Dropout(d))
        cnn_reg.add(layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(l)))
        cnn_reg.add(layers.MaxPooling2D((2, 2)))
        cnn_reg.add(layers.Flatten())
        cnn_reg.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l)))
        cnn_reg.add(layers.Dropout(d))
        cnn_reg.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l)))
        cnn_reg.add(layers.Dropout(d))
        cnn_reg.add(layers.Dense(3))
        cnn_reg.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
        cross_validated_fit(cnn_reg, cv_aug, "CNN with L2={}, dropout={}".format(l, d), epochs=20, batch_size=50)
        cnn_reg.evaluate(X_test, y_test_proc)

###
# Adjust layer size
###
for n in range(1, 6):
    print("**************** {} convolutional layers ****************".format(n))
    cnn_conv = models.Sequential()
    for i in range(0, n):
        cnn_conv.add(layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
        cnn_conv.add(layers.MaxPooling2D((2, 2)))
        cnn_conv.add(layers.Dropout(0.5))
    cnn_conv.add(layers.Flatten())
    cnn_conv.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
    cnn_conv.add(layers.Dropout(0.5))
    cnn_conv.add(layers.Dense(3))
    cnn_conv.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])
    cross_validated_fit(cnn_conv, cv_aug, "CNN, {} convolutional layers".format(n), epochs=20, batch_size=50)
    cnn_conv.evaluate(X_test, y_test_proc)

for k in [8, 16, 32, 64]:
    cnn_outsize = models.Sequential()
    for i in range(0, 3):
        cnn_outsize.add(layers.Conv2D(k, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
        cnn_outsize.add(layers.MaxPooling2D((2, 2)))
        cnn_outsize.add(layers.Dropout(0.5))
    cnn_outsize.add(layers.Flatten())
    cnn_outsize.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
    cnn_outsize.add(layers.Dropout(0.5))
    cnn_outsize.add(layers.Dense(3))
    cnn_outsize.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
    cross_validated_fit(cnn_outsize, cv_aug, "CNN, conv output channels={}".format(k), epochs=20, batch_size=50)
    cnn_outsize.evaluate(X_test, y_test_proc)

for fc in [(256,), (256, 64), (256, 128, 64), (256, 128, 64, 64), (256, 128, 64, 64, 64)]:
    cnn_fc = models.Sequential()
    for i in range(0, 3):
        cnn_fc.add(layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
        cnn_fc.add(layers.MaxPooling2D((2, 2)))
        cnn_fc.add(layers.Dropout(0.5))
    cnn_fc.add(layers.Flatten())
    for n in fc:
        cnn_fc.add(layers.Dense(n, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
        cnn_fc.add(layers.Dropout(0.5))
    cnn_fc.add(layers.Dense(3))
    cnn_fc.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    cross_validated_fit(cnn_fc, cv_aug, "CNN, fc layer={}".format(fc), epochs=20, batch_size=50)
    cnn_fc.evaluate(X_test, y_test_proc)

start_cnn = time.time()
cnn_final = models.Sequential()
for i in range(0, 3):
    cnn_final.add(layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
    cnn_final.add(layers.MaxPooling2D((2, 2)))
    cnn_final.add(layers.Dropout(0.5))
cnn_final.add(layers.Flatten())
for n in (256,):
    cnn_final.add(layers.Dense(n, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
    cnn_final.add(layers.Dropout(0.5))
cnn_final.add(layers.Dense(3))
cnn_final.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
cross_validated_fit(cnn_final, cv_aug, "Final CNN", epochs=20, batch_size=50)
print(time.time()-start_cnn, "seconds to train CNN")
cnn_final.evaluate(X_test, y_test_proc)
cnn_final.summary()
cnn_final.save("model.h5")

###
# Augment with Pixabay photos
###

blacklist_cherry = ["blossom", "berry", "strawberry", "tomato", "flower"]
blacklist_strawberry = ["cherry", "dessert", "ice cream", "tomato", "flower", "drink", "breakfast"]
blacklist_tomato = ["sauce", "vegetables", "strawberry", "cherry", "flower", "animal", "onion", "spaghetti", "egg",
                    "bread", "sandwich", "salad", "food", "soup", "bacon"]

cherry_urls = []
page = 1
while len(cherry_urls) <= 250:
    query = {"key": "13430298-902314550e850f2bc67d227f5", "q": "cherry",
             "image_type": "photo", "min_width": 100, "min_height": 100, "page": page, "per_page": 100}
    resp = requests.get("https://pixabay.com/api/", params=query).json()
    for hit in resp["hits"]:
        if any(word in hit["tags"] for word in blacklist_cherry): continue
        cherry_urls.append(hit["largeImageURL"])
    page += 1

strawberry_urls = []
page = 1
while len(strawberry_urls) <= 250:
    query = {"key": "13430298-902314550e850f2bc67d227f5", "q": "strawberry",
             "image_type": "photo", "min_width": 100, "min_height": 100, "page": page, "per_page": 100}
    resp = requests.get("https://pixabay.com/api/", params=query).json()
    for hit in resp["hits"]:
        if any(word in hit["tags"] for word in blacklist_strawberry): continue
        strawberry_urls.append(hit["largeImageURL"])
    page += 1

tomato_urls = []
page = 1
while len(tomato_urls) <= 250:
    query = {"key": "13430298-902314550e850f2bc67d227f5", "q": "tomato",
             "image_type": "photo", "min_width": 100, "min_height": 100, "page": page, "per_page": 100}
    resp = requests.get("https://pixabay.com/api/", params=query).json()
    for hit in resp["hits"]:
        if any(word in hit["tags"] for word in blacklist_tomato): continue
        tomato_urls.append(hit["largeImageURL"])
    page += 1

base_dir = "traindata"
for i, url in enumerate(cherry_urls):
    file_format = url[-3:]
    r = requests.get(url)
    open(os.path.join(base_dir, "cherry", "pixabay_{}.{}".format(i, file_format)), "wb").write(r.content)
for i, url in enumerate(strawberry_urls):
    file_format = url[-3:]
    r = requests.get(url)
    open(os.path.join(base_dir, "strawberry", "pixabay_{}.{}".format(i, file_format)), "wb").write(r.content)
for i, url in enumerate(tomato_urls):
    file_format = url[-3:]
    r = requests.get(url)
    open(os.path.join(base_dir, "tomato", "pixabay_{}.{}".format(i, file_format)), "wb").write(r.content)
