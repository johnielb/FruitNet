import numpy as np
import os
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
X_list = []
y = []
throwaway = []
for class_type in sorted(os.listdir(base_dir)):
    if class_type == ".DS_Store": continue
    subdir = os.path.join(base_dir, class_type)
    for filename in sorted(os.listdir(subdir)):
        if filename == ".DS_Store" or "pixabay" in filename: continue
        filepath = os.path.join(subdir, filename)
        image = imread(filepath)
        if image.shape == (300, 300, 3):
            X_list.append(image/255.0)
            y.append(class_type)
        else:
            print(filepath, "is not size 300x300, image is:", image.shape)
            throwaway.append(image)

# Wrangle X, y datasets
X = np.stack(X_list, axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, stratify=y)
X_train_sm = tf.image.resize(X_train, [100, 100])
X_test_sm = tf.image.resize(X_test, [100, 100])

labeller = LabelEncoder().fit(y_train)
y_train_proc = labeller.transform(y_train)
y_test_proc = labeller.transform(y_test)

# Cross-validation
kfold = StratifiedKFold(n_splits=3, shuffle=True)
cv = []
for train, val in kfold.split(X_train, y_train):
    cv.append((X_train_sm.numpy()[train], X_train_sm.numpy()[val],
               y_train_proc[train], y_train_proc[val]))

###
# Build models
###


def cross_validated_fit(model, fold, title, with_training=False, **kwargs):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)
    loss = []
    acc = []
    for i, (X1, X2, y1, y2) in enumerate(fold):
        print("*****     Fold {}     *****".format(i+1))
        es = EarlyStopping(monitor='val_accuracy', patience=3, min_delta=0.01, verbose=1)
        history = model.fit(X1, y1, verbose=2, validation_data=(X2, y2),
                            callbacks=[es],
                            **kwargs)
        score = model.evaluate(X2, y2)
        ax[0].plot(history.history["val_loss"], label="Validation {}".format(i+1))
        ax[1].plot(history.history["val_accuracy"], label="Validation {}".format(i+1))
        if with_training:
            ax[0].plot(history.history["loss"], '--', label="Training {}".format(i+1))
            ax[1].plot(history.history["accuracy"], '--', label="Training {}".format(i+1))
        loss.append(score[0])
        acc.append(score[1])
    print("Training loss:", loss, " = ", np.mean(loss))
    print("Training acc:", acc, " = ", np.mean(acc))
    ax[0].set_title("Loss")
    ax[1].set_title("Accuracy")
    for a in ax:
        a.set_xlabel("Epoch")
        # Shrink current axis's height by 10% on the bottom
        box = a.get_position()
        a.set_position([box.x0, box.y0 + box.height * 0.03,
                         box.width, box.height * 0.97])

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.7, 0), ncol=3)


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
mlp.evaluate(X_test_sm, y_test_proc)

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
    cnn1.evaluate(X_test_sm, y_test_proc)


###
# Test augmentation
###
def create_delta_channels(data):
    result = np.empty((data.shape[0],data.shape[1],data.shape[2],3))
    result[:, :, :, 0:3] = data
    red = result[:, :, :, 0]
    green = result[:, :, :, 1]
    blue = result[:, :, :, 2]
    result[:, :, :, 1] = (red - green) / 2 + 0.5
    # result[:, :, :, 2] = (red - blue) / 2 + 0.5
    # result[:, :, :, 2] = (green - blue) / 2 + 0.5
    return result


X_train_aug = tf.concat([X_train_sm,
                         tf.image.rot90(X_train_sm, k=1),
                         tf.image.rot90(X_train_sm, k=2),
                         tf.image.rot90(X_train_sm, k=3)
                         ], axis=0).numpy()
y_train_aug = np.concatenate((y_train_proc, y_train_proc, y_train_proc, y_train_proc), axis=0)
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
    cross_validated_fit(cnn_adam, cv, "CNN optimised by Adam, lr={}".format(l), with_training=True, epochs=20, batch_size=50)
    cnn_adam.evaluate(X_test_sm, y_test_proc)

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
    cross_validated_fit(cnn_rp, cv, "CNN optimised by RMSprop, lr={}".format(l), with_training=True, epochs=20, batch_size=50)
    cnn_rp.evaluate(X_test_sm, y_test_proc)

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
        cross_validated_fit(cnn_sgd, cv, "CNN optimised by SGD, lr={}, p={}".format(l, p), with_training=True, epochs=20, batch_size=50)
        cnn_sgd.evaluate(X_test_sm, y_test_proc)

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
cnn_adam1.evaluate(X_test_sm, y_test_proc)
predictions1 = np.argmax(cnn_adam1.predict(X_test_sm), axis=1)
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
        cnn_reg.evaluate(X_test_sm, y_test_proc)



###
# Adjust layer size
###
cnn_1layer = models.Sequential()
cnn_1layer.add(layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
cnn_1layer.add(layers.MaxPooling2D((2, 2)))
cnn_1layer.add(layers.Dropout(0.5))
cnn_1layer.add(layers.Flatten())
cnn_1layer.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
cnn_1layer.add(layers.Dropout(0.5))
cnn_1layer.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
cnn_1layer.add(layers.Dropout(0.5))
cnn_1layer.add(layers.Dense(3))
cnn_1layer.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
cross_validated_fit(cnn_1layer, cv_aug, "CNN, 1 layer", epochs=20, batch_size=50)
cnn_1layer.evaluate(X_test_sm, y_test_proc)

cnn_2layer = models.Sequential()
for i in range(0, 2):
    cnn_2layer.add(layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
    cnn_2layer.add(layers.MaxPooling2D((2, 2)))
    cnn_2layer.add(layers.Dropout(0.5))
cnn_2layer.add(layers.Flatten())
cnn_2layer.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
cnn_2layer.add(layers.Dropout(0.5))
cnn_2layer.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
cnn_2layer.add(layers.Dropout(0.5))
cnn_2layer.add(layers.Dense(3))
cnn_2layer.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
cross_validated_fit(cnn_2layer, cv_aug, "CNN, 2 layers", epochs=20, batch_size=50)
cnn_2layer.evaluate(X_test_sm, y_test_proc)

cnn_3layer = models.Sequential()
for i in range(0, 3):
    cnn_3layer.add(layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
    cnn_3layer.add(layers.MaxPooling2D((2, 2)))
    cnn_3layer.add(layers.Dropout(0.5))
cnn_3layer.add(layers.Flatten())
cnn_3layer.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
cnn_3layer.add(layers.Dropout(0.5))
cnn_3layer.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
cnn_3layer.add(layers.Dropout(0.5))
cnn_3layer.add(layers.Dense(3))
cnn_3layer.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
cross_validated_fit(cnn_3layer, cv_aug, "CNN, 3 layers", epochs=20, batch_size=50)
cnn_3layer.evaluate(X_test_sm, y_test_proc)

cnn_4layer = models.Sequential()
for i in range(0, 4):
    cnn_4layer.add(layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
    cnn_4layer.add(layers.MaxPooling2D((2, 2)))
    cnn_4layer.add(layers.Dropout(0.5))
cnn_4layer.add(layers.Flatten())
cnn_4layer.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
cnn_4layer.add(layers.Dropout(0.5))
cnn_4layer.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
cnn_4layer.add(layers.Dropout(0.5))
cnn_4layer.add(layers.Dense(3))
cnn_4layer.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
cross_validated_fit(cnn_4layer, cv_aug, "CNN, 4 layers", epochs=20, batch_size=50)
cnn_4layer.evaluate(X_test_sm, y_test_proc)

cnn_5layer = models.Sequential()
for i in range(0, 5):
    cnn_5layer.add(layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
    cnn_5layer.add(layers.MaxPooling2D((2, 2)))
    cnn_5layer.add(layers.Dropout(0.5))
cnn_5layer.add(layers.Flatten())
cnn_5layer.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
cnn_5layer.add(layers.Dropout(0.5))
cnn_5layer.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
cnn_5layer.add(layers.Dropout(0.5))
cnn_5layer.add(layers.Dense(3))
cnn_5layer.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
cross_validated_fit(cnn_5layer, cv_aug, "CNN, 5 layers", epochs=20, batch_size=50)
cnn_5layer.evaluate(X_test_sm, y_test_proc)

for k in [8, 16, 32, 64, 96]:
    cnn_kernel = models.Sequential()
    for i in range(0,3):
        cnn_kernel.add(layers.Conv2D(k, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
        cnn_kernel.add(layers.MaxPooling2D((2, 2)))
        cnn_kernel.add(layers.Dropout(0.5))
    cnn_kernel.add(layers.Flatten())
    cnn_kernel.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
    cnn_kernel.add(layers.Dropout(0.5))
    cnn_kernel.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5)))
    cnn_kernel.add(layers.Dropout(0.5))
    cnn_kernel.add(layers.Dense(3))
    cnn_kernel.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])
    cross_validated_fit(cnn_kernel, cv_aug, "CNN, kernel size={}".format(k), epochs=20, batch_size=50)
    cnn_kernel.evaluate(X_test_sm, y_test_proc)

for fc in [(256, ), (256, 64), (256, 128, 64), (512, 256, 128, 64), (512, 256, 128, 64, 32)]:
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
    cnn_fc.summary()
    cnn_fc.evaluate(X_test_sm, y_test_proc)


###
# Augment with Pixabay photos
###
X_list_pixabay = []
y_pixabay = []
for class_type in sorted(os.listdir(base_dir)):
    if class_type == ".DS_Store": continue
    subdir = os.path.join(base_dir, class_type)
    for filename in sorted(os.listdir(subdir)):
        if "pixabay" in filename:
            filepath = os.path.join(subdir, filename)
            image = imread(filepath)[:, :, :3]/255.0
            image_sm = tf.image.central_crop(image, central_fraction=1)
            X_list_pixabay.append(tf.image.resize(image_sm, [100, 100]).numpy())
            y_pixabay.append(class_type)

X_pixabay = np.stack(X_list_pixabay, axis=0)
y_pixabay_proc = labeller.transform(y_pixabay)