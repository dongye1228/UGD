#Depth information extraction layer
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import *
import glob
import random 
from sklearn.cluster import DBSCAN
import Function_Libraries as FL
def data_load_tv(data_dir, test_data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=2023,
        image_size=(img_height, img_width),
        shuffle=False,
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=2023,
        image_size=(img_height, img_width),
        shuffle=False,
        batch_size=batch_size)
    class_names = train_ds.class_names
    return train_ds, val_ds, class_names

# 构建CNN模型
def model_load(IMG_SHAPE=(224, 224, 3), class_num=12):
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # 将二维的输出转化为一维
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
def show_loss_acc(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('cnn_train.png', dpi=100)

def train(epochs):
    begin_time = time()
    train_ds, val_ds, class_names = data_load_tv("data_train/train","data_train/val", 224, 224, 8)
    model = model_load(class_num=len(class_names))
    history = model.fit(train_ds, validation_data=train_ds, epochs=epochs)   
    model.save("models/cnn_covid.h5")
    model.summary()
    #获取该模型倒数第二层的输出(特征)
    sub_model = tf.keras.models.Model(inputs = model.input,outputs = model.layers[-2].output)
    train_all_ds, class_names = FL.data_load("data_c2",224,224,16)
    features_cnn = sub_model.predict(train_all_ds) 
    features_cnn=pd.DataFrame(data=features_cnn)
    features_cnn.to_csv('features_cnn.csv', index=False, sep=',')
    tru_labels = FL.trulabels(train_all_ds)
    tru_labels = pd.DataFrame(tru_labels)
    tru_labels.to_csv('tru_labels.csv', index=False, sep=',')
    model.save("models/cnn_covid.h5")
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s") 
    show_loss_acc(history)
    model.summary()

if __name__ == '__main__':
    train(epochs=30)
