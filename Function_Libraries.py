from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing
import os
import glob
import cv2
from skimage import io,transform

#模型评价
def score(y_true, y_pred):
    print('accuracy :',accuracy_score(y_true, y_pred))
    print('precision :',precision_score(y_true, y_pred, average='micro'))
    print('recall :',recall_score(y_true, y_pred, average='micro'))
    print('F1 score :',f1_score(y_true, y_pred, average='micro'))
#数据载入并分片
def data_load(test_data_dir, img_height, img_width, batch_size):
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=2023,
        image_size=(img_height, img_width),
        shuffle=False,
        batch_size=batch_size)
    class_names = val_ds.class_names
    return val_ds, class_names
#返还对应真实标签或伪标签
def trulabels(test_ds):
    tru_labels = []
    k = 0
    for batch in test_ds:    
        k += 1  
    for images, labels in test_ds.take(k):
     m = labels.get_shape()
     n = m[0]
     p = m[1]    
     u = np.array(labels)
     for i in range(n):
        q = u[i]
        for j in range(p):
            if q[j] == 1:
                tru_labels.append(j)
    return tru_labels
#载入模型和数据，将中间处理层的结果返还
def model_processing(input_ds):
    svm1 = joblib.load("models/svm_1.h5")
    svm2 = joblib.load("models/svm_2.h5")
    svm3 = joblib.load("models/svm_3.h5")
    svm4 = joblib.load("models/svm_4.h5")
    svm5 = joblib.load("models/svm_5.h5")
    svm6 = joblib.load("models/svm_6.h5")
    cnn_dense = tf.keras.models.load_model("models/cnn_covid.h5")
    sub_model = tf.keras.models.Model(inputs = cnn_dense.input,outputs = cnn_dense.layers[-2].output)
    tra_dense = sub_model.predict(input_ds)
    tra_dense = tra_dense.astype(float)
    tra_dense = preprocessing.scale(tra_dense)
    pre_labels_1 = list(svm1.predict(tra_dense))
    pre_labels_2 = list(svm2.predict(tra_dense))
    pre_labels_3 = list(svm3.predict(tra_dense))
    pre_labels_4 = list(svm4.predict(tra_dense))
    pre_labels_5 = list(svm5.predict(tra_dense))
    pre_labels_6 = list(svm6.predict(tra_dense))
    pre_labels_7 = cnn_dense.predict(input_ds,verbose=1)
    pre_labels_7 = np.argmax(pre_labels_7, axis=1)
    pre_labels_7 = list(pre_labels_7)
    feature_output = {"f1" : pre_labels_1,
    "f2" : pre_labels_2,
    "f3" : pre_labels_3,
    "f4" : pre_labels_4,
    "f5" : pre_labels_5,
    "f6" : pre_labels_6,
    "f7" : pre_labels_7}
    feature_output = pd.DataFrame(feature_output)
    return feature_output
#直接载入图片
def load_img(path):
    box = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs = []
    labels = []
    for idx,folder in enumerate(box):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img = cv2.imread(im,cv2.IMREAD_COLOR)
            imgs.append(img)
            labels.append(idx)
    return imgs,labels
#(模型对比)直方图 特征提取与转化 Feature extraction
def FE(img):
    hist = cv2.calcHist([img],[0,1,2],None,[1,1,1],[0,128,0,128,0,128])
    return hist.ravel()

