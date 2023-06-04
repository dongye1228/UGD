#Final output layer
import tensorflow as tf
import pandas as pd
import numpy as np
from itertools import chain
import joblib
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import Function_Libraries as FL
#测试以直接投票法作为最终输出层的结果
def test_vode():
    test_ds, class_names = FL.data_load("data_test", 224, 224, 16)
    tru_labels= FL.trulabels(test_ds)
    feature_output = FL.model_processing(test_ds)
    a = feature_output.mode(axis='columns').values
    a = list(chain.from_iterable(a))
    FL.score(tru_labels,a)
if __name__ == '__main__':
    test_vode()
    

