#Final output layer
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import Function_Libraries as FL
#测试以决策树作为最终输出层的结果
def test_dct():
    test_ds, class_names = FL.data_load("data_test", 224, 224, 16)
    tru_labels= FL.trulabels(test_ds)  
    lab_rf = FL.trulabels(test_ds)
    feature_output = FL.model_processing(test_ds)
    feature_output = pd.DataFrame(feature_output)
    fname = "models/dct.dot"
    dct = joblib.load(fname)
    pre_labels = dct.predict(feature_output)
    FL.score(tru_labels, pre_labels)
    pre_labels = pd.DataFrame(pre_labels)
if __name__ == '__main__':
    test_dct()

