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
def test_svm():
    test_ds, class_names = FL.data_load("data_test", 224, 224, 16)
    tru_labels= FL.trulabels(test_ds)
    feature_output = FL.model_processing(test_ds)
    print(feature_output)
    f1 = feature_output['f1']
    f2 = feature_output['f2']
    f3 = feature_output['f3']
    f4 = feature_output['f4']
    f5 = feature_output['f5']
    f6 = feature_output['f6']
    f7 = feature_output['f7']
    FL.score(tru_labels,f1)
    FL.score(tru_labels,f2)
    FL.score(tru_labels,f3)
    FL.score(tru_labels,f4)
    FL.score(tru_labels,f5)
    FL.score(tru_labels,f6)
    FL.score(tru_labels,f7)
if __name__ == '__main__':
    test_svm()