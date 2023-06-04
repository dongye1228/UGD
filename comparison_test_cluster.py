import tensorflow as tf
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import Function_Libraries as FL
def cluster_test():
    test_ds, class_names = FL.data_load("data", 224, 224,16)
    tru_labels = FL.trulabels(test_ds)
    #pre_ds,class_names = FL.data_load("model_comparison\cluster_test",224, 224, 16)
    pre_ds,class_names = FL.data_load("model_comparison/traindata",224, 224, 16)
    pre_labels = FL.trulabels(pre_ds)
    FL.score(tru_labels,pre_labels)
if __name__ == '__main__':
    cluster_test()
