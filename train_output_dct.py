#Final output layer
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import Function_Libraries as FL
def train_model_dct():
    train_ds, class_names = FL.data_load("data_c2", 224, 224, 16)
    feature_train = FL.model_processing(train_ds)
    tru_labels = FL.trulabels(train_ds)
    tru_labels = pd.DataFrame(tru_labels)
    dct = tree.DecisionTreeClassifier(criterion='gini',max_depth=7,random_state=2023)
    dct.fit(feature_train, tru_labels)
    fname = "models/dct.dot"
    joblib.dump(dct, fname)

if __name__ == '__main__':
    train_model_dct()

