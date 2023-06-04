import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import Function_Libraries as FL
import rf_gbdt_train as rg

if __name__ == '__main__':
    test_img_dir = 'model_comparison/testdata/'
    test_x,test_y = rg.data_go(test_img_dir)
    rf = joblib.load('model_comparison/models/rf.model')
    pre_rf = rf.predict(test_x)
    tru_labels = test_y
    FL.score(pre_rf,tru_labels)



