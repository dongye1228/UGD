import numpy as np
from sklearn.preprocessing import LabelEncoder
import Function_Libraries as FL
import joblib
import rf_gbdt_train as rg

if __name__ == '__main__':
    test_img_dir = 'model_comparison/testdata/'
    test_x,test_y = rg.data_go(test_img_dir)
    gbdt = joblib.load('model_comparison/models/gbdt.model')
    pre_gbdt = gbdt.predict(test_x)
    tru_labels = test_y
    FL.score(pre_gbdt,tru_labels)
