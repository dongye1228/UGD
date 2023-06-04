import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import Function_Libraries as FL
def data_go(path):
    images,labels = FL.load_img(path)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    train_y = y
    train_rgb = np.row_stack([FL.FE(img) for img in images])
    train_x = train_rgb
    return train_x,train_y
def rf_train(train_x,train_y):
    model_rf = RandomForestClassifier(n_estimators =1000, max_depth =7, random_state=2023) 
    model_rf.fit(train_x,train_y)
    joblib.dump(model_rf, 'model_comparison/models/rf.model')
def gbdt_train(train_x,train_y):
    model_gbdt = GradientBoostingClassifier(loss='log_loss', learning_rate=1, n_estimators=1000, subsample=1
                                  , min_samples_split=2, min_samples_leaf=1, max_depth=7
                                  , init=None, random_state=None, max_features=None
                                  , verbose=0, max_leaf_nodes=None, warm_start=False
                                  )
    model_gbdt.fit(train_x,train_y)
    joblib.dump(model_gbdt, 'model_comparison/models/gbdt.model')

if __name__ == '__main__':
    train_img_dir = 'model_comparison/traindata/'
    train_x,train_y = data_go(train_img_dir)
    rf_train(train_x,train_y)
    gbdt_train(train_x,train_y)



 