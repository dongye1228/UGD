#Enhanced classification layer
from sklearn import svm
from sklearn import preprocessing
from sklearn.utils import shuffle
import pandas as pd
import joblib
def train_multiple_svm(X,y):  
    X = X.copy()
    y = y.copy()
    X,y = shuffle(X,y, random_state=2023) 
    l1 = int(len(X)/2) 
    l2 = int(len(X))  
    X1 = X.iloc[0:l1,:]
    y1 = y.iloc[0:l1,:]
    X2 = X.iloc[l1:l2,:]
    y2 = y.iloc[l1:l2,:]
    y1 = y1.values.ravel()
    y2 = y2.values.ravel()
    X1 = preprocessing.scale(X1)
    X2 = preprocessing.scale(X2)
    svm1 = svm.SVC(C=0.8,kernel='rbf').fit(X1,y1)
    svm2 = svm.SVC(C=0.8,kernel='linear').fit(X1,y1)
    svm3 = svm.SVC(C=5,kernel='poly').fit(X1,y1)
    svm4 = svm.SVC(C=0.8,kernel='rbf').fit(X2,y2)
    svm5 = svm.SVC(C=0.8,kernel='linear').fit(X2,y2)
    svm6 = svm.SVC(C=5,kernel='poly').fit(X2,y2)
    joblib.dump(svm1, "models/svm_1.h5")
    joblib.dump(svm2, "models/svm_2.h5")
    joblib.dump(svm3, "models/svm_3.h5")
    joblib.dump(svm4, "models/svm_4.h5")
    joblib.dump(svm5, "models/svm_5.h5")
    joblib.dump(svm6, "models/svm_6.h5")
    
if __name__ == '__main__':
    X =pd.read_csv('features_cnn.csv')
    y =pd.read_csv('tru_labels.csv')
    train_multiple_svm(X,y)