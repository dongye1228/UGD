import tensorflow as tf
import pandas as pd
import numpy as np
import Function_Libraries as FL
def test_cnn():
    test_ds, class_names = FL.data_load("data_test", 224, 224, 16)
    model = tf.keras.models.load_model("model_comparison\models\cnn_covid.h5")
    tru_labels = FL.trulabels(test_ds)
    pre_labels = model.predict(test_ds)
    pre_labels = np.argmax(pre_labels, axis=1)
    pre_labels = list(pre_labels)
    FL.score(tru_labels, pre_labels)
if __name__ == '__main__':
    test_cnn()
