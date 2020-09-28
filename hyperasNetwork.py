from __future__ import print_function
import pandas as pd
import numpy as np

#from hyperopt.hp import uniform


# Imports

import re
import sklearn
import warnings

from sklearn.model_selection import train_test_split
from tensorflow import keras  

from keras import optimizers


from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim

from keras.layers import LeakyReLU

from math import log


np.random.seed(0)



#print(i)

def data():
    df = pd.read_csv("500f.csv")
    labels = df['Inhibitor'].to_numpy()
    data = pd.read_csv("500f.csv", usecols=range(1,269)).values

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=10)
    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    Dense = keras.layers.Dense
    Dropout = keras.layers.Dropout
    Sequential = keras.Sequential

    warnings.filterwarnings('ignore')

    model = Sequential()
    model.add(Dense({{choice([2**i for i in range(11)])}}, activation=LeakyReLU(alpha=0.1), input_shape=(268,)))
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense({{choice([2**i for i in range(11)])}}, activation=LeakyReLU(alpha=0.1)))
    model.add(Dropout({{uniform(0,1)}}))

    model.add(Dense({{choice([2**i for i in range(11)])}}, activation=LeakyReLU(alpha=0.1)))
    model.add(Dropout({{uniform(0,1)}}))
       
    model.add(Dense(1, activation='sigmoid'))


    # Compile and Train Model
    class_weight = {0: 72.,
                    1: 500.}

    learning_rate = {{uniform(0.000001,0.1)}}
    opt = keras.optimizers.Adam(lr=learning_rate)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    result = model.fit(x_train, y_train, 
                        validation_split = 0.2, 
                        epochs = {{choice([10,20,30,40,50,60,70,80,90,100])}}, 
                        batch_size = 8,
                        class_weight=class_weight)
    # Evaluate the model's performance
    train_loss, train_acc = model.evaluate(x_train, y_train)
    test_loss, test_acc = model.evaluate(x_test, y_test)

    print('Training set accuracy:', train_acc)
    print('Test set accuracy:', test_acc)

    validation_acc = np.amax(result.history['val_loss']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': validation_acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=1000,
                                          eval_space=True,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
