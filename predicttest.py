import pandas as pd
import numpy as np

df = pd.read_csv("500f.csv", nrows = 0)
del df['Unnamed: 0']
del df['Inhibitor']
df2 = pd.read_csv("last8.csv")
outputdf = pd.DataFrame()

print(df2)

for i in df:
    name = str(i)
    outputdf[name] = df2[i].to_numpy()

#Normalize
from sklearn import preprocessing

x = outputdf.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
outputdf = pd.DataFrame(x_scaled)

#print(outputdf)

# Imports

import re
import sklearn
import warnings

from sklearn.model_selection import train_test_split
from tensorflow import keras  

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from keras import optimizers

from keras.layers import LeakyReLU

np.random.seed(0)

# Data Collection

df = pd.read_csv("500f.csv")
labels = df['Inhibitor'].to_numpy()
data = pd.read_csv("500f.csv", usecols=range(1,269)).values

outputdf.to_csv('output.csv')

df2 = pd.read_csv("output.csv")
testData = pd.read_csv("output.csv", usecols=range(1,269)).values

#print(labels)

outputs = []

for i in range(10):
    #Split Data
    X_train = data
    Y_train = labels
    X_test = testData
    
    #print(X_test)
    
    
    #Compile Model
    Dense = keras.layers.Dense
    Dropout = keras.layers.Dropout
    Sequential = keras.Sequential
    
    warnings.filterwarnings('ignore')
    
    model = Sequential()
    model.add(Dense(64, input_shape=(268,), activation=LeakyReLU(alpha=0.1)))
    model.add(Dropout(0.561424))
    model.add(Dense(64, activation=LeakyReLU(alpha=0.1)))
    model.add(Dropout(0.562382))
    model.add(Dense(32, activation=LeakyReLU(alpha=0.1)))
    model.add(Dropout(0.472685))
    model.add(Dense(1, activation='sigmoid'))
    
    #print(i)
      
    from keras import optimizers
    
    # Compile and Train Model
    class_weight = {0: 72.,1: 500.}
    
    learning_rate = 0.000338288
    opt = keras.optimizers.Adam(lr=learning_rate)
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(X_train, Y_train, 
                        validation_split = 0.2, 
                        epochs = 40, 
                        batch_size = 8,
                        class_weight=class_weight) #40
    
    train_loss, train_acc = model.evaluate(X_train, Y_train)
    
    print('Training set accuracy:', train_acc)
    
    
    
    novelCompounds = 0
    y_pred = model.predict(X_test, batch_size=64, verbose=1)
    outputs.append(y_pred) 
    print("-----------------------")
print("========================outputs==========================") 
outputs = np.array(outputs)
print(outputs.shape)

print("*******************")
print(outputs[0])
for i in outputs[0]:
    print(i) 
print("-------------")
stdDev = np.std(outputs, axis=0)
print("STD = ")
print(stdDev)
print(stdDev.shape)

np.savetxt('stdDev.csv', stdDev, delimiter=',')

mean = np.mean(stdDev, axis=0)
print(mean)
print(novelCompounds)
