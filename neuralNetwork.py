
import pandas as pd
import numpy as np
import re
import sklearn
import warnings

from sklearn.model_selection import train_test_split
from tensorflow import keras  

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from keras import optimizers

np.random.seed(0)

# Data Collection

df = pd.read_csv("500f.csv")
labels = df['Inhibitor'].to_numpy()
data = pd.read_csv("500f.csv", usecols=range(1,269)).values

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=10)

Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Sequential = keras.Sequential

warnings.filterwarnings('ignore')

model = Sequential()
model.add(Dense(256, input_shape=(268,), activation='relu'))
model.add(Dropout(0.145))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4726))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.288))


model.add(Dense(1, activation='sigmoid'))



from keras import optimizers



# Compile and Train Model
class_weight = {0: 72.,
1: 500.}

learning_rate = 0.000338288
opt = keras.optimizers.Adam(lr=learning_rate)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(X_train, Y_train, 
validation_split = 0.2, 
epochs = 100, 
batch_size = 8,
class_weight=class_weight)
# Evaluate the model's performance
train_loss, train_acc = model.evaluate(X_train, Y_train)
test_loss, test_acc = model.evaluate(X_test, Y_test)

print('Training set accuracy:', train_acc)
print('Test set accuracy:', test_acc)

