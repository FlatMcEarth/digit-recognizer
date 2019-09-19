import numpy as np
import pandas as pd
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
from keras import optimizers


data = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')

labels = data.iloc[:,0].values
inputs = data.iloc[:,1:].values

x_train,x_test,y_train,y_test = train_test_split(inputs,labels)
x_train = x_train.reshape(-1,28,28,1)/255
x_test = x_test.reshape(-1,28,28,1)/255
y_tra = np.zeros([len(y_train),10])
y_tes = np.zeros([len(y_test),10])

for i,lab in enumerate(y_tra):
    lab[y_train[i]]=1

for i,lab in enumerate(y_tes):
    lab[y_test[i]]=1

y_train = y_tra
y_test = y_tes

model = models.Sequential()
model.add(layers.Conv2D(64,(5,5), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(96,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                                    beta_initializer='zeros', gamma_initializer='ones',
                                    moving_mean_initializer='zeros', moving_variance_initializer='ones'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                                    beta_initializer='zeros', gamma_initializer='ones',
                                    moving_mean_initializer='zeros', moving_variance_initializer='ones'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, 
                                    beta_initializer='zeros', gamma_initializer='ones',
                                    moving_mean_initializer='zeros', moving_variance_initializer='ones'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = optimizers.RMSprop(lr=2e-4),
              metrics = ['acc'])

model.fit(x_train,y_train,epochs=35,batch_size=512,validation_data=[x_test,y_test])

test = test.values

test = test.reshape(-1,28,28,1)/255

result = model.predict(test)
out = np.argmax(result,axis=1) 


tag = np.array([i+1 for i in range(len(out))])
final = np.concatenate((tag.reshape(1,-1),out.reshape(1,-1)),axis=0)
final = pd.DataFrame({'ImageId':tag,'Label':out})
final.to_csv('result.csv',index=False)
