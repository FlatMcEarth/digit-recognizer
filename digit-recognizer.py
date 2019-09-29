import numpy as np
import pandas as pd
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
from keras import optimizers


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras import regularizers
from keras import callbacks
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

data = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

labels = to_categorical(data.iloc[:,0].values)
inputs = data.iloc[:,1:].values.reshape(-1,28,28,1)

x_train,x_test,y_train,y_test = train_test_split(inputs,labels)

model = models.Sequential()

model.add(layers.Conv2D(128,(7,7), activation='relu', input_shape=(28,28,1),padding='same'))
model.add(layers.Dropout(0.25))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128,(7,7), activation='relu',padding='same'))
model.add(layers.Dropout(0.25))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128,(5,5), activation='relu',padding='same'))
model.add(layers.Dropout(0.25))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128,(5,5), activation='relu',padding='same'))
model.add(layers.Dropout(0.25))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64,(3,3), activation='relu',padding='same'))
model.add(layers.Dropout(0.25))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64,(3,3), activation='relu',padding='same'))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())

model.add(layers.BatchNormalization())
model.add(layers.Dense(256,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.BatchNormalization())
model.add(layers.Dense(128,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.BatchNormalization())
model.add(layers.Dense(32,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(10,activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = optimizers.RMSprop(lr=1e-4),
              metrics = ['acc'])

checkpoint = callbacks.ModelCheckpoint(filepath='best_model.h5',monitor='val_loss',save_best_only=True)
earlystop = callbacks.EarlyStopping(monitor='val_loss',patience=5)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2
)
train_generator = train_datagen.flow(x_train,y_train,batch_size=128)

test_datagen = ImageDataGenerator(rescale=1./255)
val_generator = test_datagen.flow(x_test,y_test,batch_size=128)

history = model.fit_generator(
    train_generator,
    epochs=25,
    steps_per_epoch=x_train.shape[0]/64,
    callbacks=[checkpoint,earlystop],
    validation_data=val_generator,
    validation_steps=x_test.shape[0]/64,
)

model.load_weights('best_model.h5')

test = test.iloc[:,1:].values
test = test.reshape(-1,28,28,1)/255


result = model.predict(test)
out = np.argmax(result,axis=1) 


tag = np.array([i for i in range(len(out))])
final = np.concatenate((tag.reshape(1,-1),out.reshape(1,-1)),axis=0)
final = pd.DataFrame({'id':tag,'label':out})
final.to_csv('result.csv',index=False)
