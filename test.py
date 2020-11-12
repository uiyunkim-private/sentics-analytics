from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=(100,100,3)))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(10))

model.add(Activation('softmax'))
model.summary()