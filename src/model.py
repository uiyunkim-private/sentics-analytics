import tensorflow as tf
from tensorflow.keras.utils import plot_model
def tf_dense(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, input_shape=input_shape))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(8, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def tf_lstm(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Permute((2, 1), input_shape=input_shape))

    model.add(tf.keras.layers.Conv1D(256,kernel_size=7,activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(3))
    model.add(tf.keras.layers.Conv1D(256,kernel_size=7,activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(3))

    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.LSTM(256,activation='relu'))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(8, activation='softmax'))
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model