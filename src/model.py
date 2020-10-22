import tensorflow as tf

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
    model.add(tf.keras.layers.Reshape(target_shape=(input_shape[0], input_shape[1] * input_shape[2]) , input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(8, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model