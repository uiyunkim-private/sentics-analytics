import scipy.io
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
list_data = []
for d in range(1,21):
    day = 'day' + str(d)
    mat = scipy.io.loadmat('set_a/' + day+'.mat')

    data = mat[day]

    data = data.reshape((32,2001))


    for i in range(1,len(data)+1):

        if i % 8 == 1:
            t1 = 'No Emotion'
        elif i % 8 == 2:
            t1 = 'Anger'
        elif i % 8 == 3:
            t1 = 'Hate'
        elif i % 8 == 4:
            t1 = 'Grief'
        elif i % 8 == 5:
            t1 = 'P-Love'
        elif i % 8 == 6:
            t1 = 'R-Love'
        elif i % 8 == 7:
            t1 = 'Joy'
        else:
            t1 = 'Reverence'

        if i / 8 < 1.01:
            t2 = 'EMG(jaw)'
        elif i / 8 < 2.01:
            t2 = 'BVP'
        elif i / 8 < 3.01:
            t2 = 'GSR(palm)'
        else:
            t2 = 'Respiration'
        tmp_dict = {}
        tmp_dict.update({'emotion':t1,
                          'day':day,
                          'sensor':t2,
                          'data':data[i-1]})
        # for i,d in enumerate(data[0]):
        #     tmp_dict.update({'data'+str(i):d})
        list_data.append(tmp_dict)

table = []
for day in range(1,21):
    data_day = [x for x in list_data if x['day'] == 'day' + str(day)]


    for emotion in ['Anger','No Emotion','Grief','Hate','P-Love','R-Love','Joy','Reverence']:
        emotion_table = {}
        emotion_day = [x for x in data_day if x['emotion'] == emotion]

        emotion_table.update({'emotion':emotion ,
                              'EMG(jaw)':np.array([x['data'] for x in emotion_day if x['sensor']=='EMG(jaw)'][0]),
                              'BVP': np.array([x['data'] for x in emotion_day if x['sensor'] == 'BVP'][0]),
                              'GSR(palm)': np.array([x['data'] for x in emotion_day if x['sensor'] == 'GSR(palm)'][0]),
                              'Respiration': np.array([x['data'] for x in emotion_day if x['sensor'] == 'Respiration'][0])
                              })

        table.append(emotion_table)


# df = pd.DataFrame(table)
# df.to_csv('test.csv')
#
# label = df[['emotion']]
# data = df[['EMG(jaw)','BVP','GSR(palm)','Respiration']]

#
# label = np.array(label)
# data = np.array(data)
#
# print(label.shape)
# print(data.shape)

emotion_list = ['Anger','No Emotion','Grief','Hate','P-Love','R-Love','Joy','Reverence']
label = []
for i in range(len(table)):
    label.append(emotion_list.index(table[i]['emotion']) )

data = []
for i in range(len(table)):
    sub_data = []

    for emotion in ['EMG(jaw)','BVP','GSR(palm)','Respiration']:
        sub_data.append(table[i][emotion])

    data.append(sub_data)

label = np.array(label)


data = np.array(data)

label = tf.keras.utils.to_categorical(label,num_classes=8)
print(label)
print(label.shape)
print(data.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256,input_shape=(4,2001)))
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(8,activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0000005)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x=data,y=label,epochs=100,validation_split=0.1,batch_size=8)

def plot_train_and_val(history):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'Validation'], loc='upper left')
  plt.show()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'Validation'], loc='upper left')
  plt.show()

plot_train_and_val(history)