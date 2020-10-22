import scipy.io
import pandas as pd
import numpy as np

from src.configuration import ROOT_DIR

def window_sliding(data,window_size=10,stride=2):
    windows = []
    for x in data:
        instance = x.copy()
        instance = instance.transpose()
        window = []
        for i in range(0,len(instance)-window_size,stride):
            window.append(instance[i:i+window_size])
        windows.append(window)
    windows = np.array(windows)
    return windows

def load_dataset(dtype='np'):
    '''

    dtype: 'np' or 'df'

    if 'np': return data, label
    if 'df': return df

    '''
    list_data = []
    for d in range(1,21):
        day = 'day' + str(d)
        mat = scipy.io.loadmat(ROOT_DIR / 'dataset'/ 'set_a' / (day+'.mat'))
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

    emotion_list = ['Anger', 'No Emotion', 'Grief', 'Hate', 'P-Love', 'R-Love', 'Joy', 'Reverence']
    label = []
    for i in range(len(table)):
        label.append(emotion_list.index(table[i]['emotion']))

    data = []
    for i in range(len(table)):
        sub_data = []
        for emotion in ['EMG(jaw)', 'BVP', 'GSR(palm)', 'Respiration']:
            sub_data.append(table[i][emotion])
        data.append(sub_data)

    label = np.array(label)
    data = np.array(data)
    count = 0
    if dtype == 'df':
        mesurement = ['EMG(jaw)', 'BVP', 'GSR(palm)', 'Respiration']

        collection = []
        for x in data:
            instance = []
            x = [(x[i],mesurement[i]) for i in range(4)]
            for y in x:
                d = y[0].reshape(1,y[0].shape[0])
                columns = [y[1]+'_'+str(x) for x in range(len(y[0]))]
                instance.append(pd.DataFrame(d,columns=columns,index=[count]))
            count += 1
            collection.append(pd.concat(instance, axis=1, sort=False))
        result = pd.concat(collection,axis=0,sort=False)

        return result, label
    else:
        # label = tf.keras.utils.to_categorical(label, num_classes=8)
        return data,label

if __name__ == '__main__':
    data,label = load_dataset(dtype='np')
    window_sliding(data)
