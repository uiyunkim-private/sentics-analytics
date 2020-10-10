import scipy.io
import matplotlib.pyplot as plt
import math

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

        list_data.append({'data':data[i-1],
                          'emotion':t1,
                          'day':day,
                          'sensor':t2})
data = [x for x in list_data if x['emotion'] == 'Anger']
print(data)

