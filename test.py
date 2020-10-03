import scipy.io
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('set_a/day1.mat')

print(list(mat.keys()))

data = mat['day1']

print(data.shape)
data = data.reshape((32,2001))
plt.plot(data[0])
plt.show()
print(type(mat))

print(mat)