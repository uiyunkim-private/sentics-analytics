import scipy.io
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('set_b/BVPr.mat')

print(list(mat.keys()))

data = mat['BVPr_37']

print(data.shape)
data = data.reshape((1,31983))
plt.plot(data[0])
plt.show()
print(type(mat))

print(mat)