import numpy as np

a = np.load("Dataset/images.npy")

print(a)
print(len(a))
print(len(a[0]))
print(len(a[0][0]))

b = np.load("Dataset/labels.npy")

print(b)
print(len(b))
print(len(b[0]))