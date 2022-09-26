import numpy as np

# npy file
input_path = "pooling/images.npy"
output_path = "../-.npy"

array = np.load(input_path)

print("before shape (枚数, 縦, 横) : ",array.shape)

x,y = array.shape

array = array.reshape([x, 1, y])
# array = array.reshape([5000,250,250])

print("after shape (枚数, 縦, 横) : ",array.shape)

np.save(output_path, array)