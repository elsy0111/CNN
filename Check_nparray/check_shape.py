import numpy as np

# npy file
input_path = "../DATASET/images.npy"

array = np.load(input_path)

array = array.reshape([1,250,250])

print("shape (枚数, 縦, 横) : ",array.shape)
np.save("../DATASET/images.npy",array)