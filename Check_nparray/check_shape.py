import numpy as np

# npy file
input_path = "../DATASET/images.npy"

array = np.load(input_path)

print("shape (枚数, 縦, 横) : ",array.shape)