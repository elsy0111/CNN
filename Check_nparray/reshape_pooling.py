import numpy as np
import skimage.measure

# npy file
input_path  = "../DATASET/images.npy"
output_path = "../DATASET/images_reshape.npy"

array = np.load(input_path)

print("before shape (枚数, 縦, 横) : ",array.shape)

x,y = array.shape

array = array.reshape([1, x, y])

print("after shape (枚数, 縦, 横) : ",array.shape)

# pooling = (x,y,z) = (枚数, 縦, 横)
# array = skimage.measure.block_reduce(array, (1,1,2), np.max)

# print("after pooling shape (枚数, 縦, 横) : ",array.shape)

np.save(output_path,array)