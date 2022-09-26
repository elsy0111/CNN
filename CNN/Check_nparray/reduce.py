import numpy as np

# npy file
input_path = "../-.npy"
output_path = "../-.npy"

array = np.load(input_path)

print("before n array : ",len(array))

array = array[:5000]

print("after n array : ",len(array))

np.save(output_path,array)