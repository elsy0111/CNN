import numpy as np
import matplotlib.pyplot as plt

input_path1 = "../DATASET/Dataset_3/images.npy"
input_path2 = "../DATASET/Dataset_3/labels.npy"

images = np.load(input_path1)
labels = np.load(input_path2)
show_images = True
for j in range(9):
    print(labels[j])
if show_images == True:
    # 画像の確認
    plt.figure(figsize=(90,90))
    """plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[5])
    print(labels[5])
    print(labels[5])
    plt.show()
    """
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        #plt.xlabel(np.argmax(labels[i]) + 1)
        plt.xlabel(i)
        # print(labels[i])
    plt.show()
    plt.figure(figsize=(90,90))
    
