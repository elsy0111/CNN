import numpy as np
import matplotlib.pyplot as plt

input_path1 = "../DATASET/Dataset/images.npy"
# input_path2 = "D:/DATASET/88in3/Dataset_3000_88in3(1)/labels.npy"


images = np.load(input_path1)
# labels = np.load(input_path2)

show_images = True
if show_images == True:
    # 画像の確認
    plt.figure(figsize=(90,90))
    for i in range(1):
        plt.subplot(1,1,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i + 1],cmap="gray")
        #plt.xlabel(np.argmax(labels[i]) + 1)
        plt.xlabel(i)
        # print(labels[i])
    plt.show()