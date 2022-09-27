import numpy as np

input_path = "../-.npy"

array = np.load(input_path)
show_images = True

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
        print(labels[i])
    plt.show()
    plt.figure(figsize=(90,90))
    
