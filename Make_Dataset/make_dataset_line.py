#-----IMPORT-----#
from itertools import chain
from random import randint
import numpy as np
from scipy.io.wavfile import read
import os
import random
#-----IMPORT-----#













dataset_num = 2000
N = 3

images = []
labels =[]

dataset_cnt = 0

while dataset_cnt < dataset_num:

    ValueErr = False

#--------------Make Random List(length = 88)--------------#

    t = []

    while len(t) < N:
        j = randint(1,4)
        t.append(j)
        t = list(set(t))
    t.sort()

    s_list = [0] * 44

    for i in t:
        s_list[i-1] = 1

    cnt = 0
    list88 = [0] * 88

    for i in s_list:
        if i == 1:
            j = randint(0,1)
            if j == 1:
                list88[cnt] = 1
                list88[cnt + 1] = 0
            else:
                list88[cnt] = 0
                list88[cnt + 1] = 1
        cnt += 2

    # print("answer_label : ",list88)
    
#--------------Make Random List(length = 88)--------------#

#--------------Make filename by list88--------------#
    audio_list = []


    for i,j in enumerate(list88):
        if j == 1:
            if i%2 == 0: #日本語
                i = int(i/2) + 1
                if len(str(i)) == 1:
                    l = "J0" + str(i)
                else:
                    l = "J" + str(i)
            else:#英語
                i = int(i/2) + 1
                if len(str(i)) == 1:
                    l = "E0" + str(i)
                else:
                    l = "E" + str(i)
            audio_list.append(l)

#--------------Make filename by list88--------------#



#--------------Make delay_list--------------#
    all_data = []
    delay_list = []
    raw_audio_length_list = []
    audio_length_list = []

    for name in audio_list:
        PCM, data = read("audio/Sample_Audio/"+name+".wav")
        raw_audio_length_list.append(len(data))
        delay_random_num = randint(0, 5) * 4800    #! random delay No DEBUG
        delay_list.append(delay_random_num)
        cut_offset_data = data[delay_random_num:]
        all_data.append(cut_offset_data)
        audio_length_list.append(len(cut_offset_data))

#--------------Make delay_list--------------#

#------------------Fill Zero----------------#
    max_audio_length = max(audio_length_list)

    result = np.zeros(max_audio_length,dtype = int)

    for data in all_data:
        n_empty = max_audio_length - len(data)
        empty_list = np.zeros(n_empty,dtype = int)
        long_data = list(chain(data,empty_list))
        result += long_data

#------------------Fill Zero----------------#


#------------------Delete------------------#
    split_bool = True
    timeout_cnt = 0
    timeout_bool = False

    while split_bool:
        timeout_cnt += 1
        cnt = 0
        n_split = randint(2,5) #! n_split
        if timeout_cnt > 20:
            print("TIME OUT")
            timeout_bool = True
            break
        while (cnt < 50):
            cnt += 1
            delete_num = randint(0,250000)
            if delete_num <= len(result) - 0.5 * 48000 * n_split:
                if (len(result) - delete_num)/n_split <= 48000 * 3:
                    split_bool = False
                    break   # ok
    
    if timeout_bool:
        continue

    result = result[:len(result) - delete_num]
#------------------Delete------------------#

#------------------Export audio----------------#

    result = np.array(result,dtype=np.float32)
    result /= 2**15
    frames = len(result)

#------------------Export audio----------------#

#-----------------cut list------------------
    c = True
    timeout_cnt = 0
    timeout_bool = False

    while c:
        timeout_cnt += 1
        split_list = []
        if timeout_cnt > 100:
            print("TIME OUT C")
            timeout_bool = True
            break 
        for i in range(n_split - 1):
            split_list.append(randint(1,frames))
        split_list.sort()
        split_list.insert(0,0)
        split_list.append(frames)
        c = False
        for i in range(n_split):
            if split_list[i + 1] - split_list[i] <= 0.5 * 48000:
                c = True
    
    if timeout_bool:
        continue
#-----------------cut list------------------

#-----------------cut audio------------------
    split_list[-1] += 1
    split_audio = []

    for j in range(n_split):
        split_data = data[split_list[j]:split_list[j + 1]]
        n_empty = 48000 * 3 - len(split_data)
        try:
            empty_list = np.zeros(n_empty)
        except ValueError:
            print("value Error (split_data is too large)")
            ValueErr = True
            break
        same_length_data = np.array(list(chain(split_data,empty_list)))
        split_audio.append(same_length_data)
#---------------------------Make Audio end-----------------------------#

    if ValueErr:
        continue
    
    for j in range(n_split):
        data = split_audio[j] 













        images.append(data)
        labels.append(np.array(list88))    #* ラベルに追加

    dataset_cnt += 1
    print("dataset_cnt : ", dataset_cnt)

r = random.randint(1000000,10000000)

os.mkdir("../Dataset_" + str(dataset_num) + "_" + str(N) + "h" + str(r))

np.save("../Dataset_" + str(dataset_num) + "_" + str(N) +  "h" + str(r) + "/images.npy",images)
np.save("../Dataset_" + str(dataset_num) + "_" + str(N) +  "h" + str(r) + "/labels.npy",labels)

#*------------------------------------------------------------------------