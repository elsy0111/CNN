import librosa
import numpy as np
from scipy.io.wavfile import read
from itertools import chain
import os
import datetime


#--------------Set Parameter--------------#
PCM = 48000
fft_size = 2048                 # Frame length
hl = int(fft_size / 4)          # Frame shift length
hi = 250                        # Height of image
wi = 250 - 1                    # Width of image
F_max = 20000                   # Freq max
#--------------Set Parameter--------------#

images = []
read_path = "audio\.wav" # ! !!!!!!!!!!!!!!!!!!!!
output_path = "../DATASET/"

PCM_dontuse, data = read(read_path)

print(len(data))

n_empty = 48000 * 3 - len(data)


if n_empty < 0:
    split_num = 72000
    data1 = data[:split_num]
    data2 = data[split_num:]

    n_empty1 = 48000 * 3 - len(data1)
    n_empty2 = 48000 * 3 - len(data2)

    empty_list1 = np.zeros(n_empty1)
    data1 = np.array(list(chain(data1,empty_list1)))

    empty_list2 = np.zeros(n_empty2)
    data2 = np.array(list(chain(data2,empty_list2)))

    data1 = np.array(data1)
    data1 = data1/2**15

    data2 = np.array(data2)
    data2 = data2/2**15

    mono_data1 = data1[0:wi*hl]
    mono_data2 = data2[0:wi*hl]

#--------------STFT--------------#
    S1 = librosa.feature.melspectrogram(
        y = mono_data1, sr = PCM, n_mels = hi, fmax = F_max, hop_length = hl, 
        win_length = fft_size, n_fft = fft_size)

    S_dB1 = librosa.power_to_db(S1, ref = np.max)
#--------------STFT--------------#

    S_dB1 = np.flipud(S_dB1)

    print("shape (枚数, 縦, 横) : ",S_dB1.shape)

    S_dB1 = S_dB1.reshape([1,250,250])

    print("shape (枚数, 縦, 横) : ",S_dB1.shape)


    dt_now = datetime.datetime.now()

    da = dt_now.day
    ho = dt_now.hour
    mi = dt_now.minute
    se = dt_now.second
    ms = dt_now.microsecond

    file_name = str(da) + str(ho) + str(mi) + str(se) + str(ms)

    os.mkdir(output_path + file_name)
    np.save(output_path + file_name + "/images.npy",S_dB1)

    zero_arr = np.zeros(88,dtype = int)
    zero_arr = zero_arr.reshape([1, 88])

    np.save(output_path + file_name + "/labels.npy",zero_arr)


#--------------STFT--------------#
    S2 = librosa.feature.melspectrogram(
        y = mono_data2, sr = PCM, n_mels = hi, fmax = F_max, hop_length = hl, 
        win_length = fft_size, n_fft = fft_size)

    S_dB2 = librosa.power_to_db(S2, ref = np.max)
#--------------STFT--------------#

    S_dB2 = np.flipud(S_dB2)

    print("shape (枚数, 縦, 横) : ",S_dB2.shape)

    S_dB2 = S_dB2.reshape([1,250,250])

    print("shape (枚数, 縦, 横) : ",S_dB2.shape)

    np.save(output_path,S_dB2)

    dt_now = datetime.datetime.now()

    da = dt_now.day
    ho = dt_now.hour
    mi = dt_now.minute
    se = dt_now.second
    ms = dt_now.microsecond

    file_name = str(da) + str(ho) + str(mi) + str(se) + str(ms)

    os.mkdir(output_path + file_name)
    np.save(output_path + file_name + "/images.npy",S_dB2)

    zero_arr = np.zeros(88,dtype = int)
    zero_arr = zero_arr.reshape([1, 88])

    np.save(output_path + file_name + "/labels.npy",zero_arr)

else:
    empty_list= np.zeros(n_empty)

    data = np.array(list(chain(data,empty_list)))
    data = np.array(data)
    data = data/2**15

    mono_data = data[0:wi*hl]

#--------------STFT--------------#
    S = librosa.feature.melspectrogram(
        y = mono_data, sr = PCM, n_mels = hi, fmax = F_max, hop_length = hl, 
        win_length = fft_size, n_fft = fft_size)

    S_dB = librosa.power_to_db(S, ref = np.max)
#--------------STFT--------------#

    S_dB = np.flipud(S_dB)

    print("shape (枚数, 縦, 横) : ",S_dB.shape)

    S_dB = S_dB.reshape([1,250,250])

    print("shape (枚数, 縦, 横) : ",S_dB.shape)


    dt_now = datetime.datetime.now()

    da = dt_now.day
    ho = dt_now.hour
    mi = dt_now.minute
    se = dt_now.second
    ms = dt_now.microsecond

    file_name = str(da) + str(ho) + str(mi) + str(se) + str(ms)

    os.mkdir(output_path + file_name)
    np.save(output_path + file_name + "/images.npy",S_dB)

    zero_arr = np.zeros(88,dtype = int)
    zero_arr = zero_arr.reshape([1, 88])

    np.save(output_path + file_name + "/labels.npy",zero_arr)