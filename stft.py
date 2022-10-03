import librosa
import numpy as np
from scipy.io.wavfile import read
from itertools import chain

#--------------Set Parameter--------------#
PCM = 48000
fft_size = 2048                 # Frame length
hl = int(fft_size / 4)          # Frame shift length
hi = 250                        # Height of image
wi = 250 - 1                    # Width of image
F_max = 20000                   # Freq max
#--------------Set Parameter--------------#

images = []

PCM_dontuse, data = read("audio\Sample_Q\sample_Q_E01\problem1.wav")

n_empty = 48000 * 3 - len(data)

empty_list = np.zeros(n_empty)

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

np.save("../DATASET/images.npy",S_dB)