import numpy as np
from scipy.io.wavfile import read

all_max = []
all_min = []

for i in range(1,9 + 1):
    b,a = read("audio\Sample_Audio\E0"+str(i)+".wav")
    print(np.max(a))
    print(np.min(a))
    print()
    all_max.append(np.max(a))
    all_min.append(np.min(a))

for i in range(10,44+1):
    b,a = read("audio\Sample_Audio\E"+str(i)+".wav")
    print(np.max(a))
    print(np.min(a))
    print()
    all_max.append(np.max(a))
    all_min.append(np.min(a))

print(max(all_max))
print(min(all_min))