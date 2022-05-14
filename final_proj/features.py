#%% 
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os 
import wave
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import spectrogram

#%% file -> data, label 
major_path = "./Major"
minor_path = "./Minor"
length = []
data = []
label = []
max_length = 99559
# samplerate = 44100
def zeros_starts(signal):
    N = len(signal)
    for i in range(1, N):
        if signal[-i] != 0:
            return N - i + 1

for path in [major_path, minor_path]:

    for root, dirs, files in os.walk(major_path):
        for file in files:
            if file.endswith(".wav"):
                wavepath = os.path.join(root, file)
                fs, signal = wavfile.read(wavepath)
                # data.append(signal)
                # signal = np.array(signal)
                # print(np.array(signal).shape)
                # length.append(len(signal)/44100)
                # print(samplerate)
                
                
                # wave gram
                # time = np.linspace(0. , len(signal)/samplerate, len(signal))
                # plt.plot(time, signal)
                # plt.show()
                

                # remove zeros behind
                first_zero_index = zeros_starts(signal)
                valid_signal = signal[:first_zero_index]
                # append zeros
                length.append(len(valid_signal))

                row = np.hstack((np.array(valid_signal), np.zeros((max_length - len(valid_signal), ))))
                f, t, Sxx = spectrogram(row, fs)
                data.append(Sxx)
                if path == major_path:
                    label.append(1)
                else:
                    label.append(0)

# print(max(length))
# print(min(length))
data = np.array(data)
label = np.array(label)

# mm_scaler = MinMaxScaler()
# data = mm_scaler.fit_transform(data)

if False:
    np.savetxt("data.txt", data)
    np.savetxt("label.txt", label)

print(data)
print(data.shape)
print(data.dtype)


#%% analyze the first file 
import wave

example_path = "./Major/Major_0.wav"

with wave.open(example_path) as wave_file:
    nchannels, sample_width, framerate, nframes = wave_file.getparams()[:4]
    print(nchannels, sample_width, framerate, nframes)

fs, signal = wavfile.read(example_path)
print(fs)
for i in range(1, len(signal)):
    if signal[-i] != 0:
        print(signal[len(signal)-i+1])
        break

def zeros_starts(signal):
    N = len(signal)
    for i in range(1, N):
        if signal[-i] != 0:
            return N - i
            

#%%

# split into train valid test dataset
from sklearn.model_selection import train_test_split

X_train, X_rem, y_train, y_rem = train_test_split(data,label, train_size=0.7)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)


#%% fft




#%% spectrum 
# from torchaudio.functional import spectrogram
example_minor = "./Minor/Minor_0.wav"
example_major = "./Major/Major_0.wav"
fs, signal = wavfile.read(example_major)
f, t, Sxx = spectrogram(signal, fs)#, nperseg=10, nfft=50000)
print(Sxx.shape)
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].pcolormesh(t, f, np.log(Sxx), cmap="jet")

fs, signal = wavfile.read(example_minor)
f, t, Sxx = spectrogram(signal, fs)#, nperseg=10, nfft=50000)
axes[1].pcolormesh(t, f, np.log(Sxx), cmap="jet")

axes[0].set_title(example_major)
axes[1].set_title(example_minor)

# axes[0].set(xlabel='Time [sec]', ylabel='Frequency [Hz]')
# # axes[1].pcolormesh(t, f[:1500], np.log(Sxx)[:1500,:], cmap="jet")
# # axes[1].set_title("Spectogram (Zoomed)")
# # axes[1].set(xlabel='Time [sec]', ylabel='Frequency [Hz]')
plt.show()
import torchaudio
waveform, sample_rate = torchaudio.load(example_minor)
yes_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=1024, hop_length=None, n_mels=64)(waveform)
im = yes_spectrogram.log2()[0,:,:].numpy()
print(im.shape)
plt.imshow(im, cmap='viridis')
plt.show()


#%% 


import librosa
import numpy as np


def add_noise(audio_path, out_path, percent=0.2, sr=16000):
    src, sr = librosa.load(audio_path, sr=sr)
    random_values = np.random.rand(len(src))
    src = src + percent * random_values
    librosa.output.write_wav(out_path, src, sr, norm=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    for root, dirs, files in os.walk(args.audio_dir):
        for file in files:
            if not file.endswith(".wav"):
                continue
            audio_path = os.path.join(root, file)
            out_path = os.path.join(args.out_dir, file + ".noise.wav")
            add_noise(audio_path, out_path)
#%%

from torchaudio.transforms import MFCC
import torch
example_minor = "./Minor/Minor_0.wav"
fs, signal = wavfile.read(example_major)
mm_scaler = MinMaxScaler()
signal = mm_scaler.fit_transform([signal])

mfcc_module = MFCC(sample_rate=fs,  n_mfcc=20, melkwargs={"n_fft": 2048, "hop_length": 1024, "power": 2})
torch_mfcc = mfcc_module(torch.tensor(signal).to(torch.float32))
print(torch_mfcc)
import matplotlib.pyplot as plt 

plt.imshow(torch_mfcc[0])
plt.show()





# %%
