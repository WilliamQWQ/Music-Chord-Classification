#%%
from statistics import mode
import torch
import torch.nn as nn
import numpy as np
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os 
import wave
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import spectrogram
import torchaudio
from  torchaudio import transforms

import os
import IPython
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram, find_peaks
import torch
import torch.nn as nn
import torchaudio
from torchaudio import transforms
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import random_split
import torchvision.models as models
import torchvision
import librosa

#%%
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

def add_noise(audio_path, out_path, percent=0.2, sr=16000):
    src, sr = librosa.load(audio_path, sr=sr)
    random_values = np.random.rand(len(src))
    src = src + percent * random_values
    librosa.output.write_wav(out_path, src, sr, norm=True)

for path in [major_path, minor_path]:

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                # print(file)
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


                # file_name = wavepath
                # y, sr = librosa.load(file_name, sr=None)
                # y_16k = librosa.resample(y, sr, 16000)
                # librosa.output.write_wav(file_name, y_16k, 16000)


                # remove zeros behind
                first_zero_index = zeros_starts(signal)
                valid_signal = signal[:first_zero_index]
                # append zeros
                # length.append(len(valid_signal))

                row = np.hstack((np.array(valid_signal), np.zeros((max_length - len(valid_signal), ))))
                f, t, Sxx = spectrogram(row, fs)
                data.append(Sxx.T)
                pr
                # data.append(row)
                if path == major_path: 
                    label.append(1)
                else:
                    label.append(0)
                

# print(max(length))
# print(min(length))
data = np.array(data)
print(data.shape)
label = np.array(label)
# input size: batch * 99559 * 1 
from sklearn.model_selection import train_test_split

X_train, X_rem, y_train, y_rem = train_test_split(data, label, train_size=0.7)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

print(X_train.shape)

print(torch.cuda.is_available())

y_train = y_train.astype('int32')






#%% 
path = "."
data = []
classid = []
max_len = 101429
n_mels=64
n_fft=1024
hop_len=None
count = 0
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith(".wav"):
            count+=1
            foldername = os.path.basename(dirname)
            full_path = os.path.join(dirname, filename)
            
            # print(dirname)
            if(dirname == './Major'):
                classid.append(1)
            if(dirname == './Minor'):
                classid.append(0)
            signal,fs = torchaudio.load(full_path)
            num_rows, sig_len = signal.shape
            if(sig_len < max_len):
                pad_begin_len = random.randint(0, max_len - sig_len)
                pad_end_len = max_len - sig_len - pad_begin_len
                # Pad with 0s
                pad_begin = torch.zeros((num_rows, pad_begin_len))
                pad_end = torch.zeros((num_rows, pad_end_len))

                signal = torch.cat((pad_begin, signal, pad_end), 1)
    #             signal,fs = aud
    #             _, sig_len = signal.shape
    #             shift_amt = int(random.random() * shift_limit * sig_len)
                
    #         print(sig_len)
    #         fs, signal = wavfile.read(full_path)
    #         N = len(signal)
    #         time = np.linspace(0., N/fs, N)
    #         f, t, Sxx = spectrogram(signal, fs, nperseg=10000, nfft = 50000)

            # spec = transforms.MelSpectrogram(fs, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(signal)
            # spec = transforms.AmplitudeToDB(top_db=80)(spec)
            # spec = spec.squeeze()
            # mm_scaler = MinMaxScaler()
            # spec = mm_scaler.fit_transform(spec)

#         print(np.array(Sxx).shape)
            # data.append(torch.tensor(spec.T).to(torch.float32))
            data.append(signal.T)
#         print(signal.shape)
# print(count)
# print(np.array(data).shape)

# print(data)
x = list(zip(data, classid))
# print(np.array(x).shape)
num_items = len(x)
# num_items = len(data)
num_train = round(num_items * 0.85)
num_test = num_items - num_train 
train_ds, test_ds = random_split(x, [num_train, num_test])
# print(num_items)
# print(num_train)
num_val = num_test # 0.25

num_newtrain = num_train - num_val # 0.85 - 0.25


print(num_val)
print(num_newtrain)
# print(num_)
newtrain_ds, val_ds = random_split(train_ds,[num_newtrain, num_val])
train_dl = torch.utils.data.DataLoader(newtrain_ds, batch_size=32,  shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds,batch_size=32, shuffle=True)


#%%

# X_train = X_train[:,:, np.newaxis]

#%%
class rnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 3, batch_first=True) # raw data
        # self.lstm = nn.LSTM(64, 128, 3, batch_first=True) # spec data
        self.classifier = nn.Linear(64, 2)
        self.act = nn.Softmax()

        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)


    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)

        x = self.classifier(out[:, -1, :])
        return x


def predict (model, val_dl):
  correct_prediction = 0
  total_prediction = 0

  # Disable gradient updates
  with torch.no_grad():
    for data in val_dl:
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data[0].cuda(), data[1].cuda()

      # Normalize the inputs
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      # Get predictions
      outputs = model(inputs)

      # Get the predicted class with the highest score
      _, prediction = torch.max(outputs,1)
      # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
    
  acc = correct_prediction/total_prediction
  return acc
#   print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

loss_list = []
iteration_list = []
accuracy_list = []
num_epochs = 800
count = 0
valid_acc_list =[]
model= rnn().cuda()

loss_func = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam( model.parameters(), lr=0.0001)#, weight_decay=0.001)

batch_size = 8

for epoch in range(num_epochs):
    correct_prediction = 0
    running_loss = 0.0
    total_prediction = 0

    for i, data in enumerate(train_dl):
    # for i in range(0, 696, batch_size):
        # input_vec = torch.tensor(X_train[i:i+batch_size]).to(torch.float32).cuda()
        # label = torch.tensor(y_train[i:i+batch_size]).to(torch.long).cuda()
        # print(input_vec.shape)
        input_vec, label = data[0].cuda(), data[1].cuda()
        # input_vec = input_vec.to(torch.float32).cuda()
        # print(input_vec.shape)
        outputs = model(input_vec)
        # print(outputs)
        # print(label.shape)
        loss = loss_func(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Keep stats for Loss and Accuracy
        running_loss += loss.item()

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == label).sum().item()
        total_prediction += prediction.shape[0]
        print()
        #if i % 10 == 0:    # print every 10 mini-batches
        #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
    
    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    loss_list.append(avg_loss)
    accuracy_list.append(acc)

    valid_acc = predict(model, val_dl)
    valid_acc_list.append(valid_acc)
    print(f'Epoch: {epoch}, Loss: {avg_loss:.5f}, Accuracy: {acc:.5f}',"valid:", valid_acc)

#   print('Finished Training')
    #     predictions = torch.max(outputs, 1)[1]
    #     correct += (predictions == label).sum()
    #     total_prediction += predictions.shape[0]

    #     loss_list.append(loss.data)
    #     iteration_list.append(i)
    #     accuracy_list.append(correct / len())

    #     count += 1
    #     print('epoch:', str(epoch), 'iteration',count, 'loss', loss.data, 'accuracy:', correct/len(X_train))

    # num_batches = len(X_train)
    # avg_loss = running_loss / num_batches
    # acc = correct/total_prediction
    # print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

#%%



# Run inference on trained model with the validation set
predict(model, test_dl)

#%%
import matplotlib.pyplot as plt
# print(accuracy_list)
plt.title("LSTM learning curve with raw wave data")
plt.plot([x for x in accuracy_list], label = "acc")
plt.plot( [x for x in loss_list], label="loss")
# plt.plot([x for x in valid_acc_list], label = "valid")

plt.legend()
plt.show()

#%%
for epoch in range(num_epochs):
    correct = 0
    for i in range(0, len(data_train)-1, batch_size):
        chunk_end = min(batch_size+i, len(data_train)-1)
        input_vec = torch.tensor(data_train[i:chunk_end]).to(torch.float32)
        # print(input_vec)
        label = torch.tensor(label_train[i+1:chunk_end+1])

        outputs = model(input_vec)
        
        loss = loss_func(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        predictions = torch.max(outputs, 1)[1]
        correct += (predictions == label).sum()
        loss_list.append(loss.data)
        iteration_list.append(count)
        accuracy_list.append(correct / len(data_train))

    print('epoch:', str(epoch), 'iteration',count, 'loss', loss.data, 'accuracy:', correct/len(data_train))


#%%
# print(label_test.shape)


preds = []
labs = []
with torch.no_grad():
    total = 0
    correct = 0
    batch_size =1
    total = len(label)

    
    model.eval()
    label = torch.tensor(label_test)

    for i in range(0, len(data_test)-1, batch_size):
        chunk_end = min(batch_size+i, len(data_test))
        # if i == 0:
        #     continue
        # print(input_vec.shape)
        input_vec = torch.tensor(data_test[i:chunk_end]).to(torch.float32)
        # print(input_vec)
        
        
        output = model(input_vec)

        predictions = torch.max(output, 1)[1]
        preds.append(predictions.numpy()[:])
        labs.append(label.numpy()[:])
        correct += 1 if predictions == label[i+1] else 0 #(predictions == label).sum().numpy()
        if predictions == label[i+1]:
            print(i+1)
        

        
    print('test acc:',correct/total)

