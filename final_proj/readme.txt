#########################################

CNN.ipynb: CNN model
features.py: data processing
lstm.py: lstm model
ml.py : ml algorithms

#########################################
For reading and preprocessing .wav file:
In ML model, we use scipy.io.wavfile
In CNN model, we use torchaudio
In LSTN model,

For ML model:
We use these libraries to help us preprocessing data:
IPython
scipy.fft.fft
scipy.fft.fftfreq
scipy.signal.spectrogram.  //help us generate spectrogram
scipy.signal.find_peaks.    //help us get harmonics

We use sklearn to help us generate the model.
Here are the libraries we used:
sklearn.tree
sklearn.svm
sklearn.neighbors
sklearn.naive_bayes.GaussianNB
skelarn.ensemble.RandomForestClassifier

In this code, we manually select the harmonics and interval as our features and just put the data into several sklearn model, then we got the result.

For CNN model:
We use Pytorch to build our CNN model.
Here are the libraries we used:
torch
torchaudio.transforms
torchaudio.MelSpectrogram         //generate MelSpectrogram
torch.utils.data.Dataloader
torch.utils.data.Dataset
torch.utils.data.ranom_split
torch.nn.functional
torch.nn.init

In our code, we first read the data, uniform the size of each data, and assign training set, validation set and test set. Then we build a CNN model called class AudioClassifier(nn.module), define the training process, and inference is to calculate validation accuracy rate. Finally, we got the result.

 
