from __future__ import print_function
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape, noise
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D, LSTM
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras import backend as BK
from keras.regularizers import l1,l2,l1_l2
from sklearn import preprocessing
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
import cv2
import IPython
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from scipy.signal import spectrogram
from scipy import signal
from scipy.signal import firls, lfilter
from numpy.lib import stride_tricks
import scipy.io as sio
from collections import OrderedDict
import random 
random.seed(100)

def corr2_mse_loss(a,b):
    a = BK.tf.subtract(a, BK.tf.reduce_mean(a))
    b = BK.tf.subtract(b, BK.tf.reduce_mean(b))
    tmp1 = BK.tf.reduce_sum(BK.tf.multiply(a,a))
    tmp2 = BK.tf.reduce_sum(BK.tf.multiply(b,b))
    tmp3 = BK.tf.sqrt(BK.tf.multiply(tmp1,tmp2))
    tmp4 = BK.tf.reduce_sum(BK.tf.multiply(a,b))
    r = -BK.tf.divide(tmp4,tmp3)
    m=BK.tf.reduce_mean(BK.tf.square(BK.tf.subtract(a, b)))
    rm=BK.tf.add(r,m)
    return rm

NUM_SAMPLES=10
NUM_SUBJECTS=34
FRAME_WIDTH=int(128)
FRAME_HEIGHT=int(64)
NUM_CHANNELS=1
THRESH_FRAME_COUNT=75
PATH='/home/user/LipReading/GRID'
TIME_WINDOW=40*10 #must be multiple of 40*5
TIME_OVERLAP=40*5 #must be multiple of 40*5
K=3000.0/TIME_WINDOW
O=TIME_WINDOW/TIME_OVERLAP
window_size=TIME_WINDOW/40 #from time to number of frames 
overlap=TIME_OVERLAP/40 #number of frames
num_slices=int(((THRESH_FRAME_COUNT-window_size)/overlap)+1)

#### Getting auditory spectrograms
def get_padded_spec(data):
    #calculate padding
    data=np.power(data,.3)
    t=data.shape[1]
    num_pads=int((2*num_slices)-(t%(2*num_slices)))
    #print(num_pads)
    padded_data=np.pad(data,((0,0),(0,num_pads)),'reflect')
    
    return padded_data


#read path from text file and load audio into tensor
text_file = open(PATH+'/valid_aud_specs.txt', 'r')
lines = text_file.read().split('\n')

index_shuf=range(len(lines))
random.shuffle(index_shuf)

lines_shuf=[]
for i in index_shuf:
    lines_shuf.append(lines[i])

num_audios=len(lines)
#num_audios=NUM_SAMPLES
#Get audio length
mat=sio.loadmat(lines[0])
data = mat['aud'].T[:,2:]
data=get_padded_spec(data=data)
global AUDIO_LENGTH
AUDIO_LENGTH=data.shape[1]
audio_input =np.empty((num_audios*AUDIO_LENGTH,data.shape[0]), np.dtype('float32'))


i=0
first=True
for row in lines_shuf:
    mat=sio.loadmat(row)
    data = mat['aud'].T[:,2:]
    #data=20*data/np.amax(data)
    data=get_padded_spec(data=data)
    audio_input[i*AUDIO_LENGTH:(i+1)*AUDIO_LENGTH,:]=data.T
    i+=1
    if i>=num_audios:
        break
    if i%100==0:
        print(str(i)+'/'+str(num_audios))
#sio.savemat('index_s1_2_4_29.mat', mdict={'lines_shuf':lines_shuf})      

N=10
num_test=20
num_train=num_audios-num_test
train_edge=num_train*AUDIO_LENGTH
audio_input_train=audio_input[:train_edge,:]
audio_input_test=audio_input[train_edge:,:]

print('Shape of all the data:'+str(audio_input.shape))
print('Shape of the train data to autoencoder:'+str(audio_input_train.shape))
print('Shape of the validation data:'+str(audio_input_test.shape))


config = BK.tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = BK.tf.Session(config=config)
adam=Adam(lr=.0001)
reg=.001
model=Sequential()

model.add(Dense(512, input_shape=(audio_input_train.shape[1],)))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(128))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(32,kernel_regularizer=l1_l2(.001)))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))
model.add(noise.GaussianNoise(.05))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(128))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(audio_input_train.shape[1]))
model.compile(loss=corr2_mse_loss, optimizer=adam)
model.summary()

num_iter=40
loss_history=np.empty((num_iter,2), dtype='float32')
for i in range(num_iter):
    print('################ Autoencoder model, iteration: '+str(i)+'/'+str(num_iter))
    history = model.fit(audio_input_train, audio_input_train, batch_size=128, epochs=1, verbose=1, validation_data=(audio_input_test,audio_input_test))
    loss_history[i,0]=history.history['loss'][0]
    loss_history[i,1]=history.history['val_loss'][0]
    sio.savemat('autoencoder_sigmoid32_Noise_history_s1_2_4_29.mat', mdict={'history':loss_history})

model.save('autoencoder.h5')
model.save_weights('autoencoder_weights.h5')
#steps per epoch=ceil(2730/32)* 10
#valiation=1

