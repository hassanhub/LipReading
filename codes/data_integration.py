from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Convolution3D, MaxPooling3D, LSTM
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import Bidirectional
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras import backend as BK
from keras.regularizers import l2
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

def get_activations(model, layer_in, layer_out, X_batch):
    get_activations = BK.function([model.layers[layer_in].input, BK.learning_phase()], [model.layers[layer_out].output])
    activations = get_activations([X_batch,0])
    return activations

print('Loading autoencoder model...')
config = BK.tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = BK.tf.Session(config=config)
model=load_model('autoencoder.h5',custom_objects={'corr2_mse_loss': corr2_mse_loss})
model.load_weights('autoencoder_weights.h5')

NUM_SAMPLES=20
NUM_SUBJECTS=34
FRAME_WIDTH=int(128)
FRAME_HEIGHT=int(128)
NUM_CHANNELS=1
THRESH_FRAME_COUNT=75
PATH='/home/user/LipReading/GRID'
TIME_WINDOW=40*5 #must be multiple of 40*5
TIME_OVERLAP=40*0 #must be multiple of 40*5
K=3000.0/TIME_WINDOW
#O=TIME_WINDOW/TIME_OVERLAP
window_size=TIME_WINDOW/40 #from time to number of frames 
overlap=TIME_OVERLAP/40 #number of frames
num_slices=int(THRESH_FRAME_COUNT/window_size)

################## VIDEO INPUT WITH DIFF ##################
def load_video_3D(path):
    
    cap = cv2.VideoCapture(path)
    frameCount = THRESH_FRAME_COUNT
    frameHeight=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT ))
    frameWidth=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH ))

    buf =np.empty((frameHeight, frameWidth, frameCount), np.dtype('float32'))
    fc = 0
    ret = True
    
    while (fc < frameCount  and ret):
        ret, frame = cap.read()
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame=mapstd(frame)
        frame=frame.astype('float32')
        #frame = frame-np.mean(frame)
        #frame = frame/np.std(frame)
        frame = frame-np.amin(frame)
        frame = frame/np.amax(frame)
        buf[:,:,fc]=frame
        fc += 1
    cap.release()

    #if ret==False:
    #    return -1

    return buf

def diff(buf_input):
    
    buf_input=np.pad(buf_input,((0,0),(0,0),(1,0)),'edge')
    buf_output=np.diff(buf_input,axis=2)
    return buf_output
    

def slice_video_3D(video):
    window_size=int(TIME_WINDOW/40) #from time to number of frames 
    video_output =np.empty((num_slices,3,FRAME_HEIGHT,FRAME_WIDTH,window_size), np.dtype('float32'))
    
    start=0
    for i in range(0,num_slices):
        video_output[i,:,:,:,:]=video[:,:,:,start:start+window_size]
        #print('start: '+str(start)+', end: '+str(start+window_size-1))
        start+=window_size
#        if start>THRESH_FRAME_COUNT-window_size:
#            break
    return video_output

#read path from text file and load video into tensor
text_file = open(PATH+'/valid_videos.txt', 'r')
lines = text_file.read().split('\n')
index_shuf=range(len(lines))
random.shuffle(index_shuf)

lines_shuf=[]
for i in index_shuf:
    lines_shuf.append(lines[i])
sio.savemat('index_s1_2_4_29.mat', mdict={'lines_shuf':lines_shuf})      

num_videos=len(lines)
#num_videos=NUM_SAMPLES
video_input =np.empty((num_videos*(num_slices),3,FRAME_HEIGHT,FRAME_WIDTH,TIME_WINDOW/40), np.dtype('float32'))

speaker_id=OrderedDict()
i=0
for row in lines_shuf:
    this_id=row.split('/')[6]
    if this_id in speaker_id:
        speaker_id[this_id]+=1
    else:
        speaker_id[this_id]=1

    tmp0=load_video_3D(row)
    
    diff_video=np.empty((3,tmp0.shape[0],tmp0.shape[1],tmp0.shape[2]))
    diff_video[0,:,:,:]=tmp0
    diff_video[1,:,:,:]=diff(tmp0)
    diff_video[2,:,:,:]=diff(diff_video[1,:,:,:])
    
    data_vid=slice_video_3D(diff_video)
    video_input[i*(num_slices):(i+1)*(num_slices),:,:,:,:]=data_vid[:,:,:,:,:]
    i+=1
    if i>=num_videos:
        break
    if i%10==0:
        print(str(i)+'/'+str(num_videos))
print('Video slices shape:'+str(video_input.shape))

################### AUDIO: AUDITORY SPECTROGRAM ##################
#FIR filtering funcitons
def slice_audio_spec(audio_spec):
    global AUDIO_LENGTH
    window_size=int(AUDIO_LENGTH/num_slices) #from time to number of audio index 
    #print('window_size='+str(window_size)+'AUDIO_LENGTH='+str(AUDIO_LENGTH))
    #print('SLICES:'+str(num_slices))
    audio_output =np.empty((num_slices,audio_spec.shape[0],window_size), np.dtype('float32'))
    
    start=0
    for i in range(0,num_slices):
        audio_output[i,:,:]=audio_spec[:,start:start+window_size]
        #print('start: '+str(start)+', end: '+str(start+window_size-1))
        start+=window_size
        if start>AUDIO_LENGTH-window_size:
            break
    return audio_output


def get_padded_spec(data):
    #calculate padding
#    data=np.log(data)
    data=np.power(data,.3)	
    t=data.shape[1]
    num_pads=int((2*num_slices)-(t%(2*num_slices)))
    #print(num_pads)
    padded_data=np.pad(data,((0,0),(0,num_pads)),'reflect')
    print('Getting bottleneck feature...')
    bottleneck=get_activations(model, 0, 12, padded_data.T)
    bottleneck=bottleneck[0].T
    return bottleneck


#read path from text file and load audio into tensor
text_file = open(PATH+'/valid_aud_specs.txt', 'r')
lines = text_file.read().split('\n')

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
audio_input =np.empty((num_audios*(num_slices),data.shape[0],int(AUDIO_LENGTH/num_slices)), np.dtype('float32'))
tmp =np.zeros((AUDIO_LENGTH), np.dtype('float32'))


i=0
first=True
for row in lines_shuf:
    mat=sio.loadmat(row)
    data = mat['aud'].T[:,2:]
    #data=20*data/np.amax(data)
    data=get_padded_spec(data=data)
    data=slice_audio_spec(data)
    audio_input[i*(num_slices):(i+1)*(num_slices),:,:]=data[:,:,:]
    i+=1
    if i>=num_audios:
        break
    if i%10==0:
        print(str(i)+'/'+str(num_audios))

audio_output=np.reshape(audio_input,(audio_input.shape[0],audio_input.shape[1]*audio_input.shape[2]))
print('Audio slices shape:'+str(audio_input.shape))
print('Target features to network shape:'+str(audio_output.shape))

print(speaker_id)

N=30
num_test=200
num_train=num_audios-num_test
L=int(np.ceil(num_train/N)*num_slices)

for i in range(N):
    if i<29:
        print('Saving data part'+str(i+1)+'...')
        start=i*L
        end=(i+1)*L
        print(str(start)+' to '+str(end))
        sio.savemat('/home/hassan/LipReading/data/preprocessed_data_final_part'+str(i+1)+'.mat', mdict={'video_input': video_input[start:end,:,:,:,:], 'audio_input' : audio_input[start:end,:,:]})
    else:
        print('Saving data part'+str(i+1)+'...')
        start=i*L
        end=num_train*num_slices
        print(str(start)+' to '+str(end))
        sio.savemat('/home/hassan/LipReading/data/preprocessed_data_final_part'+str(i+1)+'.mat', mdict={'video_input': video_input[start:end,:,:,:,:], 'audio_input' : audio_input[start:end,:,:]})

print('Saving validation data...')
start=num_train*num_slices
print(str(start)+' to '+str(video_input.shape[0]))
sio.savemat('/home/hassan/LipReading/data/preprocessed_data_final_validation.mat', mdict={'video_input': video_input[start:,:,:,:,:], 'audio_input' : audio_input[start:,:,:]})
