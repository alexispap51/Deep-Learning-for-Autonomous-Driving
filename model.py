"""
Created on Sun Jun 30 18:59:22 2019

@author: Papadimitriou Alexios
"""

#Import Libraries
from __future__ import division
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalAveragePooling3D
from keras.models import Model
from keras.utils import plot_model
from keras.layers.merge import concatenate
from keras.layers import Dropout, Input, BatchNormalization, LSTM, UpSampling1D, TimeDistributed
from keras.models import Model, load_model, model_from_json
from keras.optimizers import Adam
from tensorflow.python.client import device_lib
import pydot
import numpy as np
import keras
import cv2
import h5py
import matplotlib.pyplot as plt
import sklearn
import math
import tensorflow as tf
import random

EPOCHS = 2 #Number of training epochs 
SEQ_LEN = 10 #LSTM Sequence Length
STEER = 10 #Past Predictions/Steering input sequence length
BATCH_SIZE = 5 #Number of mini batches
LEFT_CONTEXT = 3 #CNN Depth Dimension
# Image downsampling parameters.
R_HEIGHT = 240
HEIGHT = 120
WIDTH = 320
CHANNELS = 1 # Number of image channels. We use the Y of YUV color space.

# Our training data follows the "interpolated.csv" format from Ross Wightman's scripts.
CSV_HEADER = "index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt".split(",")
OUTPUTS = CSV_HEADER[-6:-3] # angle,torque,speed
OUTPUT_DIM = len(OUTPUTS) 

#Data generation during training
def dataGen(fileName, imageSequence, batch_size, steps, session, graph, model):
    depth = LEFT_CONTEXT + SEQ_LEN
    batch_start = depth
    batch_end = batch_size+depth
    while True:         
        inputs = []
        targets = []
        steering = []
        for j in range(batch_start,batch_end): 
            k = indexes[j]            
            ln_target = zip(imageSequence[k])
            for l in range(k-LEFT_CONTEXT, k):
                for i in range(l - SEQ_LEN, l):
                    ln_input = zip(imageSequence[i])
                    temp_image = cv2.cvtColor(cv2.imread(fileName + ''.join(ln_input[0])), cv2.COLOR_BGR2YUV)
                    temp_image = cv2.resize(temp_image, (WIDTH, R_HEIGHT))
                    temp_image = temp_image[-HEIGHT:, 0:WIDTH]                
                    temp_image, u, v = cv2.split(temp_image)
                    img_mean = np.mean(temp_image/255)
                    temp_image = (temp_image/255)-img_mean              
                    inputs.append(temp_image)
            angle = (float(''.join(ln_target[1])))
            targets.append(angle)    
            for i in range(k-STEER,k):
                t = zip(imageSequence[i])
                steering.append(float(''.join(t[1]))+random.uniform(-noise,noise))
            if k == indexes[batch_end-1]:
                X1 = np.array(inputs)
                X1 = np.reshape(X1, (batch_size,SEQ_LEN,HEIGHT,WIDTH,LEFT_CONTEXT,CHANNELS))
                X2 = np.array(steering)
                X2 = np.reshape(X2, (batch_size, STEER, 1))
                Y = np.array(targets)
                Y = np.transpose(Y)
                yield ([X1, X2], Y)  #a tuple with two numpy arrays with batch_size samples     
                batch_start += batch_size  
                batch_end += batch_size
            if batch_end >= len(indexes):             
                np.random.shuffle(indexes)
                batch_start = depth
                batch_end = batch_size+depth

#Data generation for validation purposes                
def valGen(fileName, imageSequence, batch_size, steps, session, graph, model):
    depth = LEFT_CONTEXT + SEQ_LEN
    batch_start = depth
    batch_end = batch_size+depth
    while True:         
        targets = []
        inputs = []
        steering = []
        for k in range(batch_start,batch_end): 
            ln_target = zip(imageSequence[k])
            for l in range(k-LEFT_CONTEXT, k):
                for i in range(l - SEQ_LEN, l):
                    ln_input = zip(imageSequence[i])
                    temp_image = cv2.cvtColor(cv2.imread(fileName + ''.join(ln_input[0])), cv2.COLOR_BGR2YUV)
                    temp_image = cv2.resize(temp_image, (WIDTH, R_HEIGHT))
                    temp_image = temp_image[-HEIGHT:, 0:WIDTH]                
                    temp_image, u, v = cv2.split(temp_image)
                    img_mean = np.mean(temp_image/255)
                    temp_image = (temp_image/255)-img_mean              
                    inputs.append(temp_image)
            angle = (float(''.join(ln_target[1])))
            targets.append(angle)
            for i in range(k-STEER,k):
                t = zip(imageSequence[i])
                steering.append(float(''.join(t[1])))
            if k == batch_end-1:
                X1 = np.array(inputs)
                X1 = np.reshape(X1, (batch_size,SEQ_LEN,HEIGHT,WIDTH,LEFT_CONTEXT,CHANNELS))
                X2 = np.array(steering)
                X2 = np.reshape(X2, (batch_size, STEER, 1))
                Y = np.array(targets)
                Y = np.transpose(Y)
                yield ([X1, X2], Y)  #a tuple with two numpy arrays with batch_size samples 
                batch_start += batch_size  
                batch_end += batch_size
            if batch_end >= steps*batch_size:
                batch_start = depth
                batch_end = batch_size+depth

def read_csv(filename):
    with open(filename, 'r') as f:
        next(f)
        lines = [ln.strip().split(",")[-8:-3] for ln in f]
        return lines
    
def process_csv_images(filename):
    sum_f = np.float128([0.0] * OUTPUT_DIM)
    sum_sq_f = np.float128([0.0] * OUTPUT_DIM)
    lines = read_csv(filename)
    # leave val% for validation
    sequence = []
    for ln in lines:
        if ln[0] == "center_camera": 
            sequence.append(ln[1:])
            steering = np.float32(ln[2])
            sum_f += steering
            sum_sq_f += steering * steering
    mean = sum_f / len(sequence)
    var = sum_sq_f / len(sequence) - mean * mean
    std = np.sqrt(var)
    print len(sequence)
    return (sequence), (mean, std)
    
def trainValidSeq(filename, valPercentage):
    (totalSequence), (mean, std) = process_csv_images(filename)
    valLen = int(valPercentage*len(totalSequence))
    valid_seq = totalSequence[0:valLen]
    train_seq = totalSequence[valLen+1:]
    return (train_seq, valid_seq), (mean, std)
    
def process_csv_seq(filename, valPercentage):
    with open(filename, 'r') as f:
        next(f)
        lines = [ln.strip().split(",")[1:4] for ln in f]
    # leave val% for validation
    steeringTrain = []
    speedTrain = []
    steeringVal = []
    speedVal = []
    for ln in lines:
        steeringTrain.append(np.float32(ln[0]))
        speedTrain.append(np.float32(ln[2]))
    valLen = int(valPercentage*len(steeringTrain))
    steeringVal = steeringTrain[0:valLen]
    speedVal = speedTrain[0:valLen]
    steeringTrain = steeringTrain[:-valLen]
    speedTrain = speedTrain[:-valLen]
    return (steeringTrain, speedTrain), (steeringVal, speedVal)
    
#Set the path file here    
trainFile = "/Dataset/train/"
testFile = "/Dataset/test/"

(train_seq, valid_seq), (mean, std) = trainValidSeq(trainFile + "interpolated.csv", 0.15) # concatenated interpolated.csv from rosbags 
(test_seq), (testMean, testStd) = process_csv_images(filename=testFile + "interpolated.csv") # interpolated.csv for testset filled with dummy values
(steeringTrain, speedTrain),(steeringVal, speedVal) = process_csv_seq(trainFile + "steering.csv", 0.15)
steeringTrain = steeringTrain[55:]
speedTrain = speedTrain[55:]
indexes = list(range(LEFT_CONTEXT + SEQ_LEN,len(train_seq)))
np.random.shuffle(indexes)

def model():    
    visibleImage = Input(shape=(SEQ_LEN, HEIGHT, WIDTH, LEFT_CONTEXT, CHANNELS), batch_shape=(BATCH_SIZE, SEQ_LEN, HEIGHT, WIDTH, LEFT_CONTEXT, CHANNELS), name='imageInput')    
    norm0 = TimeDistributed(BatchNormalization(), name='batch_normalization_1', trainable=cnnTrain)(visibleImage)    

    conv11 = TimeDistributed(Conv3D(24, (5,5,1), strides=(2, 2, 1) , padding='same', activation='elu', data_format='channels_last'), name='conv3d_1', trainable=cnnTrain)(norm0)
    norm1 = TimeDistributed(BatchNormalization(), name='batch_normalization_2', trainable=cnnTrain)(conv11)    

    conv12 = TimeDistributed(Conv3D(36, (5,5,1), strides=(2, 2, 1) , padding='same', activation='elu'), name='conv3d_2', trainable=cnnTrain)(norm1)
    norm2 = TimeDistributed(BatchNormalization(), name='batch_normalization_3', trainable=cnnTrain)(conv12)

    conv13 = TimeDistributed(Conv3D(48, (5,5,1), strides=(2, 2, 1) , padding='same', activation='elu'), name='conv3d_3', trainable=cnnTrain)(norm2)
    norm3 = TimeDistributed(BatchNormalization(), name='batch_normalization_4', trainable=cnnTrain)(conv13)

    conv14 = TimeDistributed(Conv3D(64, (3,3,1), strides=(1, 1, 1) , padding='same', activation='elu'), name='conv3d_4', trainable=cnnTrain)(norm3)
    norm4 = TimeDistributed(BatchNormalization(), name='batch_normalization_5', trainable=cnnTrain)(conv14)

    conv15 = TimeDistributed(Conv3D(64, (3,3,1), strides=(1, 1, 1) , padding='same', activation='elu'), name='conv3d_5', trainable=cnnTrain)(norm4)
    norm5 = TimeDistributed(BatchNormalization(), name='batch_normalization_6', trainable=cnnTrain)(conv15)

    flat1 = TimeDistributed(Flatten(), name='flatten_1')(norm5)
    
    lstm1 = LSTM(32, dropout=0.25, return_sequences=True, stateful=Stateful, trainable=lstmTrain)(flat1)
    lstm2 = LSTM(16, dropout=0.25, stateful=Stateful, trainable=lstmTrain)(lstm1)
    
    drop1 = Dropout(0.25)(lstm2)
    
    visibleSteering = Input(shape=(STEER,1), batch_shape=(BATCH_SIZE, STEER, 1), name='steeringInput')
    conv21 = Conv1D(8, 2, strides=2, activation='relu')(visibleSteering)
    conv22 = Conv1D(16, 2, strides=2, activation='relu')(conv21)
    conv23 = Conv1D(32, 2, strides=2, activation='relu')(conv22)
    flat2 = Flatten()(conv23)
    
    merge = concatenate([drop1, flat2], axis=-1)
    print('models concatenated', merge._keras_shape)
    
    hidden1 = Dense(256, activation='elu', name='d11', trainable=lstmTrain)(merge)
    drop6 = Dropout(0.25)(hidden1)
   
    hidden2 = Dense(128, activation='elu', name='d12', trainable=lstmTrain)(drop6)
    drop7 = Dropout(0.25)(hidden2)

    hidden3 = Dense(64, activation='elu', name='d13', trainable=lstmTrain)(drop7)
    drop8 = Dropout(0.5)(hidden3)

    hidden4 = Dense(16, activation='elu', name='d14')(drop8)

    out = Dense(1, activation='linear', name='d15')(hidden4)    
    model = Model(inputs=[visibleImage, visibleSteering], outputs=out)
    opt = keras.optimizers.Adadelta(lr=1, decay=1e-6)
    model.compile(optimizer=opt, loss="mse", metrics=['mae'])
    with open('summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    print('Model is created and compiled..')  
    return model

Shuffle = "True"
Norm = "True"
Diraction = "Left"
Crop = "True"
Optimizer = 'adadelta'
Stateful = False
cnnTrain = False
lstmTrain = True
noise = 0.2

model = model()
#Load weights of pre-trained model
model.load_weights('saved_model5.1_5747_4_5_3_10_True_True_adadelta_1_Left_True_False_False100.2.h5', by_name = True)
print('Weights Loaded Succesfully')

stepsPerEpoch =int(len(train_seq)/(BATCH_SIZE))
stepsVal = int(len(valid_seq)/(BATCH_SIZE))
stepsTest = int(len(test_seq)/(BATCH_SIZE))
tf_session = keras.backend.get_session()
tf_graph = tf.get_default_graph()
tgen = dataGen(trainFile, train_seq, batch_size=BATCH_SIZE,
               steps=stepsPerEpoch,
               session=tf_session,
               graph=tf_graph,
               model=model)
vgen = valGen(trainFile, valid_seq, batch_size=BATCH_SIZE,
              steps=stepsVal,
              session=tf_session,
              graph=tf_graph,
              model=model)
testgen = valGen(testFile, test_seq, batch_size=BATCH_SIZE,
              steps=stepsTest,
              session=tf_session,
              graph=tf_graph,
              model=model)
meta = '5.1_'+str(stepsPerEpoch)+'_'+str(EPOCHS+4)+'_'+str(BATCH_SIZE)+'_'+str(LEFT_CONTEXT)+'_'+str(SEQ_LEN)+'_'+str(Shuffle)+'_'+str(Norm)+'_'+str(Optimizer)+'_'+str(CHANNELS)+"_"+str(Diraction)+"_"+str(Crop)+"_"+str(Stateful)+"_"+str(cnnTrain)+str(STEER)+str(noise)

history = model.fit_generator(tgen,
                     validation_data=testgen,
                     validation_steps=stepsTest,
                     steps_per_epoch=stepsPerEpoch,
                     epochs=EPOCHS,
                     verbose=1,                     
                     shuffle='False',
                     initial_epoch=0,
                     )

np.savetxt("loss_"+meta+".csv", history.history['loss'], delimiter=" ")
# serialize model to JSON
model_json = model.to_json()
with open("model"+meta+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_"+meta+".h5")
print("Saved model to disk")

model.save("saved_model"+meta+".h5")

#Create model for online predictions
def pred_model():   
    visibleImage = Input(shape=(SEQ_LEN, HEIGHT, WIDTH, LEFT_CONTEXT, CHANNELS), batch_shape=(1, SEQ_LEN, HEIGHT, WIDTH, LEFT_CONTEXT, CHANNELS), name='imageInput')    
    norm0 = TimeDistributed(BatchNormalization(), name='batch_normalization_1', trainable=cnnTrain)(visibleImage)    

    conv11 = TimeDistributed(Conv3D(24, (5,5,1), strides=(2, 2, 1) , padding='same', activation='elu', data_format='channels_last'), name='conv3d_1', trainable=cnnTrain)(norm0)
    norm1 = TimeDistributed(BatchNormalization(), name='batch_normalization_2', trainable=cnnTrain)(conv11)    

    conv12 = TimeDistributed(Conv3D(36, (5,5,1), strides=(2, 2, 1) , padding='same', activation='elu'), name='conv3d_2', trainable=cnnTrain)(norm1)
    norm2 = TimeDistributed(BatchNormalization(), name='batch_normalization_3', trainable=cnnTrain)(conv12)

    conv13 = TimeDistributed(Conv3D(48, (5,5,1), strides=(2, 2, 1) , padding='same', activation='elu'), name='conv3d_3', trainable=cnnTrain)(norm2)
    norm3 = TimeDistributed(BatchNormalization(), name='batch_normalization_4', trainable=cnnTrain)(conv13)

    conv14 = TimeDistributed(Conv3D(64, (3,3,1), strides=(1, 1, 1) , padding='same', activation='elu'), name='conv3d_4', trainable=cnnTrain)(norm3)
    norm4 = TimeDistributed(BatchNormalization(), name='batch_normalization_5', trainable=cnnTrain)(conv14)

    conv15 = TimeDistributed(Conv3D(64, (3,3,1), strides=(1, 1, 1) , padding='same', activation='elu'), name='conv3d_5', trainable=cnnTrain)(norm4)
    norm5 = TimeDistributed(BatchNormalization(), name='batch_normalization_6', trainable=cnnTrain)(conv15)

    flat1 = TimeDistributed(Flatten(), name='flatten_1')(norm5)
    
    lstm1 = LSTM(32, dropout=0.25, return_sequences=True, stateful=Stateful, trainable=lstmTrain)(flat1)
    lstm2 = LSTM(16, dropout=0.25, stateful=Stateful, trainable=lstmTrain)(lstm1)
    
    drop1 = Dropout(0.25)(lstm2)
    
    visibleSteering = Input(shape=(STEER,1), batch_shape=(1, STEER, 1), name='steeringInput')
    conv21 = Conv1D(8, 2, strides=2, activation='relu')(visibleSteering)
    conv22 = Conv1D(16, 2, strides=2, activation='relu')(conv21)
    conv23 = Conv1D(32, 2, strides=2, activation='relu')(conv22)
    flat2 = Flatten()(conv23)
    
    merge = concatenate([drop1, flat2], axis=-1)
    print('models concatenated', merge._keras_shape)
    
    hidden1 = Dense(256, activation='elu', name='d11', trainable=lstmTrain)(merge)
    drop6 = Dropout(0.25)(hidden1)
   
    hidden2 = Dense(128, activation='elu', name='d12', trainable=lstmTrain)(drop6)
    drop7 = Dropout(0.25)(hidden2)

    hidden3 = Dense(64, activation='elu', name='d13', trainable=lstmTrain)(drop7)
    drop8 = Dropout(0.5)(hidden3)

    hidden4 = Dense(16, activation='elu', name='d14')(drop8)

    out = Dense(1, activation='linear', name='d15')(hidden4)    
    model = Model(inputs=[visibleImage, visibleSteering], outputs=out)
    opt = keras.optimizers.Adadelta(lr=1, decay=1e-6)
    model.compile(optimizer=opt, loss="mse", metrics=['mae'])
    with open('summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    print('Model is created and compiled..')  
    return model

model = pred_model()
model.load_weights("model_"+meta+".h5")

#Model prediction
depth = LEFT_CONTEXT + SEQ_LEN
p = []
print('Predict...')
steering = []
for i in range(depth - STEER, depth):
    t = zip(test_seq[i])
    steering.append(random.uniform(-0.2,0.2))
for k in range(depth,len(test_seq)):         
    targets = []
    inputs = []
    ln_target = zip(test_seq[k-1])
    for l in range(k-LEFT_CONTEXT, k):
        for i in range(l - SEQ_LEN, l):
            ln_input = zip(test_seq[i])
            temp_image = cv2.cvtColor(cv2.imread(testFile + ''.join(ln_input[0])), cv2.COLOR_BGR2YUV)
            temp_image = cv2.resize(temp_image, (WIDTH, R_HEIGHT))
            temp_image = temp_image[-HEIGHT:, 0:WIDTH]                
            temp_image, u, v = cv2.split(temp_image)
            img_mean = np.mean(temp_image/255)
            temp_image = (temp_image/255)-img_mean              
            inputs.append(temp_image)
    
    X1 = np.array(inputs)
    X1 = np.reshape(X1, (1,SEQ_LEN,HEIGHT,WIDTH,LEFT_CONTEXT,CHANNELS))
    X2 = np.array(steering)
    X2 = np.reshape(X2, (1, STEER, 1))
    pred = model.predict([X1, X2]) 
    p.append(float(pred))
    steering.append(float(pred))
    steering = steering[-STEER:]
