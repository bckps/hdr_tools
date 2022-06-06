# import numpy as np
import h5py
import cupy as np
import matplotlib.pyplot as plt
import numpy
import os
from hdr_tools.etcfile.folder import data_folder_path

"""
dimx=256;
dimy=256;
dimt=1024;

c=299792458; %m/sec
frame_length =  0.008993773740; %m -> 30psec 
pulse_length = dimt; %  1024frame = 30.72nsec
tap_num = 4; %1画素のタップの数
subframe_num = 32; %32subframe * 32frame = 1024frame
resize_scale = 1;
"""

data_path_list = data_folder_path(r'C:\Users\919im\Documents\local-香川研\makeToFDataset\DeepToFDataset')

for data_path in data_path_list:
    f1 = h5py.File(os.path.join(data_path['1st']),'r')
    f2 = h5py.File(os.path.join(data_path['2nd']),'r')
    f3 = h5py.File(os.path.join(data_path['3rd']),'r')
    print(data_path['folder_path'])
    print(list(f1.keys()))
    print(list(f2.keys()))
    print(list(f3.keys()))
    data = f1.get('imgVolume1')
    data2 = f2.get('imgVolume2')
    data3 = f3.get('imgVolume3')

    data = np.array(data) # For converting to a NumPy array
    data2 = np.array(data2)
    data3 = np.array(data3)

    array_40MHz_sin = (np.sin(np.linspace(0, np.pi * 417*2 / 417, num=417*2)) + 1) / 2
    array_70MHz_sin = (np.sin(np.linspace(0, np.pi * 238*2 / 238, num=238*2)) + 1) / 2


    array_40MHz_sin = np.concatenate((array_40MHz_sin,array_40MHz_sin,array_40MHz_sin,array_40MHz_sin))
    array_70MHz_sin = np.concatenate((array_70MHz_sin,array_70MHz_sin,array_70MHz_sin,array_70MHz_sin,array_70MHz_sin,array_70MHz_sin))

    sin40_length = 417*8
    sin40_half_len = 417*4
    sin70_length = 238*12
    sin70_half_len = 238*6
    array_40MHz_sin = np.broadcast_to(array_40MHz_sin, (256,256,sin40_length)).transpose((2,0,1))
    array_70MHz_sin = np.broadcast_to(array_70MHz_sin, (256,256,sin70_length)).transpose((2,0,1))


    sense_40MHz_img = np.zeros((4,256,256))
    sense_70MHz_img = np.zeros((4,256,256))

    #40MHz:Q1
    for i in range(417):
        sense_40MHz_img[0, :, :] += np.sum(data[:, 0, :, :] * array_40MHz_sin[sin40_half_len-i:sin40_half_len-i+1024,:,:]
                                           +data2[:, 0, :, :] * array_40MHz_sin[sin40_half_len-i:sin40_half_len-i+1024,:,:]
                                           +data3[:, 0, :, :] * array_40MHz_sin[sin40_half_len-i:sin40_half_len-i+1024,:,:], axis=0)
    #40MHz:Q2
    for i in range(417, 417*2):
        sense_40MHz_img[1, :, :] += np.sum(data[:, 0, :, :] * array_40MHz_sin[sin40_half_len-i:sin40_half_len-i+1024,:,:]
                                           +data2[:, 0, :, :] * array_40MHz_sin[sin40_half_len-i:sin40_half_len-i+1024,:,:]
                                           +data3[:, 0, :, :] * array_40MHz_sin[sin40_half_len-i:sin40_half_len-i+1024,:,:], axis=0)
    #40MHz:Q3
    for i in range(417//2, 417*3//2):
        sense_40MHz_img[2, :, :] += np.sum(data[:, 0, :, :] * array_40MHz_sin[sin40_half_len-i:sin40_half_len-i+1024,:,:]
                                           +data2[:, 0, :, :] * array_40MHz_sin[sin40_half_len-i:sin40_half_len-i+1024,:,:]
                                           +data3[:, 0, :, :] * array_40MHz_sin[sin40_half_len-i:sin40_half_len-i+1024,:,:], axis=0)
    #40MHz:Q4
    for i in np.concatenate((np.arange(417//2),np.arange(417*3//2,417*2))):
        sense_40MHz_img[3, :, :] += np.sum(data[:, 0, :, :] * array_40MHz_sin[sin40_half_len-i:sin40_half_len-i+1024,:,:]
                                           +data2[:, 0, :, :] * array_40MHz_sin[sin40_half_len-i:sin40_half_len-i+1024,:,:]
                                           +data3[:, 0, :, :] * array_40MHz_sin[sin40_half_len-i:sin40_half_len-i+1024,:,:], axis=0)

    #70MHz:Q1
    for i in range(238):
        sense_70MHz_img[0, :, :] += np.sum(data[:, 0, :, :] * array_70MHz_sin[sin70_half_len-i:sin70_half_len-i+1024,:,:]
                                           +data2[:, 0, :, :] * array_70MHz_sin[sin70_half_len-i:sin70_half_len-i+1024,:,:]
                                           +data3[:, 0, :, :] * array_70MHz_sin[sin70_half_len-i:sin70_half_len-i+1024,:,:], axis=0)
    #70MHz:Q2
    for i in range(238, 238*2):
        sense_70MHz_img[1, :, :] += np.sum(data[:, 0, :, :] * array_70MHz_sin[sin70_half_len-i:sin70_half_len-i+1024,:,:]
                                           +data2[:, 0, :, :] * array_70MHz_sin[sin70_half_len-i:sin70_half_len-i+1024,:,:]
                                           +data3[:, 0, :, :] * array_70MHz_sin[sin70_half_len-i:sin70_half_len-i+1024,:,:], axis=0)
    #70MHz:Q3
    for i in range(238//2, 238*3//2):
        sense_70MHz_img[2, :, :] += np.sum(data[:, 0, :, :] * array_70MHz_sin[sin70_half_len-i:sin70_half_len-i+1024,:,:]
                                           +data2[:, 0, :, :] * array_70MHz_sin[sin70_half_len-i:sin70_half_len-i+1024,:,:]
                                           +data3[:, 0, :, :] * array_70MHz_sin[sin70_half_len-i:sin70_half_len-i+1024,:,:], axis=0)
    #70MHz:Q4
    for i in np.concatenate((np.arange(238//2),np.arange(238*3//2,238*2))):
        sense_70MHz_img[3, :, :] += np.sum(data[:, 0, :, :] * array_70MHz_sin[sin70_half_len-i:sin70_half_len-i+1024,:,:]
                                           +data2[:, 0, :, :] * array_70MHz_sin[sin70_half_len-i:sin70_half_len-i+1024,:,:]
                                           +data3[:, 0, :, :] * array_70MHz_sin[sin70_half_len-i:sin70_half_len-i+1024,:,:], axis=0)


    #depth_map
    depth_1024 = np.argmax(data[:, 0, :, :],axis=0) / 1024


    filename = data_path['folder_path'].split('\\')[-2]+'_'+data_path['folder_path'].split('\\')[-1]+'.npz'
    with open(filename, 'wb') as f:
        np.savez(f, sense_40MHz_img.T, sense_70MHz_img.T, depth_1024.T)