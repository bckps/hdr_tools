import numpy as np
import h5py
# import cupy as np
import matplotlib.pyplot as plt
import numpy

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


def plot_data(cap_data,sense_40MHz_img, sense_70MHz_img, depth_1024):

    plt.figure()
    plt.subplot(421)
    plt.imshow(cap_data[0,:,:].T)
    plt.axis('off')
    plt.title('40MHz:Q1')
    plt.subplot(422)
    plt.imshow(sense_40MHz_img[:,:,0])
    plt.axis('off')
    plt.title('make:40MHz:Q1')
    plt.subplot(423)
    plt.imshow(cap_data[1,:,:].T)
    plt.axis('off')
    plt.title('40MHz:Q2')
    plt.subplot(424)
    plt.imshow(sense_40MHz_img[:,:,1])
    plt.axis('off')
    plt.title('make:40MHz:Q2')

    plt.subplot(425)
    plt.imshow(cap_data[2,:,:].T)
    plt.axis('off')
    plt.title('70MHz:Q1')
    plt.subplot(426)
    plt.imshow(sense_70MHz_img[:,:,0])
    plt.axis('off')
    plt.title('make:70MHz:Q1')
    plt.subplot(427)
    plt.imshow(cap_data[3,:,:].T)
    plt.axis('off')
    plt.title('70MHz:Q2')
    plt.subplot(428)
    plt.imshow(sense_70MHz_img[:,:,1])
    plt.axis('off')
    plt.title('make:70MHz:Q2')
    plt.show()

    plt.figure()
    plt.imshow(depth_1024)
    plt.show()

data = np.load('bathroom_1.npz', 'r')
print(data.files)
sense_40MHz_img = data.get('arr_0')
sense_70MHz_img = data.get('arr_1')
depth_1024 = data.get('arr_2')
plt.plot(response_img[64,64,:,0], label="test")
# プロット表示(設定の反映)
plt.show()
print(sense_40MHz_img.shape)


with h5py.File(r"C:\Users\919im\Documents\local-香川研\makeToFDataset\dataset_shizuoka\bathroom\bathroom_1.mat",'r') as imgs:
    cap_img = numpy.array(imgs['cap'])
    cap_ideal_img = numpy.array(imgs['cap_ideal'])
    depth_img = numpy.array(imgs['depth_1024'])


plot_data(cap_img, sense_40MHz_img, sense_70MHz_img, depth_1024)

