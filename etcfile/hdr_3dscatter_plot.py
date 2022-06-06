import cv2
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
from scipy.signal import find_peaks
import time
"""
dimx=256;
dimy=256;(255)
dimt=2220;

c=299792458; %m/sec
frame_length =  0.008993773740; %m -> 30psec 
66.6ns = 3*22.2ns = 3*(740*30ps)
dimt 2220 = 3(windows) * 740(time window width)
resize_scale = 1;
"""


def collect_inpulsefile(folder_path):
    files = os.listdir(folder_path)
    files.sort()
    b1_list = []
    b2_list = []
    b3_list = []
    b4_list = []
    for file in files:
        file_bounce = file.split('_')
        if len(file_bounce) < 2:
            continue
        if file_bounce[1] == 'b01':
            b1_list.append(file)
        elif file_bounce[1] == 'b02':
            b2_list.append(file)
        elif file_bounce[1] == 'b03':
            b3_list.append(file)
        elif file_bounce[1] == 'b04':
            b4_list.append(file)
    return b1_list, b2_list, b3_list, b4_list



def video_show(response_img):
    fig = plt.figure()
    ims = []
    for i in range(2220):  # bunny
        im = plt.imshow(np.transpose(np.squeeze(response_img[:, :, i]), (1, 0)) / np.max(response_img[:, :, i]),
                        animated=True)
        ims.append([im])  # グラフを配列 ims に追加
    ani = animation.ArtistAnimation(fig, ims, blit=True, interval=3, repeat_delay=10)
    plt.show()

def save_npzes(npz_folder_path, dataname, A0, A1, A2, depth_from_phase, depth_2220):
    # panasonic ToF sensor output
    filenameA0 = os.path.join(npz_folder_path, dataname + '-A0.npz')
    filenameA1 = os.path.join(npz_folder_path, dataname + '-A1.npz')
    filenameA2 = os.path.join(npz_folder_path, dataname + '-A2.npz')
    filenamePhaseDepth = os.path.join(npz_folder_path, dataname + '-phase-depth.npz')
    filenameGroundTruth = os.path.join(npz_folder_path, dataname + '-ground-truth.npz')

    # data save by npz file
    with open(filenameA0, 'wb') as f:
        np.savez(f, A0)
    with open(filenameA1, 'wb') as f:
        np.savez(f, A1)
    with open(filenameA2, 'wb') as f:
        np.savez(f, A2)
    with open(filenamePhaseDepth, 'wb') as f:
        np.savez(f, depth_from_phase)
    with open(filenameGroundTruth, 'wb') as f:
        np.savez(f, depth_2220)

if __name__ == '__main__':
    dataname = r'bedroom-test/00012/hdrs'
    # dataname = r'bathroom-smal2'
    # folder_path = os.path.join(r'/home/saijo/labwork/PythonSandbox/hdr_process/hdrs',dataname)
    folder_path = os.path.join(r'/home/saijo/labwork/simulator_origun/scene-results', dataname)
    npz_folder_path = os.path.join(r'/home/saijo/labwork/PythonSandbox/hdr_process/npz', dataname)
    os.makedirs(npz_folder_path, exist_ok=True)
    b1_list, b2_list, b3_list, b4_list = collect_inpulsefile(folder_path)

    height, width = 256, 256


    response_img = np.zeros((height,width,2220))
    conv_window = np.ones((740))
    conved_img = np.zeros((height,width,2220))
    conv = np.zeros((height,width,2959))
    first_peak_time = np.zeros((height, width))
    first_peak_index = np.zeros((height, width), dtype=np.int64)

    for i,b1 in enumerate(b1_list):
        img = np.array(cv2.imread(os.path.join(folder_path, b1), flags=cv2.IMREAD_ANYDEPTH))
        # print(img.shape)
        response_img[:,i,:] += img[:width,:,2]

    for i,b2 in enumerate(b2_list):
        img = np.array(cv2.imread(os.path.join(folder_path, b2), flags=cv2.IMREAD_ANYDEPTH))
        # print(img.shape)
        response_img[:,i,:] += img[:width,:,2]

    for i,b3 in enumerate(b3_list):
        img = np.array(cv2.imread(os.path.join(folder_path, b3), flags=cv2.IMREAD_ANYDEPTH))
        # print(img.shape)
        response_img[:,i,:] += img[:width,:,2]

    x_list = []
    y_list = []
    t_list = []
    print(np.max(response_img))

    for i in range(width):
        for j in range(height):
            for t in range(1020):
                if 0.001 < response_img[i,j,t]:
                    x_list.append(i)
                    y_list.append(height-j)
                    t_list.append(t * 0.03)
    print(len(x_list))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_list, t_list, y_list)
    ax.set_xlabel('X', fontsize=18)
    ax.set_ylabel('Time [ns]', fontsize=18)
    ax.set_zlabel('Y', fontsize=18)
    plt.show()

    # for i,b4 in enumerate(b4_list):
    #     img = np.array(cv2.imread(os.path.join(folder_path, b4), flags=cv2.IMREAD_ANYDEPTH))
    #     # print(img.shape)
    #     response_img[:,i,:] += img[:width,:,2]



    ######################################################################


