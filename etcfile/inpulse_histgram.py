import cv2
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation

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
    return b1_list, b2_list, b3_list

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
    dataname = r'bathroom'
    # dataname = r'bathroom-smal2'
    width, height = 258, 258
    # width, height = 256, 256
    time_span = 1024
    # time_span = 2220


    folder_path = os.path.join(r'/home/saijo/labwork/PythonSandbox/hdr_process/hdrs',dataname)
    npz_folder_path = os.path.join(r'/home/saijo/labwork/PythonSandbox/hdr_process/npz', dataname)
    os.makedirs(npz_folder_path, exist_ok=True)
    b1_list, b2_list, b3_list = collect_inpulsefile(folder_path)




    response_img = np.zeros((height,width,time_span))
    conv_window = np.ones((740))
    conved_img = np.zeros((height,width,time_span))

    for i,b1 in enumerate(b1_list):
        img = np.array(cv2.imread(os.path.join(folder_path, b1), flags=cv2.IMREAD_ANYDEPTH))
        # print(img.shape)
        response_img[:,i,:] += img[:,:,2]

    for i,b2 in enumerate(b2_list):
        img = np.array(cv2.imread(os.path.join(folder_path, b2), flags=cv2.IMREAD_ANYDEPTH))
        # print(img.shape)
        response_img[:,i,:] += img[:,:,2]

    for i,b3 in enumerate(b3_list):
        img = np.array(cv2.imread(os.path.join(folder_path, b3), flags=cv2.IMREAD_ANYDEPTH))
        # print(img.shape)
        response_img[:,i,:] += img[:,:,2]



    plt.subplot(1, 2, 1)
    plt.plot(response_img[94,32,:], label="test")
    plt.figure()
    plt.plot(conved_img[94,32,:], label="conv")
    plt.show()

    # resp_max_map = np.argmax(response_img, axis=2)
    # print(np.max(resp_max_map))
    # print(resp_max_map.shape)
    # histogram, bin_edges = np.histogram(resp_max_map, bins=np.max(resp_max_map))
    # plt.figure()
    # plt.xlabel("time max values")
    # plt.ylabel("# pixels")
    #
    # plt.plot(bin_edges[0:-1], histogram)  # <- or here
    # plt.show()

    print(response_img.shape)
    resp_one = np.where(response_img > 0, 1, 0)
    histogram = np.sum(resp_one, axis=(0, 1))
    xaxis = np.linspace(0,time_span-1, num=time_span) * 30 * 1e-3



    plt.figure()
    plt.xlabel("time [ns]")
    plt.ylabel("# pixels")

    plt.plot(xaxis, histogram)  # <- or here
    plt.show()

    # plt.subplot(1, 2, 1)
    # plt.imshow(depth_from_phase.T, label="test")
    # plt.figure()
    # plt.imshow(depth_2220.T, label="test")
    # plt.show()