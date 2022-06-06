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
    dataname = r'v3-livingroom-train/00002/hdrs'
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

    first_response = response_img.transpose(1, 0, 2).copy()
    # first_response = response_img.copy()

    for i,b2 in enumerate(b2_list):
        img = np.array(cv2.imread(os.path.join(folder_path, b2), flags=cv2.IMREAD_ANYDEPTH))
        # print(img.shape)
        response_img[:,i,:] += img[:width,:,2]

    for i,b3 in enumerate(b3_list):
        img = np.array(cv2.imread(os.path.join(folder_path, b3), flags=cv2.IMREAD_ANYDEPTH))
        # print(img.shape)
        response_img[:,i,:] += img[:width,:,2]

    # for i,b4 in enumerate(b4_list):
    #     img = np.array(cv2.imread(os.path.join(folder_path, b4), flags=cv2.IMREAD_ANYDEPTH))
    #     # print(img.shape)
    #     response_img[:,i,:] += img[:width,:,2]

    response_img = response_img.transpose(1,0,2)

    y, x = 90,195
    # y,x = 65, 190
    # y, x = 90,150
    timeseq = np.linspace((30e-3),(30e-3)*2221, num=2220)
    # count = 0
    # for i in range(2220):
    #     if 0.00001 < response_img[y,x,i]:
    #         # print(timeseq[i])
    #         count += 1
    #         if 0 < count:
    #             response_img[y, x, i] = 0


    for i in range(response_img.shape[0]):
        for j in range(response_img.shape[1]):
            conv[i, j, :] = np.convolve(response_img[i, j, :], conv_window[:], 'full')
            conved_img[i, j, :] = conv[i, j, :][:2220]

    A0 = np.sum(conved_img[:, :, :740], axis=2)
    A1 = np.sum(conved_img[:, :, 740:1480], axis=2)
    A2 = np.sum(conved_img[:, :, 1480:], axis=2)

    # phase shift depth
    c = 299792458
    eps = 1e-5
    z_coef = c * 22.2 * (10**-9) / 2.0
    depth_from_phase = np.where(A0 < A2, z_coef*((A2-A0)/(A1+A2-2*A0+eps)+1), z_coef*(A1-A2)/(A0+A1-2*A2+eps))

    # depth max is 6.66 [m]
    z_max = c*(30e-12)*2220 / 2.0
    # cflight_index = int(z_max // (c * 30 * (10**-12)))
    # depth_2220 = z_max * np.argmax(response_img[:, :, :cflight_index], axis=2) / 2
    # print(np.max(np.argmax(response_img, axis=2) / response_img.shape[2]))
    # depth_2220 = z_max * np.argmax(response_img, axis=2) / response_img.shape[2]


    # depth_2220 = z_max * np.argmax(response_img, axis=2) / response_img.shape[2]


    # save_npzes(npz_folder_path, dataname, A0, A1, A2, depth_from_phase, depth_2220)


    # accumconv2220 = conved_img[y,x,:].cumsum()/conved_img[y,x,:].sum()#
    accumconv = np.sum(conv, axis=(0,1)).cumsum()/np.sum(conv)
    # arg99conv2220 = np.abs((accumconv2220-0.99)).argmin()#

    arg99conv = np.abs((accumconv - 0.99)).argmin()
    timeseq = np.linspace((30e-3),(30e-3)*2221, num=2220)
    timeseq2960 = np.linspace((30e-3),(30e-3)*2960, num=2959)
    arg666 = np.abs((timeseq2960 - 66.6)).argmin()
    arg222 = np.abs((timeseq2960 - 22.2)).argmin()
    arg444 = np.abs((timeseq2960 - 44.4)).argmin()

    for i in range(height):
        for j in range(width):
            peaks, _ = find_peaks(response_img[i,j,:])
            if len(peaks) == 0:
                first_peak_time[i,j] = timeseq[-1]
                # first_peak_index[i, j] = 2959
                first_peak_index[i, j] = 2200

            else:
                first_peak_time[i,j] = timeseq[peaks[0]]
                first_peak_index[i,j] = int(peaks[0])

    depth_2220 = z_max * first_peak_index / response_img.shape[2]

    ######################################################################

    plt.figure(figsize=(6, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(np.sum(first_response, axis=2), label="test")
    plt.scatter([x], [y], marker='.', c='r',s=20**2)
    plt.colorbar()
    # plt.show()
    plt.subplot(2, 1, 2)
    # plt.hist(first_response.flatten(), bins=1000)
    plt.plot(timeseq,first_response[y,x,:], label="test")
    print(np.nonzero(first_response[y,x, :]))
    print(len(np.sum(first_response, axis=2))-len(np.nonzero(np.sum(first_response, axis=2))))
    print(np.sum(first_response[y,x, :]))
    # plt.plot(first_peak_time[y,x], response_img[y,x,first_peak_index[y,x]], "x")
    plt.xlabel('Time [ns]',fontsize=14)
    plt.ylabel('Pixel value',fontsize=14)
    plt.xticks(fontsize=14)
    plt.savefig(os.path.join(npz_folder_path,f'1st-bounce_({x},{y}).png'))
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(timeseq,response_img[y,x,:], label="test")
    # plt.plot(first_peak_time[y,x], response_img[y,x,first_peak_index[y,x]], "x")
    plt.xlabel('Time [ns]',fontsize=14)
    plt.ylabel('Pixel value',fontsize=14)
    plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    plt.savefig(os.path.join(npz_folder_path, f'inpulse_({x},{y}).png'))
    plt.show()

    plt.figure()
    plt.plot(timeseq,np.append(conv_window,np.zeros(740*2)), label="test")
    # plt.plot(first_peak_time[y,x], response_img[y,x,first_peak_index[y,x]], "x")
    plt.xlabel('Time [ns]',fontsize=14)
    plt.ylabel('Pixel value',fontsize=14)
    plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    plt.savefig(os.path.join(npz_folder_path, f'window_({x},{y}).png'))
    plt.show()


    # plt.figure()
    # plt.plot(timeseq,conved_img[10,110,:], label="conv")
    # plt.xlabel('Time [ns]')
    # plt.ylabel('Pixel value')
    # plt.show()

    plt.figure()
    plt.plot(timeseq2960, accumconv, label="conv")
    plt.xlabel('Time [ns]')
    plt.ylabel('cumulative frequency')
    plt.vlines(timeseq2960[arg99conv], 0, 1, colors='red', linestyle='dashed', linewidth=3)
    plt.vlines(timeseq2960[arg666], 0, 1, colors='blue', linestyle='dashed', linewidth=3)
    plt.savefig(os.path.join(npz_folder_path, f'accumu_({x},{y}).png'))
    plt.show()


    # plt.figure()
    # plt.plot(timeseq, accumconv2220, label="conv")
    # plt.xlabel('Time [ns]')
    # plt.ylabel('cumulative frequency')
    # plt.vlines(timeseq2960[arg99conv], 0, 1, colors='red', linestyle='dashed', linewidth=3)
    # plt.show()

    ############################################################

    plt.figure()
    plt.plot(timeseq2960,conv[y,x,:], label="conv")
    plt.xlabel('Time [ns]',fontsize=14)
    plt.ylabel('Pixel value',fontsize=14)
    plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    upper_bound = np.max(conv[y,x,:])
    plt.vlines(timeseq2960[arg222], 0, upper_bound, colors='blue', linestyle='dashed', linewidth=3)
    plt.vlines(timeseq2960[arg444], 0, upper_bound, colors='blue', linestyle='dashed', linewidth=3)
    plt.vlines(timeseq2960[arg666], 0, upper_bound, colors='blue', linestyle='dashed', linewidth=3)
    plt.savefig(os.path.join(npz_folder_path, f'conved_({x},{y}).png'))
    plt.show()


    # plt.figure()
    # plt.plot(np.linspace((30e-3),(30e-3)*2960, num=2959),conv[y,x,:], label="conv")
    # plt.xlabel('Time [ns]')
    # plt.ylabel('Pixel value')
    # plt.show()

    # plt.subplot(1, 2, 1)


    plt.figure()
    plt.imshow(depth_from_phase, label="test")
    plt.colorbar()
    # plt.scatter([209,210,212,214,216], [58,60,60,60,60], marker='.', c='r')
    # plt.scatter([190,190,190,190], [60, 65, 70, 75], marker='.', c='r')
    plt.scatter([x], [y], marker='.', c='r',s=20**2)
    plt.savefig(os.path.join(npz_folder_path, 'phase_depth.png'))
    plt.show()


    # print(depth_2220[60, 190])
    # print(depth_2220[65, 190])
    # print(depth_2220[70, 190])
    # print(depth_2220[75, 190])

    plt.figure()
    plt.imshow(depth_2220, label="test")
    plt.colorbar()
    # plt.scatter([209,210,212,214,216], [58,60,60,60,60], marker='.', c='r')
    # plt.scatter([190,190,190,190], [60, 65, 70, 75], marker='.', c='r')
    plt.scatter([x], [y], marker='.', c='r',s=20**2)
    plt.savefig(os.path.join(npz_folder_path, 'ground_truth_depth3th.png'))
    plt.show()