import cv2
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import os, json
import matplotlib.animation as animation
from scipy.signal import find_peaks
from scipy.io import savemat

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


def collect_panaToF_inpulsefile(folder_path):
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
        elif 'depth' in file:
            depth_hdr_path = file
    # return b1_list, b2_list, b3_list, depth_hdr_path
    return b1_list, b2_list, b3_list, None


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
    dataname = 'v3-bathroom-train'

    #シミュレーションしたデータのフォルダを指定
    folder_path = os.path.join(r'/home/saijo/labwork/simulator_origun/scene-results', dataname)
    #時間分解後のデータの保存場所
    npz_folder_path = os.path.join(r'/home/saijo/labwork/PythonSandbox/hdr_process/npz', dataname)
    os.makedirs(npz_folder_path, exist_ok=True)

    data_files = [f for f in os.listdir(folder_path) if f.isdecimal()]
    data_files.sort()
    for file in data_files:
        print(file)
        hdrfile_path = os.path.join(folder_path, file, 'hdrs')
        b1_list, b2_list, b3_list, depth_hdr_path = collect_panaToF_inpulsefile(hdrfile_path)
        npzdata_path = os.path.join(r'/home/saijo/labwork/PythonSandbox/hdr_process/npz', dataname, file)
        os.makedirs(npzdata_path, exist_ok=True)

        #################################################################
        # Declaration
        #################################################################
        height, width = 256, 256
        response_img = np.zeros((height,width,2220))
        conv_window = np.ones((740))
        conved_img = np.zeros((height,width,2220))
        conv = np.zeros((height,width,2959))

        timeseq = np.linspace((30e-3),(30e-3)*2221, num=2220)


        #################################################################
        # Read hdr files
        #################################################################
        for i,b1 in enumerate(b1_list):
            img = np.array(cv2.imread(os.path.join(hdrfile_path, b1), flags=cv2.IMREAD_ANYDEPTH))
            response_img[:,i,:] += img[:width,:,2]

        first_response = np.sum(response_img, axis=2).transpose(1, 0)
        one_responces = response_img.transpose(1, 0, 2).copy()
        print(len(first_response.flatten()) - np.count_nonzero(first_response.flatten()))

        for i,b2 in enumerate(b2_list):
            img = np.array(cv2.imread(os.path.join(hdrfile_path, b2), flags=cv2.IMREAD_ANYDEPTH))
            response_img[:,i,:] += img[:width,:,2]

        second_response = np.sum(response_img, axis=2).transpose(1, 0)-first_response
        two_responces = response_img.transpose(1, 0, 2).copy()

        for i,b3 in enumerate(b3_list):
            img = np.array(cv2.imread(os.path.join(hdrfile_path, b3), flags=cv2.IMREAD_ANYDEPTH))
            response_img[:,i,:] += img[:width,:,2]

        #################################################################
        # GT calclation from 1st-bounce. Take care of 1st-bounce existance.
        #################################################################
        # Transpose xy 
        response_img = response_img.transpose(1, 0, 2)
        c = 299792458
        z_max = c*(30e-12)*2220 / 2.0

        first_nonzero_time = np.zeros((height, width))
        first_nonzero_index = np.zeros((height, width), dtype=np.int64)
        for i in range(height):
            for j in range(width):
                nonzero = np.argmax(one_responces[i, j, :])
                if nonzero == 0:
                    first_nonzero_time[i, j] = timeseq[-1]
                else:
                    first_nonzero_time[i, j] = timeseq[nonzero]
                    first_nonzero_index[i, j] = nonzero
        depth_2220_nz = z_max * first_nonzero_index / one_responces.shape[2]


        #################################################################
        # Calculate light effect by convolution.
        #################################################################
        first_nonzero_time = np.zeros((height, width))
        first_nonzero_index = np.zeros((height, width), dtype=np.int64)

        for i in range(response_img.shape[0]):
            for j in range(response_img.shape[1]):
                conv[i, j, :] = np.convolve(response_img[i, j, :], conv_window[:], 'full')
                conved_img[i, j, :] = conv[i, j, :][:2220]


        A0 = np.sum(conved_img[:, :, :740], axis=2)
        A1 = np.sum(conved_img[:, :, 740:1480], axis=2)
        A2 = np.sum(conved_img[:, :, 1480:], axis=2)

        # センサデータの光の強度を(0, 1)→(-1, 1)に正規化する
        max_pixel = np.max([A0, A1, A2])
        offset = 0.5
        scale = 2

        A0 = (A0 / max_pixel - offset) * scale
        A1 = (A1 / max_pixel - offset) * scale
        A2 = (A2 / max_pixel - offset) * scale

        # phase shift depth
        c = 299792458
        p_offset = 0     # p_offset is minimized for L1 error in training data.
        eps = 1e-8      # Avoid zero division.
        z_coef = c * 22.2 * (10**-9) / 2.0
        depth_from_phase = np.where(A0 < A2, z_coef*((A2-A0)/(A1+A2-2*A0+eps)+1), z_coef*(A1-A2)/(A0+A1-2*A2+eps))
        depth_from_phase = np.where(depth_from_phase - p_offset < 0, 0, depth_from_phase - p_offset)

        save_npzes(npzdata_path, file+'-'+dataname, A0, A1, A2, depth_from_phase, depth_2220_nz)


