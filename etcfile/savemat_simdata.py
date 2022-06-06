import cv2
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import os, json
import matplotlib.animation as animation
from scipy.signal import find_peaks
from scipy.io import savemat
import argparse

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
    # print(folder_path)
    # print(files)
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



def video_show(response_img):
    fig = plt.figure()
    ims = []
    for i in range(2220):  # bunny
        im = plt.imshow(np.transpose(np.squeeze(response_img[:, :, i]), (1, 0)) / np.max(response_img[:, :, i]),
                        animated=True)
        ims.append([im])  # グラフを配列 ims に追加
    ani = animation.ArtistAnimation(fig, ims, blit=True, interval=3, repeat_delay=10)
    plt.show()

def save_simumat(npz_folder_path, dataname, A0, A1, A2, depth_from_phase, depth_2220):
    # panasonic ToF sensor output
    filenameA0 = os.path.join(npz_folder_path, dataname + '-A0.mat')
    filenameA1 = os.path.join(npz_folder_path, dataname + '-A1.mat')
    filenameA2 = os.path.join(npz_folder_path, dataname + '-A2.mat')
    filenamePhaseDepth = os.path.join(npz_folder_path, dataname + '-phase-depth.mat')
    filenameGroundTruth = os.path.join(npz_folder_path, dataname + '-ground-truth.mat')

    # data save by mat file
    savemat(filenameA0, {"A0":A0})
    savemat(filenameA1, {"A1":A1})
    savemat(filenameA2, {"A2":A2})
    savemat(filenamePhaseDepth, {"depth_from_phase":depth_from_phase})
    savemat(filenameGroundTruth, {"depth_2220":depth_2220})

    # with open(filenameA0, 'wb') as f:
    #     np.savez(f, A0)
    #     savemat("matlab_matrix.mat", mdic)
    # with open(filenameA1, 'wb') as f:
    #     np.savez(f, A1)
    # with open(filenameA2, 'wb') as f:
    #     np.savez(f, A2)
    # with open(filenamePhaseDepth, 'wb') as f:
    #     np.savez(f, depth_from_phase)
    # with open(filenameGroundTruth, 'wb') as f:
    #     np.savez(f, depth_2220)

def save_sub_npzes(npz_folder_path, dataname, f_1st, f_2nd):
    # panasonic ToF sensor output
    filename_1st = os.path.join(npz_folder_path, dataname + '-1st.npz')
    filename_2nd = os.path.join(npz_folder_path, dataname + '-2nd.npz')

    # data save by npz file
    with open(filename_1st, 'wb') as f:
        np.savez(f, f_1st)
    with open(filename_2nd, 'wb') as f:
        np.savez(f, f_2nd)

if __name__ == '__main__':
    dataname = 'v3-livingroom-val'

    parser = argparse.ArgumentParser(description='Convert Time Rendering output to mat format!')
    parser.add_argument('dataname', help='simulation name', default='v3-livingroom-val')
    parser.add_argument('src', help='source folder path', default='scene-results')
    parser.add_argument('dist', help='destination folder path', default='mats')
    args = parser.parse_args()


    #シミュレーションしたデータのフォルダを指定
    # folder_path = os.path.join(r'/home/saijo/labwork/simulator_origun/scene-results', dataname)
    folder_path = os.path.join(args.src, args.dataname)
    #時間分解後のデータの保存場所
    # npz_folder_path = os.path.join(r'/home/saijo/labwork/PythonSandbox/hdr_process/mat', dataname)
    npz_folder_path = os.path.join(args.dist, args.dataname)
    os.makedirs(npz_folder_path, exist_ok=True)

    data_files = [f for f in os.listdir(folder_path) if f.isdecimal()]
    data_files.sort()
    for file in data_files:
        print(file)
        # print(os.path.join(folder_path, dataname, file, 'hdrs'))
        hdrfile_path = os.path.join(folder_path, file, 'hdrs')
        b1_list, b2_list, b3_list, depth_hdr_path = collect_panaToF_inpulsefile(hdrfile_path)
        # depth_hdr_path = os.path.join(folder_path, file, 'hdrs')#r"/home/saijo/labwork/simulator_origun/scene-results/single-plane/00000/hdrs/single-plane-00000_depth.hdr"
        npzdata_path = os.path.join(npz_folder_path, file)
        os.makedirs(npzdata_path, exist_ok=True)


        height, width = 256, 256


        response_img = np.zeros((height,width,2220))
        conv_window = np.ones((740))
        conved_img = np.zeros((height,width,2220))
        conv = np.zeros((height,width,2959))
        first_peak_time = np.zeros((height, width))
        first_peak_index = np.zeros((height, width), dtype=np.int64)

        timeseq = np.linspace((30e-3),(30e-3)*2221, num=2220)
        timeseq2960 = np.linspace((30e-3),(30e-3)*2960, num=2959)

        for i,b1 in enumerate(b1_list):
            img = np.array(cv2.imread(os.path.join(hdrfile_path, b1), flags=cv2.IMREAD_ANYDEPTH))
            # print(img.shape)
            response_img[:,i,:] += img[:width,:,2]
        # print(np.max(response_img))

        first_response = np.sum(response_img, axis=2).transpose(1, 0)
        one_responces = response_img.transpose(1, 0, 2).copy()
        print(len(first_response.flatten()) - np.count_nonzero(first_response.flatten()))

        for i,b2 in enumerate(b2_list):
            img = np.array(cv2.imread(os.path.join(hdrfile_path, b2), flags=cv2.IMREAD_ANYDEPTH))
            # print(img.shape)
            response_img[:,i,:] += img[:width,:,2]

        second_response = np.sum(response_img, axis=2).transpose(1, 0)-first_response
        two_responces = response_img.transpose(1, 0, 2).copy()

        for i,b3 in enumerate(b3_list):
            img = np.array(cv2.imread(os.path.join(hdrfile_path, b3), flags=cv2.IMREAD_ANYDEPTH))
            # print(img.shape)
            response_img[:,i,:] += img[:width,:,2]

        # depth_hdr_img = np.array(cv2.imread(os.path.join(hdrfile_path, depth_hdr_path), flags=cv2.IMREAD_ANYDEPTH))[:257,:257,2]
        # print(depth_hdr_img.dtype)

        response_img = response_img.transpose(1, 0, 2)
        c = 299792458
        z_max = c*(30e-12)*2220 / 2.0

        first_nonzero_time = np.zeros((height, width))
        first_nonzero_index = np.zeros((height, width), dtype=np.int64)
        for i in range(height):
            for j in range(width):
                # nonzero = np.nonzero(response_img[i, j, :])
                nonzero = np.argmax(one_responces[i, j, :])
                # print(nonzero)
                if nonzero == 0:
                    first_nonzero_time[i, j] = timeseq[-1]
                else:
                    first_nonzero_time[i, j] = timeseq[nonzero]
                    first_nonzero_index[i, j] = nonzero
        # depth_2220_nz = z_max * first_nonzero_index / response_img.shape[2]
        depth_2220_nz = z_max * first_nonzero_index / one_responces.shape[2]


        response_img = response_img.transpose(1, 0, 2)
        c = 299792458
        z_max = c*(30e-12)*2220 / 2.0

        first_nonzero_time = np.zeros((height, width))
        first_nonzero_index = np.zeros((height, width), dtype=np.int64)

        response_img = response_img.transpose(1, 0, 2)


        for i in range(response_img.shape[0]):
            for j in range(response_img.shape[1]):
                conv[i, j, :] = np.convolve(response_img[i, j, :], conv_window[:], 'full')
                conved_img[i, j, :] = conv[i, j, :][:2220]

        # print(b1_list)
        print(np.max(response_img))

        A0 = np.sum(conved_img[:, :, :740], axis=2)
        A1 = np.sum(conved_img[:, :, 740:1480], axis=2)
        A2 = np.sum(conved_img[:, :, 1480:], axis=2)

        max_pixel = np.max([A0, A1, A2])
        offset = 0.5
        scale = 2

        A0 = (A0 / max_pixel - offset) * scale
        A1 = (A1 / max_pixel - offset) * scale
        A2 = (A2 / max_pixel - offset) * scale

        # phase shift depth
        c = 299792458
        p_offset = 0     # p_offset is minimized for L1 error in training data.
        eps = 1e-5
        z_coef = c * 22.2 * (10**-9) / 2.0
        depth_from_phase = np.where(A0 < A2, z_coef*((A2-A0)/(A1+A2-2*A0+eps)+1), z_coef*(A1-A2)/(A0+A1-2*A2+eps))
        depth_from_phase = np.where(depth_from_phase - p_offset < 0, 0, depth_from_phase - p_offset)




        save_simumat(npzdata_path, file+'-'+dataname, A0, A1, A2, depth_from_phase, depth_2220_nz)

        accumconv2220 = conved_img[height//2,width//2,:].cumsum()/conved_img[height//2,width//2,:].sum()

        accumconv = np.sum(conv, axis=(0, 1)).cumsum() / np.sum(conv)
        arg99conv = np.abs((accumconv - 0.99)).argmin()
        arg999conv = np.abs((accumconv - 0.999)).argmin()
        arg666 = np.abs((timeseq2960 - 66.6)).argmin()




            #####################################################################
        store_data = {
                      'CumulativeDegree-66ns': accumconv[2220 - 1],
                      'CumulativeDegree-99%-time': timeseq2960[arg99conv],
                      'CumulativeDegree-99dot9%-time': timeseq2960[arg999conv],

                      'min-phase': np.min(depth_from_phase),
                      'avr-phase': np.mean(depth_from_phase),
                      'max-phase': np.max(depth_from_phase),


                      'min-depth-nz': np.min(depth_2220_nz),
                      'avr-depth-nz': np.mean(depth_2220_nz),
                      'max-depth-nz': np.max(depth_2220_nz),

                      'nonzero-num-phase': np.count_nonzero(depth_from_phase),
                      'nonzero-num-depth-nz': np.count_nonzero(depth_2220_nz),


        }
        with open(os.path.join(npzdata_path,file+'-'+dataname+'-info.json'), mode='w') as f:
            json.dump(store_data, f, indent=4)


        plt.figure()
        plt.plot(timeseq2960, accumconv, label="conv")
        plt.xlabel('Time [ns]')
        plt.ylabel('cumulative frequency')
        plt.vlines(timeseq2960[arg99conv], 0, 1, colors='red', linestyle='dashed', linewidth=3)
        plt.vlines(timeseq2960[arg666], 0, 1, colors='blue', linestyle='dashed', linewidth=3)
        # plt.show()
        plt.savefig(os.path.join(npz_folder_path,file+'-'+dataname+'-cumulative.png'))
        plt.close()

        plt.figure(figsize=(6,8))
        plt.subplot(2, 1, 1)
        plt.imshow(A0, cmap='gray')
        plt.colorbar()
        # plt.show()
        plt.subplot(2, 1, 2)
        plt.hist(A0.flatten(), bins=1000)
        plt.xlabel('power', fontsize=14)
        plt.ylabel('number of pixels', fontsize=14)
        plt.savefig(os.path.join(npz_folder_path,file+'-'+dataname+'-A0.png'))
        plt.close()

        plt.figure(figsize=(6,8))
        plt.subplot(2, 1, 1)
        plt.imshow(A1, cmap='gray')
        plt.colorbar()
        # plt.show()
        plt.subplot(2, 1, 2)
        plt.hist(A1.flatten(), bins=1000)
        plt.xlabel('power', fontsize=14)
        plt.ylabel('number of pixels', fontsize=14)
        plt.savefig(os.path.join(npz_folder_path,file+'-'+dataname+'-A1.png'))
        plt.close()

        plt.figure(figsize=(6,8))
        plt.subplot(2, 1, 1)
        plt.imshow(A2, cmap='gray')
        plt.colorbar()
        # plt.show()
        plt.subplot(2, 1, 2)
        plt.hist(A2.flatten(), bins=1000)
        plt.xlabel('power', fontsize=14)
        plt.ylabel('number of pixels', fontsize=14)
        plt.savefig(os.path.join(npz_folder_path,file+'-'+dataname+'-A2.png'))
        plt.close()

        plt.figure(figsize=(6,8))
        plt.subplot(2, 1, 1)
        plt.imshow(depth_from_phase, cmap='gray')
        plt.colorbar()
        # plt.show()
        plt.subplot(2, 1, 2)
        plt.hist(depth_from_phase.flatten(), bins=1000)
        plt.xlabel('Distance [m]', fontsize=14)
        plt.ylabel('number of pixels', fontsize=14)
        plt.savefig(os.path.join(npz_folder_path,file+'-'+dataname+'-phase.png'))
        plt.close()


        plt.figure(figsize=(6,8))
        plt.subplot(2, 1, 1)
        plt.imshow(depth_2220_nz, cmap='gray')
        plt.colorbar()
        # plt.show()
        plt.subplot(2, 1, 2)
        plt.hist(depth_2220_nz.flatten(), bins=1000)
        plt.xlabel('Distance [m]', fontsize=14)
        plt.ylabel('number of pixels', fontsize=14)
        plt.savefig(os.path.join(npz_folder_path,file+'-'+dataname+'-depth2220.png'))
        plt.close()

        plt.figure(figsize=(6,8))
        plt.subplot(2, 1, 1)
        plt.imshow(first_response, label="test")
        plt.colorbar()
        # plt.show()
        plt.subplot(2, 1, 2)
        plt.hist(first_response.flatten(), bins=1000)
        plt.xlabel('Light power', fontsize=14)
        plt.ylabel('number of pixels', fontsize=14)
        plt.savefig(os.path.join(npz_folder_path,file+'-'+dataname+'-1st-bounce.png'))
        plt.close()

        plt.figure(figsize=(6,8))
        plt.subplot(2, 1, 1)
        plt.imshow(second_response, label="test")
        plt.colorbar()
        # plt.show()
        plt.subplot(2, 1, 2)
        plt.hist(second_response.flatten(), bins=1000)
        plt.xlabel('Light power', fontsize=14)
        plt.ylabel('number of pixels', fontsize=14)
        plt.savefig(os.path.join(npz_folder_path,file+'-'+dataname+'-2nd-bounce.png'))
        plt.close()

        plt.figure(figsize=(6,8))
        plt.subplot(2, 1, 1)
        plt.imshow(np.sum(response_img,axis=2)-first_response-second_response, label="test")
        plt.colorbar()
        # plt.show()
        plt.subplot(2, 1, 2)
        plt.hist((np.sum(response_img,axis=2)-first_response-second_response).flatten(), bins=1000)
        plt.xlabel('Light power', fontsize=14)
        plt.ylabel('number of pixels', fontsize=14)
        plt.savefig(os.path.join(npz_folder_path,file+'-'+dataname+'-3rd-bounce.png'))
        plt.close()

