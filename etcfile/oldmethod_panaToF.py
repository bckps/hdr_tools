import cv2
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import os, json
import matplotlib.animation as animation
from scipy.signal import find_peaks
from copy import copy
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

def save_npzes(npz_folder_path, dataname, A0, A1, A2, depth_from_phase,depth_1st_phase, depth_2220):
    # panasonic ToF sensor output
    filenameA0 = os.path.join(npz_folder_path, dataname + '-A0.npz')
    filenameA1 = os.path.join(npz_folder_path, dataname + '-A1.npz')
    filenameA2 = os.path.join(npz_folder_path, dataname + '-A2.npz')
    filenamePhaseDepth = os.path.join(npz_folder_path, dataname + '-phase-depth.npz')
    filenamePhase1stDepth = os.path.join(npz_folder_path, dataname + '-phase1st-depth.npz')
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
    with open(filenamePhase1stDepth, 'wb') as f:
        np.savez(f, depth_1st_phase)
    with open(filenameGroundTruth, 'wb') as f:
        np.savez(f, depth_2220)

if __name__ == '__main__':
    # dataname = 'bathroom-extended-train2'
    # dataname = 'bathroom-train2'
    # dataname = 'bedroom-test2'
    # dataname = 'contemporaly-bathroom-extended-train2'
    # dataname = 'contemporaly-bathroom-test2'
    # dataname = 'contemporaly-bathroom-train2'
    # dataname = 'livingroom-extended-train2'
    # dataname = 'livingroom-test2'
    dataname = 'livingroom-train2'
    dataname = 'light_offset_zero'
    # folder_path = os.path.join(r'/home/saijo/labwork/PythonSandbox/hdr_process/hdrs',dataname)
    folder_path = os.path.join(r'/home/saijo/labwork/simulator_origun/scene-results', dataname)
    npz_folder_path = os.path.join(r'/home/saijo/labwork/PythonSandbox/hdr_process/npz', dataname)
    os.makedirs(npz_folder_path, exist_ok=True)

    data_files = [f for f in os.listdir(folder_path) if f.isdecimal()]
    data_files.sort()
    for file in data_files:
        print(file)
        # print(os.path.join(folder_path, dataname, file, 'hdrs'))
        hdrfile_path = os.path.join(folder_path, file, 'hdrs')
        b1_list, b2_list, b3_list = collect_panaToF_inpulsefile(hdrfile_path)

        npzdata_path = os.path.join(r'/home/saijo/labwork/PythonSandbox/hdr_process/npz', dataname, file)
        os.makedirs(npzdata_path, exist_ok=True)


        height, width = 256, 256
        infer_time_list = []

        response_img = np.zeros((height,width,2220))
        conv_window = np.ones((740))
        conved_img = np.zeros((height,width,2220))
        conved_img1st = np.zeros((height, width, 2220))
        conv = np.zeros((height,width,2959))
        conv1st = np.zeros((height,width,2959))
        first_peak_time = np.zeros((height, width))
        first_peak_index = np.zeros((height, width), dtype=np.int64)

        timeseq = np.linspace((30e-3),(30e-3)*2221, num=2220)
        timeseq2960 = np.linspace((30e-3),(30e-3)*2960, num=2959)

        for i,b1 in enumerate(b1_list):
            img = np.array(cv2.imread(os.path.join(hdrfile_path, b1), flags=cv2.IMREAD_ANYDEPTH))
            # print(img.shape)
            response_img[:,i,:] += img[:width,:,2]

        response_img1st = copy(response_img.transpose(1, 0, 2))

        for i,b2 in enumerate(b2_list):
            img = np.array(cv2.imread(os.path.join(hdrfile_path, b2), flags=cv2.IMREAD_ANYDEPTH))
            # print(img.shape)
            response_img[:,i,:] += img[:width,:,2]


        for i,b3 in enumerate(b3_list):
            img = np.array(cv2.imread(os.path.join(hdrfile_path, b3), flags=cv2.IMREAD_ANYDEPTH))
            # print(img.shape)
            response_img[:,i,:] += img[:width,:,2]


        response_img = response_img.transpose(1, 0, 2)
        c = 299792458
        z_max = c*(30e-12)*2220 / 2.0
        # for i in range(height):
        #     for j in range(width):
        #         peaks, _ = find_peaks(response_img[i, j, :])
        #         if len(peaks) == 0:
        #             first_peak_time[i, j] = timeseq[-1]
        #         else:
        #             first_peak_time[i, j] = timeseq[peaks[0]]
        #             first_peak_index[i, j] = int(peaks[0])
        # depth_2220 = z_max * first_peak_index / response_img.shape[2]

        first_nonzero_time = np.zeros((height, width))
        first_nonzero_index = np.zeros((height, width), dtype=np.int64)
        for i in range(height):
            for j in range(width):
                nonzero = np.nonzero(response_img[i, j, :])
                if len(nonzero[0]) == 0:
                    first_nonzero_time[i, j] = timeseq[-1]
                else:
                    first_nonzero_time[i, j] = timeseq[nonzero[0][0]]
                    first_nonzero_index[i, j] = int(nonzero[0][0])
        depth_2220_nz = z_max * first_nonzero_index / response_img.shape[2]


        # 1st ################################################################
        for i in range(response_img1st.shape[0]):
            for j in range(response_img1st.shape[1]):
                conv1st[i, j, :] = np.convolve(response_img1st[i, j, :], conv_window[:], 'full')
                conved_img1st[i, j, :] = conv1st[i, j, :][:2220]

        A01st = np.sum(conved_img1st[:, :, :740], axis=2)
        A11st = np.sum(conved_img1st[:, :, 740:1480], axis=2)
        A21st = np.sum(conved_img1st[:, :, 1480:], axis=2)

        max_pixel = np.max([A01st, A11st, A21st])
        offset = 0.5
        scale = 2

        A01st = (A01st / max_pixel - offset) * scale
        A11st = (A11st / max_pixel - offset) * scale
        A21st = (A21st / max_pixel - offset) * scale

        # 3rd #######################################33
        for i in range(response_img.shape[0]):
            for j in range(response_img.shape[1]):
                conv[i, j, :] = np.convolve(response_img[i, j, :], conv_window[:], 'full')
                conved_img[i, j, :] = conv[i, j, :][:2220]

        A0 = np.sum(conved_img[:, :, :740], axis=2)
        A1 = np.sum(conved_img[:, :, 740:1480], axis=2)
        A2 = np.sum(conved_img[:, :, 1480:], axis=2)

        max_pixel = np.max([A0, A1, A2])
        offset = 0.5
        scale = 2

        A0 = (A0 / max_pixel - offset) * scale
        A1 = (A1 / max_pixel - offset) * scale
        A2 = (A2 / max_pixel - offset) * scale

        start = time.time()

        # phase shift depth
        c = 299792458
        p_offset = 0.555     # p_offset is minimized for L1 error in training data.
        eps = 1e-5
        z_coef = c * 22.2 * (10**-9) / 2.0
        depth_from_phase = np.where(A0 < A2, z_coef*((A2-A0)/(A1+A2-2*A0+eps)+1), z_coef*(A1-A2)/(A0+A1-2*A2+eps))
        depth_from_phase = np.where(depth_from_phase - p_offset < 0, 0, depth_from_phase - p_offset)
        infer_time = time.time() - start
        infer_time_list.append(infer_time)

        depth_1st_phase = np.where(A01st < A21st, z_coef*((A21st-A01st)/(A11st+A21st-2*A01st+eps)+1), z_coef*(A11st-A21st)/(A01st+A11st-2*A21st+eps))
        # depth_1st_phase = np.where(depth_from_phase - p_offset < 0, 0, depth_1st_phase - p_offset)




        # depth max is 6.66 [m]
        # z_max = c*(30e-12)*2220 / 2.0
        # cflight_index = int(z_max // (c * 30 * (10**-12)))
        # depth_2220 = z_max * np.argmax(response_img[:, :, :cflight_index], axis=2) / 2
        # print(np.max(np.argmax(response_img, axis=2) / response_img.shape[2]))
        # depth_2220 = z_max * np.argmax(response_img, axis=2) / response_img.shape[2]

        # peak depth
        # for i in range(height):
        #     for j in range(width):
        #         peaks, _ = find_peaks(response_img[i, j, :])
        #         if len(peaks) == 0:
        #             first_peak_time[i, j] = timeseq[-1]
        #         else:
        #             first_peak_time[i, j] = timeseq[peaks[0]]
        #             first_peak_index[i, j] = int(peaks[0])
        # depth_2220_p = z_max * first_peak_index / response_img.shape[2]

        save_npzes(npzdata_path, file+'-'+dataname, A0, A1, A2, depth_from_phase, depth_1st_phase, depth_2220_nz)

        accumconv2220 = conved_img[height//2,width//2,:].cumsum()/conved_img[height//2,width//2,:].sum()

        accumconv = np.sum(conv, axis=(0, 1)).cumsum() / np.sum(conv)
        arg99conv = np.abs((accumconv - 0.99)).argmin()
        arg999conv = np.abs((accumconv - 0.999)).argmin()
        arg666 = np.abs((timeseq2960 - 66.6)).argmin()

        print(accumconv[2220 - 1])
        print(np.mean(depth_from_phase))
        print('i')
        print(np.mean(depth_from_phase))
        # print(np.mean(depth_2220))
        # print(np.mean(depth_2220_p))
        print(np.mean(depth_2220_nz))
        print(np.count_nonzero(depth_from_phase))
        # print(np.count_nonzero(depth_2220))
        # print(np.count_nonzero(depth_2220_p))
        print(np.count_nonzero(depth_2220_nz))
        print('error3rd', np.mean(depth_from_phase-depth_2220_nz))
        print('error1st', np.mean(depth_1st_phase - depth_2220_nz))



            #####################################################################
        store_data = {
                      'CumulativeDegree-66ns': accumconv[2220 - 1],
                      'CumulativeDegree-99%-time': timeseq2960[arg99conv],
                      'CumulativeDegree-99dot9%-time': timeseq2960[arg999conv],

                      'min-phase': np.min(depth_from_phase),
                      'avr-phase': np.mean(depth_from_phase),
                      'max-phase': np.max(depth_from_phase),

                      'min-1st-phase': np.min(depth_1st_phase),
                      'avr-1st-phase': np.mean(depth_1st_phase),
                      'max-1st-phase': np.max(depth_1st_phase),

                      # 'min-depth': np.min(depth_2220),
                      # 'avr-depth': np.mean(depth_2220),
                      # 'max-depth': np.max(depth_2220),
                      #
                      # 'min-depth-p': np.min(depth_2220_p),
                      # 'avr-depth-p': np.mean(depth_2220_p),
                      # 'max-depth-p': np.max(depth_2220_p),

                      'min-depth-nz': np.min(depth_2220_nz),
                      'avr-depth-nz': np.mean(depth_2220_nz),
                      'max-depth-nz': np.max(depth_2220_nz),

                      'nonzero-num-phase': np.count_nonzero(depth_from_phase),
                      # 'nonzero-num-depth': np.count_nonzero(depth_2220),
                      # 'nonzero-num-depth-p': np.count_nonzero(depth_2220_p),
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


        plt.figure()
        plt.imshow(depth_from_phase, label="test")
        plt.colorbar()
        # plt.show()
        plt.savefig(os.path.join(npz_folder_path,file+'-'+dataname+'-phase.png'))
        plt.close()
        #

        plt.figure()
        plt.imshow(depth_1st_phase, label="test")
        plt.colorbar()
        # plt.show()
        plt.savefig(os.path.join(npz_folder_path,file+'-'+dataname+'-phase1st.png'))
        plt.close()

        plt.figure()
        plt.imshow(depth_2220_nz, label="test")
        plt.colorbar()
        # plt.show()
        plt.savefig(os.path.join(npz_folder_path,file+'-'+dataname+'-depth2220.png'))
        plt.close()

    print(np.mean(infer_time_list))