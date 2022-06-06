# import numpy as np
import h5py
import cupy as np
import matplotlib.pyplot as plt
import numpy
import os,cv2

import torch

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


def four_phases_depth(file_path):
    # data = np.load(r'C:\Users\919im\Documents\local-香川研\pytorchDeepToF\numpy_sensor_data\bathroom_1.npz', 'r')
    data = np.load(file_path, 'r')

    sense_40MHz_img = data.get('arr_0')
    sense_70MHz_img = data.get('arr_1')
    depth_1024 = data.get('arr_2')


    z_max = 3e8 / (2*33e6)
    z40_max = 3e8/(2*40e6)
    z70_max = 3e8/(2*70e6)

    depth_1024 = z_max * depth_1024

    phase_40MHz = np.arctan2(sense_40MHz_img[:,:,2]-sense_40MHz_img[:,:,3],sense_40MHz_img[:,:,0]-sense_40MHz_img[:,:,1])
    # phase_40MHz = np.where(phase_40MHz<0,phase_40MHz+2*np.pi,phase_40MHz)
    phase_40MHz = phase_40MHz + np.pi
    # depth_40MHz = z40_max*phase_40MHz/(2*np.pi)

    phase_70MHz = np.arctan2(sense_70MHz_img[:,:,2]-sense_70MHz_img[:,:,3],sense_70MHz_img[:,:,0]-sense_70MHz_img[:,:,1])
    # phase_70MHz = np.where(phase_70MHz<0,phase_70MHz+2*np.pi,phase_70MHz)
    phase_70MHz = phase_70MHz + np.pi
    # depth_70MHz = z70_max*phase_70MHz/(2*np.pi)

    lut_40MHz = np.linspace(0, np.pi * 1024 / 417, num=1024) % (2 * np.pi)
    lut_70MHz = np.linspace(0, np.pi * 1024 / 238, num=1024) % (2 * np.pi)

    error_mat = np.zeros((1024,256,256))
    for i in range(error_mat.shape[0]):
        error_mat[i,:,:] = np.square(phase_40MHz[:,:]-lut_40MHz[i])+np.square(phase_70MHz[:,:]-lut_70MHz[i])
    z_indexes = np.argmin(error_mat, axis=0)
    z_corr = z_max * z_indexes/1024
    # print(z_indexes)

    return z_corr, depth_1024


def four_phases_cropped_depth(cap_img, datapath=None, random=False, save_path='/home/saijo/labwork/pytorchDToF-edit/eval/phase_depth'):

    filename = os.path.splitext(os.path.basename(datapath))[0]

    if random == False:
        for root, dirs, files in os.walk(save_path):
            if filename+'_phase_depth.npz' in files:
                with np.load(os.path.join(save_path, filename+'_phase_depth.npz'), 'r') as data:
                    z_corr = data.get('arr_0')
                print('load success!')
                return z_corr

    if cap_img is not torch.Tensor:
        cap_img = cap_img.to('cpu').detach().numpy().copy()

    _, img_height, img_width = cap_img.shape
    z_max = 3e8 / (2*33e6)
    z40_max = 3e8/(2*40e6)
    z70_max = 3e8/(2*70e6)


    phase_40MHz = np.arctan2(cap_img[0],cap_img[1])
    # phase_40MHz = np.where(phase_40MHz<0,phase_40MHz+2*np.pi,phase_40MHz)
    phase_40MHz = phase_40MHz + np.pi
    # depth_40MHz = z40_max*phase_40MHz/(2*np.pi)

    phase_70MHz = np.arctan2(cap_img[2],cap_img[3])
    # phase_70MHz = np.where(phase_70MHz<0,phase_70MHz+2*np.pi,phase_70MHz)
    phase_70MHz = phase_70MHz + np.pi
    # depth_70MHz = z70_max*phase_70MHz/(2*np.pi)

    lut_40MHz = np.linspace(0, np.pi * 1024 / 417, num=1024) % (2 * np.pi)
    lut_70MHz = np.linspace(0, np.pi * 1024 / 238, num=1024) % (2 * np.pi)

    error_mat = np.zeros((1024,img_height,img_width))
    for i in range(error_mat.shape[0]):
        error_mat[i,:,:] = np.square(phase_40MHz[:,:]-lut_40MHz[i])+np.square(phase_70MHz[:,:]-lut_70MHz[i])
    z_indexes = np.argmin(error_mat, axis=0)
    z_corr = z_max * z_indexes/1024
    # print(z_indexes)

    if filename and (not random):
        with open(os.path.join(save_path, filename+'_phase_depth.npz'), 'wb') as f:
            np.savez(f, z_corr)
    return z_corr

if __name__ == '__main__':
    # データが3rdまで、GT1stのみのpanaToF
    dataname = 'corner-plane-no-del'
    # dataname = 'corner-plane-del1'
    # dataname = 'corner-plane-del2'
    # dataname = 'bathroom-extended-train2'
    # dataname = 'bathroom-train2'
    # dataname = 'bedroom-test2'
    # dataname = 'contemporaly-bathroom-extended-train2'
    # dataname = 'contemporaly-bathroom-test2'
    # dataname = 'contemporaly-bathroom-train2'
    # dataname = 'livingroom-extended-train2'
    # dataname = 'livingroom-test2'
    # dataname = 'livingroom-train2'
    # dataname = 'light_offset_zero'
    # dataname = 'lambertian-test'
    # dataname = 'single-plane'
    # dataname = 'double-plane-connection'
    # dataname = 'corner-plane-front-back'
    # dataname = 'v3-bathroom-train'
    # dataname = 'v3-contemporary-bathroom-test'
    # dataname = 'v3-contemporary-bathroom-train'
    # dataname = 'v3-livingroom-test'
    # dataname = 'v3-livingroom-train'
    # dataname = 'single-plane-10ps'
    # folder_path = os.path.join(r'/home/saijo/labwork/PythonSandbox/hdr_process/hdrs',dataname)
    folder_path = os.path.join(r'/home/saijo/labwork/simulator_origun/scene-results', dataname)
    npz_folder_path = os.path.join(r'/home/saijo/labwork/PythonSandbox/hdr_process/npz', dataname)
    # os.makedirs(npz_folder_path, exist_ok=True)

    data_files = [f for f in os.listdir(folder_path) if f.isdecimal()]
    data_files.sort()
    for file in data_files:
        print(file)
        # print(os.path.join(folder_path, dataname, file, 'hdrs'))
        hdrfile_path = os.path.join(folder_path, file, 'hdrs')
        b1_list, b2_list, b3_list, depth_hdr_path = collect_panaToF_inpulsefile(hdrfile_path)
        # depth_hdr_path = os.path.join(folder_path, file, 'hdrs')#r"/home/saijo/labwork/simulator_origun/scene-results/single-plane/00000/hdrs/single-plane-00000_depth.hdr"
        npzdata_path = os.path.join(r'/home/saijo/labwork/PythonSandbox/hdr_process/npz', dataname, file)
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


        for i in range(one_responces.shape[0]):
            for j in range(one_responces.shape[1]):
                conv[i, j, :] = np.convolve(one_responces[i, j, :], conv_window[:], 'full')
                conved_img[i, j, :] = conv[i, j, :][:2220]

        data = np.array(data)  # For converting to a NumPy array
        data2 = np.array(data2)
        data3 = np.array(data3)

        array_40MHz_sin = (np.sin(np.linspace(0, np.pi * 417 * 2 / 417, num=417 * 2)) + 1) / 2
        array_70MHz_sin = (np.sin(np.linspace(0, np.pi * 238 * 2 / 238, num=238 * 2)) + 1) / 2

        array_40MHz_sin = np.concatenate((array_40MHz_sin, array_40MHz_sin, array_40MHz_sin, array_40MHz_sin))
        array_70MHz_sin = np.concatenate(
            (array_70MHz_sin, array_70MHz_sin, array_70MHz_sin, array_70MHz_sin, array_70MHz_sin, array_70MHz_sin))

        sin40_length = 417 * 8
        sin40_half_len = 417 * 4
        sin70_length = 238 * 12
        sin70_half_len = 238 * 6
        array_40MHz_sin = np.broadcast_to(array_40MHz_sin, (256, 256, sin40_length)).transpose((2, 0, 1))
        array_70MHz_sin = np.broadcast_to(array_70MHz_sin, (256, 256, sin70_length)).transpose((2, 0, 1))

        sense_40MHz_img = np.zeros((4, 256, 256))
        sense_70MHz_img = np.zeros((4, 256, 256))

        # 40MHz:Q1
        for i in range(417):
            sense_40MHz_img[0, :, :] += np.sum(
                data[:, :, :] * array_40MHz_sin[sin40_half_len - i:sin40_half_len - i + 1024, :, :]
                + data2[:, 0, :, :] * array_40MHz_sin[sin40_half_len - i:sin40_half_len - i + 1024, :, :]
                + data3[:, 0, :, :] * array_40MHz_sin[sin40_half_len - i:sin40_half_len - i + 1024, :, :], axis=0)
        # 40MHz:Q2
        for i in range(417, 417 * 2):
            sense_40MHz_img[1, :, :] += np.sum(
                data[:, 0, :, :] * array_40MHz_sin[sin40_half_len - i:sin40_half_len - i + 1024, :, :]
                + data2[:, 0, :, :] * array_40MHz_sin[sin40_half_len - i:sin40_half_len - i + 1024, :, :]
                + data3[:, 0, :, :] * array_40MHz_sin[sin40_half_len - i:sin40_half_len - i + 1024, :, :], axis=0)
        # 40MHz:Q3
        for i in range(417 // 2, 417 * 3 // 2):
            sense_40MHz_img[2, :, :] += np.sum(
                data[:, 0, :, :] * array_40MHz_sin[sin40_half_len - i:sin40_half_len - i + 1024, :, :]
                + data2[:, 0, :, :] * array_40MHz_sin[sin40_half_len - i:sin40_half_len - i + 1024, :, :]
                + data3[:, 0, :, :] * array_40MHz_sin[sin40_half_len - i:sin40_half_len - i + 1024, :, :], axis=0)
        # 40MHz:Q4
        for i in np.concatenate((np.arange(417 // 2), np.arange(417 * 3 // 2, 417 * 2))):
            sense_40MHz_img[3, :, :] += np.sum(
                data[:, 0, :, :] * array_40MHz_sin[sin40_half_len - i:sin40_half_len - i + 1024, :, :]
                + data2[:, 0, :, :] * array_40MHz_sin[sin40_half_len - i:sin40_half_len - i + 1024, :, :]
                + data3[:, 0, :, :] * array_40MHz_sin[sin40_half_len - i:sin40_half_len - i + 1024, :, :], axis=0)

        # 70MHz:Q1
        for i in range(238):
            sense_70MHz_img[0, :, :] += np.sum(
                data[:, 0, :, :] * array_70MHz_sin[sin70_half_len - i:sin70_half_len - i + 1024, :, :]
                + data2[:, 0, :, :] * array_70MHz_sin[sin70_half_len - i:sin70_half_len - i + 1024, :, :]
                + data3[:, 0, :, :] * array_70MHz_sin[sin70_half_len - i:sin70_half_len - i + 1024, :, :], axis=0)
        # 70MHz:Q2
        for i in range(238, 238 * 2):
            sense_70MHz_img[1, :, :] += np.sum(
                data[:, 0, :, :] * array_70MHz_sin[sin70_half_len - i:sin70_half_len - i + 1024, :, :]
                + data2[:, 0, :, :] * array_70MHz_sin[sin70_half_len - i:sin70_half_len - i + 1024, :, :]
                + data3[:, 0, :, :] * array_70MHz_sin[sin70_half_len - i:sin70_half_len - i + 1024, :, :], axis=0)
        # 70MHz:Q3
        for i in range(238 // 2, 238 * 3 // 2):
            sense_70MHz_img[2, :, :] += np.sum(
                data[:, 0, :, :] * array_70MHz_sin[sin70_half_len - i:sin70_half_len - i + 1024, :, :]
                + data2[:, 0, :, :] * array_70MHz_sin[sin70_half_len - i:sin70_half_len - i + 1024, :, :]
                + data3[:, 0, :, :] * array_70MHz_sin[sin70_half_len - i:sin70_half_len - i + 1024, :, :], axis=0)
        # 70MHz:Q4
        for i in np.concatenate((np.arange(238 // 2), np.arange(238 * 3 // 2, 238 * 2))):
            sense_70MHz_img[3, :, :] += np.sum(
                data[:, 0, :, :] * array_70MHz_sin[sin70_half_len - i:sin70_half_len - i + 1024, :, :]
                + data2[:, 0, :, :] * array_70MHz_sin[sin70_half_len - i:sin70_half_len - i + 1024, :, :]
                + data3[:, 0, :, :] * array_70MHz_sin[sin70_half_len - i:sin70_half_len - i + 1024, :, :], axis=0)

        plt.figure()
        plt.plot(scene40[30,30,:].get())
        plt.figure()
        plt.plot(scene70[30,30,:].get())
        # print(scene70_s.shape)
        # plt.figure()
        # plt.plot(scene40_s[30,30,:])
        # plt.figure()
        # plt.plot(scene70_s[30,30,:])
        # plt.figure()
        # plt.plot(sin40)
        # plt.figure()
        # plt.plot(sin70)
        plt.show()


        # print(b1_list)
        print(np.max(response_img))

        A0 = np.sum(conved_img[:, :, :740], axis=2)
        A1 = np.sum(conved_img[:, :, 740:1480], axis=2)
        A2 = np.sum(conved_img[:, :, 1480:], axis=2)

        max_pixel = np.max(np.array([A0, A1, A2]))
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


        accumconv2220 = conved_img[height//2,width//2,:].cumsum()/conved_img[height//2,width//2,:].sum()

        accumconv = np.sum(conv, axis=(0, 1)).cumsum() / np.sum(conv)
        arg99conv = np.abs((accumconv - 0.99)).argmin()
        arg999conv = np.abs((accumconv - 0.999)).argmin()
        arg666 = np.abs((timeseq2960 - 66.6)).argmin()