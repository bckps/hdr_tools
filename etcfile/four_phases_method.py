# import numpy as np
import h5py
import cupy as np
import matplotlib.pyplot as plt
import numpy
import os,cv2

import torch

def save_npzes(npz_folder_path, dataname, A0, A1, A2, depth_from_phase, depth_2220, AMCW_depth):
    # panasonic ToF sensor output
    filenameA0 = os.path.join(npz_folder_path, dataname + '-A0.npz')
    filenameA1 = os.path.join(npz_folder_path, dataname + '-A1.npz')
    filenameA2 = os.path.join(npz_folder_path, dataname + '-A2.npz')
    filenamePhaseDepth = os.path.join(npz_folder_path, dataname + '-phase-depth.npz')
    filenameAMCW = os.path.join(npz_folder_path, dataname + '-AMCW.npz')
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
    with open(filenameAMCW, 'wb') as f:
        np.savez(f, AMCW_depth)


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


def four_phases_depth(sense_40MHz_img,sense_70MHz_img,t_len):
    # data = np.load(r'C:\Users\919im\Documents\local-香川研\pytorchDeepToF\numpy_sensor_data\bathroom_1.npz', 'r')
    # data = np.load(file_path, 'r')

    # sense_40MHz_img = data.get('arr_0')
    # sense_70MHz_img = data.get('arr_1')
    # depth_1024 = data.get('arr_2')

    # t_len = 1024*2 # prev settings
    # t_len = int(1024 * 1.5)

    # z_max = 3e8 / (2*16.5e6)
    # z_max = 3e8 / (33e6) # prev settings
    z_max = 30e-12 * t_len * 3e8 / 2
    z40_max = 3e8/(2*40e6)
    z70_max = 3e8/(2*70e6)

    # depth_1024 = z_max * depth_1024

    phase_40MHz = np.arctan2(sense_40MHz_img[2]-sense_40MHz_img[3],sense_40MHz_img[0]-sense_40MHz_img[1])
    # phase_40MHz = np.where(phase_40MHz<0,phase_40MHz+2*np.pi,phase_40MHz)
    phase_40MHz = phase_40MHz + np.pi
    # depth_40MHz = z40_max*phase_40MHz/(2*np.pi)

    phase_70MHz = np.arctan2(sense_70MHz_img[2]-sense_70MHz_img[3],sense_70MHz_img[0]-sense_70MHz_img[1])
    # phase_70MHz = np.where(phase_70MHz<0,phase_70MHz+2*np.pi,phase_70MHz)
    phase_70MHz = phase_70MHz + np.pi
    # depth_70MHz = z70_max*phase_70MHz/(2*np.pi)

    lut_40MHz = np.linspace(0, np.pi * t_len / 417, num=t_len) % (2 * np.pi)
    lut_70MHz = np.linspace(0, np.pi * t_len / 238, num=t_len) % (2 * np.pi)

    print(phase_40MHz.shape)
    print(lut_40MHz.shape)

    error_mat = np.zeros((t_len,256,256))
    for i in range(error_mat.shape[0]):
        error_mat[i,:,:] = np.square(phase_40MHz[:,:]-lut_40MHz[i])+np.square(phase_70MHz[:,:]-lut_70MHz[i])
    z_indexes = np.argmin(error_mat, axis=0)
    z_corr = z_max * z_indexes/t_len
    # print(z_indexes)

    return z_corr

def four_phases_depth_40MHz_single(sense_40MHz_img):
    # data = np.load(r'C:\Users\919im\Documents\local-香川研\pytorchDeepToF\numpy_sensor_data\bathroom_1.npz', 'r')
    # data = np.load(file_path, 'r')

    z40_max = 3e8/(2*40e6)
    z70_max = 3e8/(2*70e6)

    # depth_1024 = z_max * depth_1024

    phase_40MHz = np.arctan2(sense_40MHz_img[2]-sense_40MHz_img[3],sense_40MHz_img[0]-sense_40MHz_img[1])
    phase_40MHz = phase_40MHz + np.pi

    print(phase_40MHz.shape)

    p40_depth = z40_max * phase_40MHz/(2*np.pi)
    # print(z_indexes)

    return p40_depth

def four_phases_depth_70MHz_single(sense_70MHz_img):
    # data = np.load(r'C:\Users\919im\Documents\local-香川研\pytorchDeepToF\numpy_sensor_data\bathroom_1.npz', 'r')
    # data = np.load(file_path, 'r')

    # z40_max = 3e8/(2*40e6)
    z70_max = 3e8/(2*70e6)

    # depth_1024 = z_max * depth_1024

    phase_70MHz = np.arctan2(sense_70MHz_img[2]-sense_70MHz_img[3],sense_70MHz_img[0]-sense_70MHz_img[1])
    phase_70MHz = phase_70MHz + np.pi

    print(phase_70MHz.shape)

    p70_depth = z70_max * phase_70MHz/(2*np.pi)
    # print(z_indexes)

    return p70_depth

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
    dataname = '5by5-front-plane'
    # dataname = 'corner-plane-no-del'
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


        # for i in range(one_responces.shape[0]):
        #     for j in range(one_responces.shape[1]):
        #         conv[i, j, :] = np.convolve(one_responces[i, j, :], conv_window[:], 'full')
        #         conved_img[i, j, :] = conv[i, j, :][:2220]
        for i in range(response_img.shape[0]):
            for j in range(response_img.shape[1]):
                conv[i, j, :] = np.convolve(response_img[i, j, :], conv_window[:], 'full')
                conved_img[i, j, :] = conv[i, j, :][:2220]

        resp_length = one_responces.shape[2]
        # tlength = 1024
        tlength = int(1024*1.5)
        sin40 = (np.sin(np.arange(834) * np.pi/416.67)+1)/2
        sin70 = (np.sin(np.arange(476) * np.pi/238.095)+1)/2
        # scene40_s = np.zeros((height, width, one_responces.shape[2]+len(sin40)-1))
        # scene70_s = np.zeros((height, width, one_responces.shape[2]+len(sin70)-1))
        scene40 = np.zeros((height, width, tlength))
        scene70 = np.zeros((height, width, tlength))
        # scene40_conved_img = np.zeros((height,width,tlength))
        # scene70_conved_img = np.zeros((height,width,tlength))
        for i in range(tlength):
            for j in range(one_responces.shape[2]):
                scene40[:,:,i] =scene40[:,:,i]+response_img[:,:,j]*(np.sin((j-i)*np.pi/416.67)+1)/2 #833.33frame=25nsecで1周期=40MHz
                # scene40[:,:,i] =scene40[:,:,i]+one_responces[:,:,j]*(np.sin((j-i)*np.pi/416.67)+1)/2 #833.33frame=25nsecで1周期=40MHz
                # scene40_conved_img[i, j, :] = scene40[i, j, :][:2220]
        for i in range(tlength):
            for j in range(one_responces.shape[2]):
                scene70[:,:,i] =scene70[:,:,i]+response_img[:,:,j]*(np.sin((j-i)*np.pi/238.095)+1)/2 #833.33frame=25nsecで1周期=40MHz
                # scene70[:,:,i] =scene70[:,:,i]+one_responces[:,:,j]*(np.sin((j-i)*np.pi/238.095)+1)/2 #833.33frame=25nsecで1周期=40MHz
                # scene40_conved_img[i, j, :] = scene40[i, j, :][:2220]

        print(scene40.shape)
        print(scene70.shape)

        Q1_40MHz = np.sum(scene40[:,:,:834//2],axis=2)
        Q2_40MHz = np.sum(scene40[:,:,834//2:834-1],axis=2)
        Q3_40MHz = np.sum(scene40[:,:,834//4:3*834//4],axis=2)
        Q4_40MHz = np.sum(scene40[:,:,:834//4]+scene40[:,:,3*834//4:834-1],axis=2)

        print(Q1_40MHz.shape)
        print(Q2_40MHz.shape)
        print(Q3_40MHz.shape)
        print(Q4_40MHz.shape)

        Q1_70MHz = np.sum(scene70[:,:,:476//2],axis=2)
        Q2_70MHz = np.sum(scene70[:,:,476//2:476-1],axis=2)
        Q3_70MHz = np.sum(scene70[:,:,476//4:3*476//4],axis=2)
        Q4_70MHz = np.sum(scene70[:,:,:476//4]+scene70[:,:,3*476//4:476],axis=2)

        print(Q1_70MHz.shape)
        print(Q2_70MHz.shape)
        print(Q3_70MHz.shape)
        print(Q4_70MHz.shape)

        print(np.array([Q1_40MHz,Q2_40MHz,Q3_40MHz,Q4_40MHz]).shape)
        print(np.array([Q1_40MHz, Q2_40MHz, Q3_40MHz, Q4_40MHz]).shape)
        # depth_40MHz = four_phases_depth_40MHz_single(np.array([Q1_40MHz,Q2_40MHz,Q3_40MHz,Q4_40MHz]))
        # depth_40MHz = depth_40MHz.get()
        # with open(os.path.join(npzdata_path, file+'_depth_40MHz.npz'), 'wb') as f:
        #     np.savez(f, depth_40MHz)
        # depth_70MHz = four_phases_depth_70MHz_single(np.array([Q1_70MHz, Q2_70MHz, Q3_70MHz, Q4_70MHz]))
        # depth_70MHz = depth_70MHz.get()
        # with open(os.path.join(npzdata_path, file+'_depth_70MHz.npz'), 'wb') as f:
        #     np.savez(f, depth_70MHz)

        z_corr_depth = four_phases_depth(np.array([Q1_40MHz,Q2_40MHz,Q3_40MHz,Q4_40MHz]),np.array([Q1_70MHz,Q2_70MHz,Q3_70MHz,Q4_70MHz]), tlength)

        # print('end self conv')
        # for i in range(one_responces.shape[0]):
        #     for j in range(one_responces.shape[1]):
        #         scene40_s[i,j,:] =np.convolve(sin40,one_responces[i,j,:]) #833.33frame=25nsecで1周期=40MHz
        #         # scene40_conved_img[i, j, :] = scene40[i, j, :][:2220]
        # for i in range(one_responces.shape[0]):
        #     for j in range(one_responces.shape[1]):
        #         scene70_s[i,j,:] =np.convolve(sin70,one_responces[i,j,:])
        #         # scene40_conved_img[i, j, :] = scene40[i, j, :][:2220]

        # for i in range(one_responces.shape[0]):
        #     for j in range(one_responces.shape[1]):
        #         conv[i, j, :] =np.convolve(imgVolume1(:,:,1,j)*(np.sin((j-i)*np.pi/416.67)+1)/2; #833.33frame=25nsecで1周期=40MHz
        #         conved_img[i, j, :] = conv[i, j, :][:2220]
        scene40_img = scene40[30, 30, :].get()
        scene70_img = scene70[30,30,:].get()
        z_corr_get = z_corr_depth.get()
        depth_2220_get = depth_2220_nz.get()

        vmin = 2
        vmax = 4.2
        plt.figure()
        plt.plot(scene40_img)
        plt.figure()
        plt.plot(scene70_img)
        plt.figure()
        plt.imshow(z_corr_get)
        plt.figure()
        plt.imshow(depth_2220_get)
        plt.figure()
        plt.imshow(z_corr_get - depth_2220_get)
        # print(scene70_s.shape)
        # plt.figure()
        # plt.plot(scene40_s[30,30,:])
        # plt.figure()
        # plt.plot(scene70_s[30,30,:])
        # plt.figure()
        # plt.plot(sin40)
        # plt.figure()
        # plt.plot(sin70)
        # plt.show()

        plt.figure()
        plt.plot(scene40_img)
        # plt.show()
        # plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner',dataname+'-'+file+'-'+'-40MHz.png'))
        plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner', dataname + '-' + file + '-' + '-40MHz-3b.png'))
        plt.close()

        plt.figure()
        plt.plot(scene70_img)
        # plt.show()
        # plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner',dataname+'-'+file+'-'+'-70MHz.png'))
        plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner', dataname + '-' + file + '-' + '-70MHz-3b.png'))
        plt.close()

        plt.figure()
        plt.imshow(z_corr_get[64:-64, 64:-64], label="test")
        plt.colorbar()
        # plt.show()
        # plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner',dataname+'-'+file+'-'+'-z_corr.png'))
        plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner',dataname+'-'+file+'-'+'-z_corr-3b.png'))
        plt.close()

        plt.figure()
        plt.imshow(depth_2220_get[64:-64, 64:-64], label="test")
        plt.colorbar()
        # plt.show()
        # plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner',dataname+'-'+file+'-'+'-depth.png'))
        plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner',dataname+'-'+file+'-'+'-depth-3b.png'))
        plt.close()

        plt.figure()
        plt.imshow(z_corr_get[64:-64, 64:-64] - depth_2220_get[64:-64, 64:-64], label="test")
        plt.colorbar()
        # plt.show()
        # plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner',dataname+'-'+file+'-'+'-diff.png'))
        plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner',dataname+'-'+file+'-'+'-diff-3b.png'))
        plt.close()

        plt.figure()
        plt.imshow(z_corr_get[64:-64, 64:-64] - depth_2220_get[64:-64, 64:-64], label="test")
        plt.colorbar()
        # plt.show()
        # plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner',dataname+'-'+file+'-'+'-diff.png'))
        plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner',dataname+'-'+file+'-'+'-diff-3b.png'))
        plt.close()

        # plt.figure()
        # plt.imshow(depth_40MHz, label="test")
        # plt.colorbar()
        # # plt.show()
        # # plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner',dataname+'-'+file+'-'+'-diff.png'))
        # plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner',dataname+'-'+file+'-'+'-40MHz-depth.png'))
        # plt.close()
        #
        # plt.figure()
        # plt.imshow(depth_70MHz, label="test")
        # plt.colorbar()
        # # plt.show()
        # # plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner',dataname+'-'+file+'-'+'-diff.png'))
        # plt.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner',dataname+'-'+file+'-'+'-70MHz-depth.png'))
        # plt.close()

        fig, axs = plt.subplots(2, 1, figsize=(4, 9))
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        im0 = axs[0].imshow(z_corr_get[64:-64, 64:-64])
        axs[0].set_title('ratio method', fontsize=28)
        axs[0].hlines([32], 0, 127, "red", linestyles='dashed', linewidth=4)
        axs[0].set_xlim([0, 127])
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].plot(depth_2220_get[128 // 4, 64:-64], label="ground_truth")
        axs[1].plot(z_corr_get[128 // 4, 64:-64], label='ratio')
        axs[1].legend(fontsize=14)
        axs[1].set_xlabel('Column number', fontsize=18)
        axs[1].tick_params(axis='x', labelsize=16)
        axs[1].tick_params(axis='y', labelsize=16)
        fig.savefig(os.path.join('/home/saijo/labwork/研究結果まとめ/AMCW-corner',dataname+'-'+file+'-'+ '_phase.png'))
        # plt.show()
        plt.close()

        del scene40
        del scene70

        # パルス方式の計算
        print(b1_list)
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
        #
        #
        # accumconv2220 = conved_img[height//2,width//2,:].cumsum()/conved_img[height//2,width//2,:].sum()
        #
        # accumconv = np.sum(conv, axis=(0, 1)).cumsum() / np.sum(conv)
        # arg99conv = np.abs((accumconv - 0.99)).argmin()
        # arg999conv = np.abs((accumconv - 0.999)).argmin()
        # arg666 = np.abs((timeseq2960 - 66.6)).argmin()

        save_npzes(npzdata_path, file + '-' + dataname, A0, A1, A2, depth_from_phase, depth_2220_nz, z_corr_get)