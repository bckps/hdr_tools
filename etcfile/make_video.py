import os
import cv2
import numpy as np
def collect_panaToF_inpulsefile(folder_path):
    files = os.listdir(folder_path)
    files.sort()
    b1_list = []
    b2_list = []
    b3_list = []
    b4_list = []
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
        elif file_bounce[1] == 'b04':
            b4_list.append(file)
    return b1_list, b2_list, b3_list, b4_list

if __name__ == '__main__':
    dataname = 'livingroom-test2'
    folder_path = os.path.join(r'/home/saijo/labwork/simulator_origun/scene-results', dataname)
    hdrfile_path = os.path.join(folder_path, '00009', 'hdrs')
    b1_list, b2_list, b3_list, b4_list = collect_panaToF_inpulsefile(hdrfile_path)

    height, width = 256, 256
    response_img = np.zeros((height, width, 2220))
    conv_window = np.ones((740))
    conved_img = np.zeros((height, width, 2220))
    conv = np.zeros((height, width, 2959))

    timeseq = np.linspace((30e-3), (30e-3) * 2221, num=2220)
    timeseq2960 = np.linspace((30e-3), (30e-3) * 2960, num=2959)

    for i, b1 in enumerate(b1_list):
        img = np.array(cv2.imread(os.path.join(hdrfile_path, b1), flags=cv2.IMREAD_ANYDEPTH))
        # print(img.shape)
        response_img[:, i, :] += img[:width, :, 2]

    for i, b2 in enumerate(b2_list):
        img = np.array(cv2.imread(os.path.join(hdrfile_path, b2), flags=cv2.IMREAD_ANYDEPTH))
        # print(img.shape)
        response_img[:, i, :] += img[:width, :, 2]

    for i, b3 in enumerate(b3_list):
        img = np.array(cv2.imread(os.path.join(hdrfile_path, b3), flags=cv2.IMREAD_ANYDEPTH))
        # print(img.shape)
        response_img[:, i, :] += img[:width, :, 2]

    for i, b4 in enumerate(b4_list):
        img = np.array(cv2.imread(os.path.join(hdrfile_path, b4), flags=cv2.IMREAD_ANYDEPTH))
        # print(img.shape)
        response_img[:, i, :] += img[:width, :, 2]

    response_img = response_img.transpose(1, 0, 2)
    for i in range(response_img.shape[0]):
        for j in range(response_img.shape[1]):
            conv[i, j, :] = np.convolve(response_img[i, j, :], conv_window[:], 'full')
            conved_img[i, j, :] = conv[i, j, :][:2220]

    maxvalue = np.max(conv)
    conv = ((conv / maxvalue)*255).astype('uint8')
    print(b1_list)
    print(conv.shape[2])

    # 動画作成
    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    video  = cv2.VideoWriter('ImgVideo-b2-bounce4-fps480.mp4', fourcc, 360.0, (height, width))
    for idx in range(conv.shape[2]):
        img = np.array([conv[:, :, idx], conv[:, :, idx], conv[:, :, idx]], dtype='uint8').transpose(1, 2, 0)
        print(img.shape)
        video.write(img)
    # color_flag = 0
    #
    # for idx in range(conv.shape[2]):
    #     if idx < 740:
    #         color_flag = 0
    #     elif idx < 740*2:
    #         color_flag = 1
    #     elif idx < 740*3:
    #         color_flag = 2
    #     else:
    #         color_flag = 3
    #
    #     if color_flag == 0:
    #         img = np.array([conv[:,:,idx],0*conv[:,:,idx],0*conv[:,:,idx]], dtype='uint8').transpose(1, 2, 0)
    #         print(img.shape)
    #         video.write(img)
    #     elif color_flag == 1:
    #         img = np.array([0*conv[:,:,idx],conv[:,:,idx],0*conv[:,:,idx]], dtype='uint8').transpose(1, 2, 0)
    #         print(img.shape)
    #         video.write(img)
    #     elif color_flag == 2:
    #         img = np.array([0*conv[:,:,idx],0*conv[:,:,idx],conv[:,:,idx]], dtype='uint8').transpose(1, 2, 0)
    #         print(img.shape)
    #         video.write(img)
    #     elif color_flag == 3:
    #         img = np.array([conv[:,:,idx],conv[:,:,idx],conv[:,:,idx]], dtype='uint8').transpose(1, 2, 0)
    #         print(img.shape)
    #         video.write(img)

    video.release()