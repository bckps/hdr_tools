import cv2
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import os, json
import matplotlib.animation as animation
from scipy.signal import find_peaks

phase_offset = 0
title = 'comapare'

def show_gen_image(results_folder_path, depth_image, gen_image,z_corr, iters):

    # retrieve distance from normarized depth map
    # 30x10^(-12): [time per frame] x 1024: [frame] x 3x10^8: [speed of light] x depth_map / 2
    # 3 x 1.024 x 3 x depth_map / 2

    depth_img = 6.6 * (depth_image[0, 0, :, :].detach().cpu().numpy() + 1) / 2
    gen_img = 6.6 * (gen_image[0, 0, :, :].detach().cpu().numpy() + 1) / 2
    z_corr = z_corr[0, :, :]
    z_corr = np.where(z_corr - phase_offset < 0, 0, z_corr - phase_offset)
    vmin = 1
    vmax = 2
    diff_vmin = -1
    diff_vmax = 1
    fig, axs = plt.subplots(2, 1,figsize=(4,9))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    im0 = axs[0].imshow(depth_img, vmin=vmin, vmax=vmax)
    # im0 = axs[0].imshow(z_corr, vmin=0, vmax=5)
    # cbar = fig.colorbar(im0, ax=axs[0])
    # cbar.ax.tick_params(labelsize=16)
    # cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    # axs[0].set_title('L1 + adv',fontsize=18)
    axs[0].set_title(title, fontsize=28)
    # axs[0,0].set_xlabel('Column number',fontsize=14)
    # axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0].hlines([32], 0, 127, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0].set_xlim([0, 127])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].plot(gen_img[128 // 4, :], label=title)
    # axs[1].plot(z_corr[128 // 4, :], label=title)
    # axs[1].plot(gen_img[128 // 4, :], label=title)
    # axs[1].plot(gen_img[128 // 4, :], label="deep_learning_depth")
    axs[1].legend(fontsize=14)
    # axs[1].set_title('32th row depth (red line)',fontsize=18)
    axs[1].set_xlabel('Column number', fontsize=18)
    axs[1].tick_params(axis='x', labelsize=16)
    axs[1].tick_params(axis='y', labelsize=16)
    # ax2 = axs[1].twinx()
    # ax2.set_ylabel('32th row depth (red line)',fontsize=18)
    fig.savefig(os.path.join(results_folder_path, 'row32th_gt_'+title+'_'+'.png'))
    # plt.show()
    plt.close()

if __name__ == '__main__':
    # with np.load('/home/saijo/labwork/PythonSandbox/hdr_process/npz/v3-corner-plane-no-del-3rdb/00000/00000-corner-plane-no-del-phase-depth.npz') as data: phase_img_no_del = data.get('arr_0')
    # with np.load('/home/saijo/labwork/PythonSandbox/hdr_process/npz/corner-plane-del1/00000/00000-corner-plane-del1-phase-depth.npz') as data: phase_img_del1 = data.get('arr_0')
    # with np.load('/home/saijo/labwork/PythonSandbox/hdr_process/npz/corner-plane-del2/00000/00000-corner-plane-del2-phase-depth.npz') as data: phase_img_del2 = data.get('arr_0')
    # # with np.load('/home/saijo/labwork/PythonSandbox/hdr_process/npz/corner-plane-no-del/00000/00000-corner-plane-no-del-ground-truth-hdr.npz') as data: depth_img = data.get('arr_0')
    # with np.load('/home/saijo/labwork/PythonSandbox/hdr_process/npz/corner-plane-no-del/00000/00000-corner-plane-no-del-ground-truth.npz') as data: depth1st_img = data.get('arr_0')

    with np.load('/home/saijo/labwork/PythonSandbox/hdr_process/npz/v3-corner-plane-no-del-3rdb/00002/00002-corner-plane-no-del-phase-depth.npz') as data: phase_img_no_del = data.get('arr_0')
    with np.load('/home/saijo/labwork/PythonSandbox/hdr_process/npz/corner-plane-del1/00002/00002-corner-plane-del1-phase-depth.npz') as data: phase_img_del1 = data.get('arr_0')
    with np.load('/home/saijo/labwork/PythonSandbox/hdr_process/npz/corner-plane-del2/00002/00002-corner-plane-del2-phase-depth.npz') as data: phase_img_del2 = data.get('arr_0')
    # with np.load('/home/saijo/labwork/PythonSandbox/hdr_process/npz/corner-plane-no-del/00000/00000-corner-plane-no-del-ground-truth-hdr.npz') as data: depth_img = data.get('arr_0')
    with np.load('/home/saijo/labwork/PythonSandbox/hdr_process/npz/corner-plane-no-del/00002/00002-corner-plane-no-del-ground-truth.npz') as data: depth1st_img = data.get('arr_0')

    phase_img_no_del = phase_img_no_del[256//4:256*3//4,256//4:256*3//4]
    phase_img_del1 = phase_img_del1[256//4:256*3//4,256//4:256*3//4]
    phase_img_del2 = phase_img_del2[256 // 4:256 * 3 // 4, 256 // 4:256 * 3 // 4]
    depth1st_img = depth1st_img[256 // 4:256 * 3 // 4, 256 // 4:256 * 3 // 4]


    vmin = 0
    vmax = 5.8
    plt.figure()
    plt.imshow(phase_img_no_del, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.figure()
    plt.imshow(phase_img_del1, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.figure()
    plt.imshow(phase_img_del2, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.figure()
    plt.imshow(depth1st_img, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.close()

    # plt.figure()

    print(np.min(phase_img_no_del),np.max(phase_img_no_del))
    print(np.min(phase_img_del1), np.max(phase_img_del1))
    print(np.min(phase_img_del2), np.max(phase_img_del2))
    print(np.min(depth1st_img), np.max(depth1st_img))



    fig, axs = plt.subplots(2, 1,figsize=(4,9))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    # print(depth_img.shape)
    im0 = axs[0].imshow(depth1st_img)
    # im0 = axs[0].imshow(z_corr, vmin=0, vmax=5)
    # cbar = fig.colorbar(im0, ax=axs[0])
    # cbar.ax.tick_params(labelsize=16)
    # cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    # axs[0].set_title('L1 + adv',fontsize=18)
    axs[0].set_title(title, fontsize=28)
    # axs[0,0].set_xlabel('Column number',fontsize=14)
    # axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0].hlines([32], 0, 126, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    # axs[0].set_xlim([0, 127])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].plot(depth1st_img[128 // 4, :], label='GT')
    axs[1].plot(phase_img_del2[128 // 4, :], label='1face')
    axs[1].plot(phase_img_del1[128 // 4, :], label='2faces')
    axs[1].plot(phase_img_no_del[128 // 4, :], label='3faces')

    # axs[1].plot(depth_img[128 // 4, :], label='GT')#simu-

    # axs[1].plot(z_corr[128 // 4, :], label=title)
    # axs[1].plot(gen_img[128 // 4, :], label=title)
    # axs[1].plot(gen_img[128 // 4, :], label="deep_learning_depth")
    axs[1].legend(fontsize=14)
    # axs[1].set_title('32th row depth (red line)',fontsize=18)
    axs[1].set_xlabel('Column number', fontsize=18)
    axs[1].tick_params(axis='x', labelsize=16)
    axs[1].tick_params(axis='y', labelsize=16)
    # ax2 = axs[1].twinx()
    # ax2.set_ylabel('32th row depth (red line)',fontsize=18)
    fig.savefig(os.path.join('/home/saijo/labwork/PythonSandbox/hdr_process/npz/corner-plane-no-del/', 'row32th_1st-gt_'+title+'_'+'.png'))
    plt.show()
    plt.close()