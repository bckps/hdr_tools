from __future__ import print_function, division
import os
import pathlib
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from skimage import io, transform
# import numpy as np
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import functional as TF
from matplotlib import pyplot as plt
from torchsummary import summary
from phase_method.four_phases_method import four_phases_depth
import h5py
from PIL import Image


from model_cat_version import Generator, Discriminator, PatchDiscriminator
import dataset_preprocess
import total_variation

batch_size = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256
RESIZE_IMG_WIDTH = 286
RESIZE_IMG_HEIGHT = 286
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 3
LAMBDA = 100


eval_folder_name = 'sample_cat_1'



class ToTensor(object):
    def __call__(self, inputs):
        cap_image, cap_ideal_image, depth_image, z_corr = inputs['cap'].transpose((0, 2, 1)), inputs['cap_ideal'].transpose((0, 2, 1)),  inputs['depth_1024'].transpose((1, 0)), inputs['phase_data']
        return {'cap': torch.from_numpy(cap_image).to(torch.float32), 'cap_ideal': torch.from_numpy(cap_ideal_image).to(torch.float32),
                'depth_1024': torch.unsqueeze(torch.from_numpy(depth_image).to(torch.float32), 0), 'path': inputs['path'],'phase_data':z_corr}



class RandomCrop(object):
    def __call__(self, inputs):
        cap_image, cap_ideal_image, depth_image, z_corr = inputs['cap'], inputs['cap_ideal'],  inputs['depth_1024'], inputs['phase_data']
        crop_size = transforms.RandomCrop.get_params(cap_image, (128, 128))
        return {'cap': TF.crop(cap_image, *crop_size), 'cap_ideal': TF.crop(cap_ideal_image, *crop_size),
                'depth_1024': TF.crop(depth_image, *crop_size), 'path': inputs['path'],'phase_data':np.array(TF.crop(Image.fromarray(z_corr), *crop_size))}




class ImageNormalize(object):
    def __init__(self):
        self.ImgNorm = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
        self.ImgNorm_depth = transforms.Normalize((0.5), (0.5))
    def __call__(self, inputs):
        cap_image, cap_ideal_image, depth_image,z_corr = inputs['cap'], inputs['cap_ideal'],  inputs['depth_1024'], inputs['phase_data']
        return {'cap': self.ImgNorm(cap_image), 'cap_ideal': self.ImgNorm(cap_ideal_image),
                'depth_1024': self.ImgNorm_depth(depth_image), 'path': inputs['path'],'phase_data':z_corr}


class ToFDataset(Dataset):
    def __init__(self, parent_path, transform=None):
        self.parent_path = parent_path
        self.transform = transform
        self.files = []
        for scene in os.listdir(self.parent_path):
            for file in os.listdir(os.path.join(self.parent_path,scene)):
                path = os.path.join(self.parent_path, scene, file)
                if os.path.isfile(path):
                    self.files.append(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.files[idx]
        with h5py.File(img_path) as imgs:
            cap_img = np.array(imgs['cap'])
            cap_ideal_img = np.array(imgs['cap_ideal'])
            depth_img = np.array(imgs['depth_1024'])

        data_path = os.path.join(r'/home/saijo/labwork/local-?????????/pytorchDeepToF/numpy_sensor_data',
                                 os.path.splitext(os.path.basename(img_path))[0] + '.npz')
        print(data_path)
        z_corr, _ = four_phases_depth(data_path)
        items = {'cap': cap_img, 'cap_ideal': cap_ideal_img, 'depth_1024': depth_img, 'path': img_path, 'phase_data': z_corr}

        if self.transform:
            items = self.transform(items)

        return items

def show_gen_image(cap_image, cap_ideal_image, depth_image, gen_image,z_corr, iters):

    # retrieve distance from normarized depth map
    # 30x10^(-12): [time per frame] x 1024: [frame] x 3x10^8: [speed of light] x depth_map / 2
    # 3 x 1.024 x 3 x depth_map / 2

    image_prefix = 'delay'

    depth_img = (depth_image[0, 0, :, :].detach().cpu().numpy() + 1) / 2
    depth_img = 3 * 3 * 1.024 * depth_img / 2
    gen_img = (gen_image[0, 0, :, :].detach().cpu().numpy() + 1) / 2
    gen_img = 3 * 3 * 1.024 * gen_img / 2

    fig, axs = plt.subplots(2, 2,figsize=(9,9))
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    im0 = axs[0,0].imshow(depth_img, vmin=0, vmax=5)
    cbar = fig.colorbar(im0, ax=axs[0,0])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[0,0].set_title('ground_truth',fontsize=18)
    axs[0,0].set_xlabel('Column number',fontsize=14)
    axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0, 0].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0, 0].set_xlim([0, 128])

    im1 = axs[0,1].imshow(z_corr, vmin=0, vmax=5)
    cbar = fig.colorbar(im1, ax=axs[0,1])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[0,1].set_title('phase_depth',fontsize=18)
    axs[0,1].set_xlabel('Column number',fontsize=14)
    axs[0, 1].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0, 1].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 1].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 1].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0, 1].set_xlim([0, 128])

    imgen = axs[1,0].imshow(gen_img, vmin=0, vmax=5)
    cbar = fig.colorbar(imgen, ax=axs[1,0])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[1,0].set_title('deep_learning_depth',fontsize=18)
    axs[1,0].set_xlabel('Column number',fontsize=14)
    axs[1,0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[1,0].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    # axs[1,0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[1,0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[1,0].set_xlim([0, 128])

    axs[1,1].plot(depth_img[128 // 4, :], label="ground_truth")
    axs[1,1].plot(z_corr[128 // 4, :], label="phase_depth")
    axs[1,1].plot(gen_img[128 // 4, :], label="deep_learning_depth")
    axs[1,1].legend()
    axs[1,1].set_title('32th row depth (red line)',fontsize=18)
    axs[1,1].set_xlabel('Column number',fontsize=14)
    axs[1,1].set_ylabel('Depth [m]',fontsize=14)
    fig.savefig(os.path.join(r'/home/saijo/labwork/local-?????????/pytorchDeepToF/numpy_sensor_data',image_prefix+'delay_row32th_{}.png'.format(iters)))
    plt.show()

    fig, axs = plt.subplots(2, 2,figsize=(9,9))
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    im0 = axs[0,0].imshow(depth_img, vmin=0, vmax=5)
    cbar = fig.colorbar(im0, ax=axs[0,0])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[0,0].set_title('ground_truth',fontsize=18)
    axs[0,0].set_xlabel('Column number',fontsize=14)
    axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0,0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0, 0].set_xlim([0, 128])

    im1 = axs[0,1].imshow(z_corr, vmin=0, vmax=5)
    cbar = fig.colorbar(im1, ax=axs[0,1])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[0,1].set_title('phase_depth',fontsize=18)
    axs[0,1].set_xlabel('Column number',fontsize=14)
    axs[0, 1].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0,1].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 1].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 1].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0, 1].set_xlim([0, 128])

    imgen = axs[1,0].imshow(gen_img, vmin=0, vmax=5)
    cbar = fig.colorbar(imgen, ax=axs[1,0])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[1,0].set_title('deep_learning_depth',fontsize=18)
    axs[1,0].set_xlabel('Column number',fontsize=14)
    axs[1,0].set_ylabel('Row number',labelpad=0,fontsize=14)
    # axs[1,0].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    axs[1,0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[1,0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[1,0].set_xlim([0, 128])

    axs[1,1].plot(depth_img[128 // 2, :], label="ground_truth")
    axs[1,1].plot(z_corr[128 // 2, :], label="phase_depth")
    axs[1,1].plot(gen_img[128 // 2, :], label="deep_learning_depth")
    axs[1,1].legend()
    axs[1,1].set_title('64th row depth (green line)',fontsize=18)
    axs[1,1].set_xlabel('Column number',fontsize=14)
    axs[1,1].set_ylabel('Depth [m]',fontsize=14)
    fig.savefig(os.path.join(r'/home/saijo/labwork/local-?????????/pytorchDeepToF/numpy_sensor_data',image_prefix+'_row64th_{}.png'.format(iters)))
    plt.show()


    fig, axs = plt.subplots(2, 2,figsize=(9,9))
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    im0 = axs[0,0].imshow(depth_img, vmin=0, vmax=5)
    cbar = fig.colorbar(im0, ax=axs[0,0])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[0,0].set_title('ground_truth',fontsize=18)
    axs[0,0].set_xlabel('Column number',fontsize=14)
    axs[0, 0].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0,0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0, 0].set_xlim([0, 128])

    im1 = axs[0,1].imshow(z_corr, vmin=0, vmax=5)
    cbar = fig.colorbar(im1, ax=axs[0,1])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[0,1].set_title('phase_depth',fontsize=18)
    axs[0,1].set_xlabel('Column number',fontsize=14)
    axs[0, 1].set_ylabel('Row number',labelpad=0,fontsize=14)
    axs[0,1].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    # axs[0, 1].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 1].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[0, 1].set_xlim([0, 128])

    imgen = axs[1,0].imshow(gen_img, vmin=0, vmax=5)
    cbar = fig.colorbar(imgen, ax=axs[1,0])
    cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90,fontsize=14)
    axs[1,0].set_title('deep_learning_depth',fontsize=18)
    axs[1,0].set_xlabel('Column number',fontsize=14)
    axs[1,0].set_ylabel('Row number',labelpad=0,fontsize=14)
    # axs[1,0].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    # axs[1,0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    axs[1,0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    axs[1,0].set_xlim([0, 128])

    axs[1,1].plot(depth_img[128*3 // 4, :], label="ground_truth")
    axs[1,1].plot(z_corr[128*3 // 4, :], label="phase_depth")
    axs[1,1].plot(gen_img[128*3 // 4, :], label="deep_learning_depth")
    axs[1,1].legend()
    axs[1,1].set_title('96th row depth (blue line)',fontsize=18)
    axs[1,1].set_xlabel('Column number',fontsize=14)
    axs[1,1].set_ylabel('Depth [m]',fontsize=14)
    fig.savefig(os.path.join(r'/home/saijo/labwork/local-?????????/pytorchDeepToF/numpy_sensor_data',image_prefix+'_row96th_{}.png'.format(iters)))
    plt.show()

    # axs[1,1].plot(depth_img[128 // 2, :], label="ground_truth")
    # axs[1,1].plot(z_corr[128 // 2, :], label="phase_depth")
    # axs[1,1].plot(gen_img[128 // 2, :], label="deep_learning_depth")
    # axs[1,1].legend()
    # axs[1, 1].set_title('64th row depth (green line)')
    #
    # axs[1,2].plot(depth_img[128*3 // 4, :], label="ground_truth")
    # axs[1,2].plot(z_corr[128*3 // 4, :], label="phase_depth")
    # axs[1,2].plot(gen_img[128*3 // 4, :], label="deep_learning_depth")
    # axs[1,2].legend()
    # axs[1, 2].set_title('96th row depth (blue line)')



    # fig, axs = plt.subplots(2, 3,figsize=(16,9))
    # im0 = axs[0,0].imshow(depth_img, vmin=0, vmax=5)
    # cbar = fig.colorbar(im0, ax=axs[0,0])
    # cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90)
    # axs[0,0].set_title('ground_truth')
    # axs[0,0].set_xlabel('Column number')
    # axs[0, 0].set_ylabel('Row number',labelpad=0)
    # axs[0, 0].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 0].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    # axs[0, 0].set_xlim([0, 128])
    #
    # im1 = axs[0,1].imshow(z_corr, vmin=0, vmax=5)
    # cbar = fig.colorbar(im1, ax=axs[0,1])
    # cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90)
    # axs[0,1].set_title('phase_depth')
    # axs[0,1].set_xlabel('Column number')
    # axs[0, 1].set_ylabel('Row number',labelpad=0)
    # axs[0, 1].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 1].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 1].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    # axs[0, 1].set_xlim([0, 128])
    #
    # imgen = axs[0,2].imshow(gen_img, vmin=0, vmax=5)
    # cbar = fig.colorbar(imgen, ax=axs[0,2])
    # cbar.ax.set_ylabel('Depth [m]',labelpad=7, rotation=90)
    # axs[0,2].set_title('deep_learning_depth')
    # axs[0,2].set_xlabel('Column number')
    # axs[0, 2].set_ylabel('Row number',labelpad=0)
    # axs[0, 2].hlines([32], 0, 128, "red", linestyles='dashed',linewidth = 4)
    # axs[0, 2].hlines([64], 0, 128, "green", linestyles='dashed',linewidth = 4)
    # axs[0, 2].hlines([96], 0, 128, "blue", linestyles='dashed',linewidth = 4)
    # axs[0, 2].set_xlim([0, 128])
    #
    # axs[1, 0].plot(depth_img[128 // 4, :], label="ground_truth")
    # axs[1, 0].plot(z_corr[128 // 4, :], label="phase_depth")
    # axs[1, 0].plot(gen_img[128 // 4, :], label="deep_learning_depth")
    # axs[1, 0].legend()
    # axs[1, 0].set_title('32th row depth (red line)')
    #
    # axs[1,1].plot(depth_img[128 // 2, :], label="ground_truth")
    # axs[1,1].plot(z_corr[128 // 2, :], label="phase_depth")
    # axs[1,1].plot(gen_img[128 // 2, :], label="deep_learning_depth")
    # axs[1,1].legend()
    # axs[1, 1].set_title('64th row depth (green line)')
    #
    # axs[1,2].plot(depth_img[128*3 // 4, :], label="ground_truth")
    # axs[1,2].plot(z_corr[128*3 // 4, :], label="phase_depth")
    # axs[1,2].plot(gen_img[128*3 // 4, :], label="deep_learning_depth")
    # axs[1,2].legend()
    # axs[1, 2].set_title('96th row depth (blue line)')
    #
    # fig.savefig(os.path.join(r'C:\Users\919im\Documents\local-?????????\pytorchDeepToF\numpy_sensor_data','four_phase_method_{}.png'.format(iters)))
    # plt.show()


    # fig.add_subplot(2, 4, 7)
    # plt.plot(depth_img[128//2, :], label="depth")
    # plt.plot(gen_img[128//2, :], label="generate")
    # plt.legend(bbox_to_anchor=(1, 1))
    # fig.savefig(os.path.join(eval_folder_name, 'eval_gen_image_{}.png'.format(iters)))



if __name__ == '__main__':
    dataset = ToFDataset(parent_path=r"/home/saijo/labwork/dataset_shizuoka",
                              transform=transforms.Compose([
                                  ToTensor(),
                                  RandomCrop(),
                                  ImageNormalize(),
                              ]))
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    generator = Generator()
    generator.load_state_dict(torch.load(r'/home/saijo/labwork/local-?????????/pytorchDeepToF/results/sample_cat_delay1/gen_3999times_model.pt')['model_state_dict'])
    generator.eval()


    for i_batch, sample_batched in enumerate(dataloader):
        cap_img, cap_ideal_img, depth_img,z_corr = sample_batched['cap'], sample_batched['cap_ideal'], \
                                            sample_batched['depth_1024'], sample_batched['phase_data']

        print(cap_img.shape)


        gen_batch = generator(sample_batched['cap'])
        sample_batched['gen'] = gen_batch
        show_gen_image(cap_img, cap_ideal_img, depth_img, gen_batch,z_corr[0,:,:], i_batch)
        # observe 4th batch and stop.
        if i_batch == 3:
            break