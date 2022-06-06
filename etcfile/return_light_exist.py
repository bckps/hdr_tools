import cv2, imageio
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import os, json

load_xyt = r"/home/saijo/labwork/PythonSandbox/hdr_process/xyt-npz/angle-light-power/00024/angle-light-power-xyt-inpulse.npz"
with np.load(load_xyt) as data:
    xyt_img = data.get('arr_0')

xy_img = np.sum(xyt_img, axis=2)
simu_exist = np.where(xy_img>0,1,0)

png_img = imageio.imread(r'/home/saijo/labwork/PythonSandbox/hdr_process/xyt-npz/angle-light-power/00024/untitled4.png')
png_img = np.sum(png_img[:,:,:2], axis=2)
png_img = np.where(png_img>10,1,0)

plt.figure()
plt.imshow(simu_exist)
plt.show()

plt.figure()
plt.imshow(png_img)
plt.show()

plt.figure()
plt.imshow(png_img-simu_exist)
plt.show()