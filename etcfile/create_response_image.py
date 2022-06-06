import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# folder_path = r'C:\Users\919im\PythonSandbox\hdr_process\hdrs'
# folder_path = r'C:\Users\919im\Downloads\simulator-20211026T061954Z-001\simulator'
folder_path = r'C:\Users\919im\Documents\local-香川研\simulator'

files = os.listdir(folder_path)
files.sort()
print(files)
b1_list = []
b2_list = []
b3_list = []
for file in files:
    file_bounce = file.split('_')
    if len(file_bounce)<2:
        continue
    if file_bounce[1] == 'b01':
        b1_list.append(file)
    elif file_bounce[1] == 'b02':
        b2_list.append(file)
    elif file_bounce[1] == 'b03':
        b3_list.append(file)

print(b1_list)
# response_img = np.zeros((256,256,1024,3))
response_img = np.zeros((128,115,512,3))

# if len(b1_list) == 257:
#     print(len(b1_list))
#     b1_list.pop(-1)
#     print(b1_list)

# for i,b1 in enumerate(b1_list):
#     img = cv2.imread(os.path.join(folder_path,b1), flags=cv2.IMREAD_ANYDEPTH)
#     response_img[:,i,:,:] = img[1:257,:,:]
for i,b1 in enumerate(b1_list):
    img = cv2.imread(os.path.join(folder_path,b1), flags=cv2.IMREAD_ANYDEPTH)
    # response_img[:, i, :, :] = img[:, :, :]
    response_img[:,i,:,:] = img[:,:,[2,1,0]]

print(np.max(response_img[:,:,300,0]))
print(np.max(response_img[:,:,300,1]))
print(np.max(response_img[:,:,300,2]))
print(np.max(response_img[:,:,300,:]))

fig = plt.figure()
ims = []
# for i in range(1024):
for i in range(512):
        im = plt.imshow(np.transpose(np.squeeze(response_img[:,:,i,:]),(1, 0, 2))/np.max(response_img[:,:,i,:]), animated=True)
        ims.append([im])                  # グラフを配列 ims に追加
ani = animation.ArtistAnimation(fig, ims, blit=True, interval=3,repeat_delay=10)
plt.show()