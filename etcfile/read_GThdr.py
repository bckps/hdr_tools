import cv2
import numpy as np
import matplotlib.pyplot as plt

# hdr_path = r"C:\Users\919im\PythonSandbox\hdr_process\hdrs\bathroom_b01_0000.hdr"
# hdr_path = r"C:\Users\919im\Downloads\simulator-20211026T061954Z-001\simulator\name_file_b01_0000.hdr"
# hdr_path = r"bathroom-small-sample-00004.hdr"
# hdr_path = r"bathroom-small-simu1-00003.hdr"
# hdr_path = r"/home/saijo/labwork/simulator_origun/scene-results/materialtest-living-room/00000/hdrs0/materialtest-living-room-00000_depth.hdr"
# hdr_path = r"/home/saijo/labwork/simulator_origun/scene-results/light_offset_zero/00062/hdrs/bedroom-test-00062_depth.hdr"
hdr_path = r"/home/saijo/labwork/simulator_origun/scene-results/double-plane-connection-suppress/00000/hdrs/double-plane-connection-suppress-00000_depth.hdr"
# hdr_path = r"/home/saijo/labwork/simulator_origun/scene-results/light_offset_zero/00050/hdrs/bedroom-test-00050_depth.hdr"

hdr2_path = r"/home/saijo/labwork/simulator_origun/scene-results/light_offset_zero/00054/hdrs/bedroom-test-00054_depth.hdr"
# hdr_path = r"/home/saijo/labwork/simulator_origun/scene-results/light_offset_zero/00045/hdrs/bedroom-test-00045_depth.hdr"

img = np.array(cv2.imread(hdr_path, flags=cv2.IMREAD_ANYDEPTH))
print(np.max(img))
print(np.sum(img[:,:,0]-img[:,:,2]))
print(img.shape)

plt.imshow(img[:,:,0], cmap='gray')
plt.colorbar()
plt.show()
# plt.imsave('test.png', image)


# cv2.imshow('8bit',res_8bit_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite("bathroom_8bit.jpg", res_8bit_img)
img2 = np.array(cv2.imread(hdr2_path, flags=cv2.IMREAD_ANYDEPTH))
print(np.max(img2))
print(np.sum(img[:,:,0]-img2[:,:,0]))
print(img2.shape)

plt.imshow(img2[:,:,0], cmap='gray')
plt.colorbar()
plt.show()