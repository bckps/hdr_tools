import cv2
import numpy as np
import matplotlib.pyplot as plt

# hdr_path = r"C:\Users\919im\PythonSandbox\hdr_process\hdrs\bathroom_b01_0000.hdr"
# hdr_path = r"C:\Users\919im\Downloads\simulator-20211026T061954Z-001\simulator\name_file_b01_0000.hdr"
# hdr_path = r"bathroom-small-sample-00004.hdr"
# hdr_path = r"bathroom-small-simu1-00003.hdr"
hdr_path = r"/home/saijo/labwork/simulator_origun/scene-results/single-plane-10ps/00001/hdrs/single-plane-10ps-00001_depth.hdr"
hdr_path = r"/home/saijo/labwork/simulator_origun/scene-results/rect-corner/00000/hdrs/corner-plane-no-del-00000.hdr"

img = cv2.imread(hdr_path, flags=cv2.IMREAD_ANYDEPTH)
print(np.max(img))
tonemap1 = cv2.createTonemap(2.2)
mapped_img = tonemap1.process(img)
print(mapped_img.shape)

res_8bit_img = np.clip(mapped_img*255, 0, 255).astype('uint8')
plt.imshow(res_8bit_img[:,:,2], cmap='gray')
plt.show()
# plt.imsave('test.png', image)


# cv2.imshow('8bit',res_8bit_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite("bathroom_8bit.jpg", res_8bit_img)