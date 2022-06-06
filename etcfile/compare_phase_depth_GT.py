import os
import numpy as np
import matplotlib.pyplot as plt

with np.load(os.path.join('/home/saijo/labwork/PythonSandbox/hdr_process/npz/bathroom-smal2/bathroom-smal2-ground-truth.npz'), 'r') as data:
    ground_truth = data.get('arr_0')

with np.load(os.path.join('/home/saijo/labwork/PythonSandbox/hdr_process/npz/bathroom-smal2/bathroom-smal2-phase-depth.npz'), 'r') as data:
    phase_depth = data.get('arr_0')


print(ground_truth.shape)
print(phase_depth.shape)

plt.figure()
plt.imshow(phase_depth)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(ground_truth)
plt.colorbar()
plt.show()


print(phase_depth-ground_truth)

plt.figure()
plt.imshow(phase_depth-ground_truth, cmap="turbo", vmin=0, vmax=1.5)
plt.colorbar()
plt.show()