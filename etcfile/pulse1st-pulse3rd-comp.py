import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    file = '/home/saijo/labwork/PythonSandbox/hdr_process/npz/contemporaly-bathroom-extended-train2/00019/00019-contemporaly-bathroom-extended-train2-phase-depth.npz'
    # file = '/home/saijo/labwork/PythonSandbox/hdr_process/npz/bathroom-extended-train2/00036/00036-bathroom-extended-train2-phase1st-depth.npz'
    print(file)
    with np.load(file) as data:
        phase = data.get('arr_0')
    # file = '/home/saijo/labwork/PythonSandbox/hdr_process/npz/bathroom-extended-train2/00036/00036-bathroom-extended-train2-phase1st-depth.npz'
    # print(file)
    # with np.load(file) as data:
    #     phase1st = data.get('arr_0')

    file = '/home/saijo/labwork/PythonSandbox/hdr_process/npz/contemporaly-bathroom-extended-train2/00019/00019-contemporaly-bathroom-extended-train2-ground-truth.npz'
    print(file)
    with np.load(file) as data:
        gt = data.get('arr_0')
    # print(np.sum(phase1st-gt))
    # print(np.sum(phase-gt))

    plt.subplot(121)
    plt.imshow(gt)
    plt.subplot(122)
    plt.imshow(phase)

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.2, 0.03, 0.6])
    pp = plt.colorbar(cax=cax)
    # plt.clim(0, 3.5)
    plt.show()