import numpy as np
import matplotlib.pyplot as plt
import os

folder = '/home/saijo/labwork/研究結果まとめ/comp_1st_facet_diff'
with np.load('xyt-npz/corner-plane-no-del/00001/corner-plane-no-del-xyt-inpulse.npz') as data:
    facet3 = data['arr_0']

with np.load('xyt-npz/corner-plane-del1/00001/corner-plane-del1-xyt-inpulse.npz') as data:
    facet2 = data['arr_0']

with np.load('xyt-npz/corner-plane-del2/00001/corner-plane-del2-xyt-inpulse.npz') as data:
    facet1 = data['arr_0']


imx, imy = 200, 50
print(np.sum(facet3,axis=2)[imy,imx]-np.sum(facet1,axis=2)[imy,imx])
print(np.sum(facet3,axis=2)[imy,imx],np.sum(facet1,axis=2)[imy,imx])
# print(np.sum(np.sum(facet3,axis=2)[imx,imy]-np.sum(facet1,axis=2)[imx,imy]))

print(facet1.shape)
plt.figure()
plt.imshow(np.sum(facet1,axis=2))
plt.scatter(imx, imy, 25, 'red')

plt.colorbar()
plt.savefig(os.path.join(folder,f'({imx},{imy})_facet1.png'))

# plt.show()

plt.figure()
plt.imshow(np.sum(facet3,axis=2))
plt.scatter(imx, imy, 25, 'red')
plt.colorbar()
plt.savefig(os.path.join(folder,f'({imx},{imy})_facet3.png'))
# plt.show()

diff_array = np.sum(facet3,axis=2)-np.sum(facet1,axis=2)
plt.figure()
plt.imshow(diff_array)
plt.colorbar()
plt.scatter(imx, imy, 25, 'red')
plt.text(imx,imy+20,f'{diff_array[imy,imx]:.2f}',color="#FF0000")
plt.savefig(os.path.join(folder,f'({imx},{imy})_facet_diff.png'))
# plt.scatter(imx, imy, 25, 'red')
plt.show()