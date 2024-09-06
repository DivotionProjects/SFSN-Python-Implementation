import numpy as np
import cv2

from compute_sf_sn import compute_sf_sn

imgref = np.double(cv2.cvtColor(cv2.imread('ref.bmp'), cv2.COLOR_BGR2GRAY))
imgdis = np.double(cv2.cvtColor(cv2.imread('dis.bmp'), cv2.COLOR_BGR2GRAY))

sf, sn = compute_sf_sn(imgref, imgdis)

print(sf)
print(sn)
pred_score = (1 + 0.9) * sf + (1 - 0.9) * sn
print(pred_score)