import numpy as np
import cv2

from compute_sf_sn import compute_sf_sn

def calc_sfsn(ref_path='ref.bmp', dis_path='dis.bmp'):
    imgref = np.double(cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2GRAY))
    imgdis = np.double(cv2.cvtColor(cv2.imread(dis_path), cv2.COLOR_BGR2GRAY))

    pred_score = (1 + 0.9) * sf + (1 - 0.9) * sn
    return pred_score
