import numpy as np
from scipy.fftpack import dct, idct
from skimage.transform import resize
from skimage.measure import shannon_entropy as entropy

from msssim import msssim

def compute_sf_sn(imgref, imgdis):
    m, n = imgdis.shape
    s = min(m, n)
    Orig_ref = resize(imgref, (s, s))
    
    # Transform
    Orig_T_ref = dct(dct(Orig_ref.T, norm='ortho').T, norm='ortho')
    
    # Split between high- and low-frequency in the spectrum
    cutoff = round(0.5 * s)
    High_T_ref = np.fliplr(np.tril(np.fliplr(Orig_T_ref), cutoff))
    Low_T_ref = Orig_T_ref - High_T_ref
    
    # Transform back
    High_ref = idct(idct(High_T_ref.T, norm='ortho').T, norm='ortho')
    Low_ref = idct(idct(Low_T_ref.T, norm='ortho').T, norm='ortho')

    Orig = resize(imgdis, (s, s))
    
    # Transform
    Orig_T = dct(dct(Orig.T, norm='ortho').T, norm='ortho')
    
    # Split between high- and low-frequency in the spectrum
    High_T = np.fliplr(np.tril(np.fliplr(Orig_T), cutoff))
    Low_T = Orig_T - High_T
    
    # Transform back
    High = idct(idct(High_T.T, norm='ortho').T, norm='ortho')
    Low = idct(idct(Low_T.T, norm='ortho').T, norm='ortho')

    sf = msssim(Low_ref.astype(np.uint8), Low.astype(np.uint8))
    sn = entropy(High.astype(np.uint8))
    
    return sf, sn

