import numpy as np
from scipy.ndimage import convolve

def my_ssim_index_new(img1, img2, K=None, win=None):
    if K is None:
        K = [0.01, 0.03]
    if win is None:
        win = np.ones((11, 11)) / (11 * 11)  # Default Gaussian window

    if img1.shape != img2.shape:
        return -np.inf, -np.inf, -np.inf, -np.inf

    M, N = img1.shape

    if (M < 11) or (N < 11):
        return -np.inf, -np.inf, -np.inf, -np.inf

    if len(K) != 2 or any(k < 0 for k in K):
        return -np.inf, -np.inf, -np.inf, -np.inf

    C1 = (K[0] * 255) ** 2
    C2 = (K[1] * 255) ** 2
    win = win / np.sum(win)

    mu1 = convolve(img1, win, mode='constant', cval=0.0)
    mu2 = convolve(img2, win, mode='constant', cval=0.0)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = convolve(img1 * img1, win, mode='constant', cval=0.0) - mu1_sq
    sigma2_sq = convolve(img2 * img2, win, mode='constant', cval=0.0) - mu2_sq
    sigma12 = convolve(img1 * img2, win, mode='constant', cval=0.0) - mu1_mu2

    if C1 > 0 and C2 > 0:
        ssim_map = (sigma12 + C2) / (np.sqrt(sigma1_sq) * np.sqrt(sigma2_sq) + C2)
        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_map = np.ones(mu1.shape)
        index = (denominator1 * denominator2 > 0)
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

        cs_map = np.ones(mu1.shape)
        index = denominator2 > 0
        cs_map[index] = numerator2[index] / denominator2[index]

    mssim = np.mean(ssim_map)
    mcs = np.mean(cs_map)

    return mssim, ssim_map, mcs, cs_map

def msssim(img1, img2, K=None, win=None, level=5, weight=None, method='product'):
    # Multi-scale Structural Similarity Index (MS-SSIM)
    
    if K is None:
        K = [0.01, 0.03]
    
    if win is None:
        win = np.outer(np.hanning(11), np.hanning(11))  # Gaussian window
    
    if weight is None:
        weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    
    if img1.shape != img2.shape:
        return -np.inf
    
    M, N = img1.shape
    if M < 11 or N < 11:
        return -np.inf
    
    if len(K) != 2 or K[0] < 0 or K[1] < 0:
        return -np.inf
    
    H, W = win.shape
    if (H * W) < 4 or (H > M) or (W > N):
        return -np.inf
    
    if level < 1:
        return -np.inf
    
    min_img_width = min(M, N) / (2 ** (level - 1))
    max_win_width = max(H, W)
    if min_img_width < max_win_width:
        return -np.inf
    
    if len(weight) != level or sum(weight) == 0:
        return -np.inf
    
    if method not in ['wtd_sum', 'product']:
        return -np.inf
    
    downsample_filter = np.ones((2, 2)) / 4
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    
    mssim_array = np.zeros(level)
    mcs_array = np.zeros(level)
    ssim_map_array = []
    cs_map_array = []
    
    for l in range(level):
        mssim_array[l], ssim_map, mcs_array[l], cs_map = my_ssim_index_new(im1, im2, K, win)
        filtered_im1 = convolve(im1, downsample_filter, mode='reflect')
        filtered_im2 = convolve(im2, downsample_filter, mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    
    if method == 'product':
        overall_mssim = np.prod(mcs_array[:level-1] ** weight[:level-1]) * (mssim_array[level-1] ** weight[level-1])
    else:
        weight = weight / sum(weight)
        overall_mssim = np.sum(mcs_array[:level-1] * weight[:level-1]) + mssim_array[level-1] * weight[level-1]
    
    return overall_mssim