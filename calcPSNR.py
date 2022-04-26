import cv2
import numpy as np
import os

from skimage.metrics import peak_signal_noise_ratio as _PSNR
from skimage.metrics import structural_similarity as _SSIM
from scipy.stats import wasserstein_distance


for missing_bits in [1, 2, 3, 4, 5, 6, 7]:
    psnr_total = 0
    wdis_total = 0
    ssim_total = 0
    for img_index in range(24):
        hr = cv2.imread("Result//%d//%d-HBD.png" % (missing_bits, img_index))
        lr = cv2.imread("Result//%d//%d-LBD.png" % (missing_bits, img_index))
        sr = cv2.imread("Result//%d//%d-RBD.png" % (missing_bits, img_index))
        psnr = _PSNR(hr, sr)
        wdis = wasserstein_distance(hr.reshape(-1), sr.reshape(-1))
        hr, sr = cv2.cvtColor(hr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(sr, cv2.COLOR_BGR2GRAY)
        ssim = _SSIM(hr, sr)
        psnr_total += psnr
        wdis_total += wdis
        ssim_total += ssim
    print("Missing Bits:%d "   %(missing_bits))
    print("Avg PSNR:%.4f"  %(psnr_total/24))
    print("Avg SSIM:%.4f"  %(ssim_total/24))
    print("Avg WDIS:%.4f"  %(wdis_total/24))
    print()
