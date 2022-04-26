import cv2
import os

for missing_bits in [1, 2, 3, 4, 5, 6, 7]:
    if not os.path.exists("Kodim/%d"%(missing_bits)):
        os.makedirs("Kodim/%d"%(missing_bits))

for imname in os.listdir("Kodim/GT"):
    gt = cv2.imread("Kodim/GT/"+imname)
    for missing_bits in [1, 2, 3, 4, 5, 6, 7]:
        octave = 2 ** missing_bits
        lbd = gt // octave * octave
        cv2.imwrite("Kodim/%d/%s"%(missing_bits, imname), lbd)

print('done.')