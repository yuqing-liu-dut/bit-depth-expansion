# Learning Weighting Map for Bit-Depth Expansion within a Rational Range

## Abstract
Bit-depth expansion (BDE) is one of the emerging technologies to display high bit-depth (HBD) image from low bit-depth (LBD) source. Existing BDE methods have no unified solution for various BDE situations, and directly learn a mapping for each pixel from LBD image to the desired value in HBD image, which may change the given high-order bits and lead to a huge deviation from the ground truth. In this paper, we design a bit restoration network (BRNet) to learn a weight for each pixel, which indicates the ratio of the replenished value within a rational range, invoking an accurate solution without modifying the given high-order bit information. To make the network adaptive for any bit-depth degradation, we investigate the issue in an optimization perspective and train the network under progressive training strategy for better performance. Moreover, we employ Wasserstein distance as a visual quality indicator to evaluate the difference of color distribution between restored image and the ground truth. Experimental results show our method can restore colorful images with fewer artifacts and false contours, and outperforms state-of-the-art methods with higher PSNR/SSIM results and lower Wasserstein distance.

## Usage

you can easily find three python files: "BRNet.py", "buildLBDinputs.py" and "calcPSNR.py". Our network is implemented by PyTorch 0.4.1 on one NVIDIA GTX-1080Ti GPU. To reproduce the results on Kodak dataset of our manuscript, please run the codes as follows.

1. run "buildLBDinputs.py" to generate the LBD inputs of Kodak image dataset with different bit depth.
2. run "BRNet.py" to generate the restored images from LBD inputs.
3. run "calcPSNR.py" to calculate the PSNR/SSIM/W-dis results.

## Citation
Please kindly cite our paper when using this project for your research.

'''
  @misc{liu2022,
    doi = {10.48550/ARXIV.2204.12039}, 
    author = {Liu, Yuqing and Jia, Qi and Zhang, Jian and Fan, Xin and Wang, Shanshe and Ma, Siwei and Gao, Wen},  
    title = {Learning Weighting Map for Bit-Depth Expansion within a Rational Range},  
    publisher = {arXiv},  
    year = {2022},  
  }
'''
