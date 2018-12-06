# Torch

This repository for DBDN in the following paper https://arxiv.org/abs/1810.04873

The code is built on EDSR Torch7, under Ubuntu16.04 cuDNN environment with Titan X GPUs

# Train
  Prepare training data 
  1. Download DIV2K training data (800 training +100 validation) from https://data.vision.ee.ethz.ch/cvl/DIV2K/
     Download Flickr2K training data from https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
  2. Place HR images in 'code/Prepare_TrainData/DIV2K/DIV2K_HR' and 'code/Prepare_TrainData/Flickr2K/Flickr2K_HR'
  3. Run 'Prepare_TrainData_HR_LR_DIV2K.m' and 'Prepare_TrainData_HR_LR_Flickr2K.m' in matlab to generate LR images
  4. cd code/Prepare_TrainData and Run "th png_to_t7.lua' to convert .png images to .t7 files in new folder 'DIV2K_decoded' and 'Flickr2K_decoded'

  Begin train
  1. Download models for the paper from "https://www.dropbox.com/home/DBDN" and shore them into 'experiment/model/'
  2. cd 'code', use the scripts in file 'Train_scripts.sh' to train the models 


# Test
1.  Download models for the paper from "https://www.dropbox.com/home/DBDN" and shore them into 'test/model'
2.  Prepare test data, Run 'Prepare_TestData_HR_LR' in Matlab to generate HR/LR images with different degradation models.
3.  Run'Test.lua' based on the code in the 'Test_scripts'.
4.  Run' Evaluation_PSNR_SSIM.m' to obtain PSNR/SSIM results.

# Citation
if you find the code helpful in your research or work, please cite the following papers.
```
@article{wang2018deep,
  title={Deep Bi-Dense Networks for Image Super-Resolution},
  author={Wang, Yucheng and Shen, Jialiang and Zhang, Jian},
  journal={arXiv preprint arXiv:1810.04873},
  year={2018}
}
```

   
