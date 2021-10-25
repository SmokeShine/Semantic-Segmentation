# Team_SegNet

# Project Title: Pixel wise semantic segmentation


## Project Summary:
Semantic segmentation has a wide array of applications ranging from scene understanding, medical imaging, inferring support-relationships among objects to autonomous driving. Our motivation for this project is to explore the network architectures for efficient mapping of low-resolution feature maps to input resolution for pixel-wise classification. This mapping must produce features which are useful for accurate boundary localization.

## Approach:
Based on our preliminary research, there are multiple approaches to address pixel wise semantic segmentation. We want to start by collecting and analysing varying approaches; e.g. DeconvNet, UNet, DeepLab-LargeFOV, DeepLab-LargeFOV- denseCRF and FCN. We expect to reproduce the SegNet approach and baseline other approaches.
As a stretch goal, we want to explore possible directions to improve the limitations of these papers. One avenue is to augment the data to provide multiple views of the same scene and ensure prediction consistency. There are a lot of methods in computer vision that utilizes the global and local features for efficient segmentation especially when multi-view of the scene is considered. We would also like to explore if such methods can be part of the decoding architecture to improve the segmentation.

Another stretch goal is to come up with a new metric and loss terms (e.g. incorporating physical constraints) to improve benchmarks.

## Resources/Related Work:
[1]  Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla  “SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation” , 1511.00561.pdf (arxiv.org)

[2] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014.

[3] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in CVPR, pp. 3431–3440, 2015.

[4] C. Liang-Chieh, G. Papandreou, I. Kokkinos, K. Murphy, and A. Yuille, “Semantic image segmentation with deep convolutional nets and fully connected crfs,” in ICLR, 2015.

[5] H. Noh, S. Hong, and B. Han, “Learning deconvolution network for semantic segmentation,” in ICCV, pp. 1520–1528, 2015

[6] Olaf Ronneberger, Philipp Fischer, Thomas Brox “U-Net: Convolutional Networks for Biomedical Image Segmentation” in MICCAI, 2015

[7] https://github.com/commaai/comma10k


## Datasets:
SUN RGB-D : https://rgbd.cs.princeton.edu/data/SUNRGBD.zip


### Team Members:
1. Shubham Soni
1. Eldho Abraham
1. Prateek Gupta
1. Abhinav Tankha

