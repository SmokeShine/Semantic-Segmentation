# Team_SegNet

# Project Title: Analyzing Semantic Segmentation on Far and Near Perspective Datasets


## Project Summary:
Semantic segmentation is used to label specific regions of an image using computer vision. It has a wide array of applications ranging from scene understanding, medical imaging, inferring support-relationships among objects to autonomous driving. Our motivation for this project is to explore one of the common neural architecture SegNet for efficient mapping of low-resolution feature maps to input resolution for pixel-wise classification. \\
We try to reproduce the SegNet approach using two datasets :CamVid and Pascal VOC. Images present in CamVid are near perspective while images present in Pascal VOC are far perspective. This could be helpful in improving model understanding across different focal ranges. We also experimented with varying the strength of data augmentation to monitor the model prediction and accuracy. To check the adaptation of the models to a more generalized setting, we also cross inferenced the model between the two datasets and check the mean IOU (Intersection over union). 
## Motivation:
Semantic segmentation has been a field of study where constant improvements are sought, to have a precise object and scene understanding. A good segmentation results in better understanding for the machines about the world and various industrial projects eg. automotive, medical etc. can leverage from that. For our projects, we start with understanding basics of semantic segmentation and then train a model with different datasets to analyze the training, prediction and scene challenges.

There are a number of datasets available to experiment with the segmentation. Each dataset has has been created keeping in mind an end application. SUN-RGB has usecase with indoor segmentation, CamVid is focused towards road scene segmentation. Similarly, comma10k is focussed towards commerical usage of self driving car specialized towards near vision focal length.\\
Semantic Segmentation has been an active field of research for a long time. Today the research is focused on architecture of the network, data augmentation and loss functions that suits the segmentation catered to domains such as medical imaging and intelligent transportation. 

In recent years, image super-resolution has caught a lot of attention, where the model is trained to produce high-definition images from low resolution. It has a lot of practical applications, like efficient data storage, high quality video calling etc.\\
For semantic segmentation, lower resolution images results in poorer segmentation, due to loss of information, especially smaller features. But humans have an innate ability to detect and segment in spite of the low quality inputs. This is due to the fact that, humans interact with objects in closer and farther settings, which enables us to generalise the objects better. We have tried to train and analyse the deviations in the labels in low resolution. We believe that this analysis could help in producing high quality semantic segmentation on further research.\\

Our research is primarily focussed on two open source datasets:
\textbf{CamVid} \href{http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/}{link}\\
It is the Cambridge-driving Labeled Video Database that has image snapshots taken from a traffic road scenes. It has 32 semantic classes commonly found on a road scene like cars, pavement, buildings trees etc. The images are of 960X720 dimension with a total of 701 labelled images in the dataset. The images are captured from a car dashboard introducing a persepective distortion in the objects down the length of the road.

## How to run the code
1. Create conda environment with the provided environment.yml
2. Change permissions for download_data.sh to 777 or run chmod +x download_data.sh
3. Run the shell script to dowload the data
4. For training CamVid following arguments are presents: 
```
usage: main.py [-h] [--gpu] [--train] [--batch_size [BATCH_SIZE]] [--num_workers [NUM_WORKERS]]
               [--num_epochs [NUM_EPOCHS]] [--num_output_classes [NUM_OUTPUT_CLASSES]]
               [--learning_rate [LEARNING_RATE]] [--sgd_momentum [SGD_MOMENTUM]]
               [--plot_output_path PLOT_OUTPUT_PATH] [--model_path MODEL_PATH]
               [--epoch_save_checkpoint [EPOCH_SAVE_CHECKPOINT]]
               [--split_percentage [SPLIT_PERCENTAGE]] [--patience [PATIENCE]]
               [--transforms TRANSFORMS] [--pred_model PRED_MODEL]

Final Group Project - Semantic Segmentation

optional arguments:
  -h, --help            show this help message and exit
  --gpu                 Use GPU for training (default: True)
  --train               Train Model (default: False)
  --batch_size [BATCH_SIZE]
                        Batch size for training the model (default: 1)
  --num_workers [NUM_WORKERS]
                        Number of Available CPUs (default: 5)
  --num_epochs [NUM_EPOCHS]
                        Number of Epochs for training the model (default: 10)
  --num_output_classes [NUM_OUTPUT_CLASSES]
                        Number of output class for semantic segmentation (default: 32)
  --learning_rate [LEARNING_RATE]
                        Learning Rate for the optimizer (default: 0.01)
  --sgd_momentum [SGD_MOMENTUM]
                        Momentum for the SGD Optimizer (default: 0.5)
  --plot_output_path PLOT_OUTPUT_PATH
                        Output path for Plot (default: ./Plots_)
  --model_path MODEL_PATH
                        Model Path to resume training (default: None)
  --epoch_save_checkpoint [EPOCH_SAVE_CHECKPOINT]
                        Epochs after which to save model checkpoint (default: 5)
  --split_percentage [SPLIT_PERCENTAGE]
                        Train and Validation data split (default: 0.1)
  --patience [PATIENCE]
                        Early stopping epoch count (default: 3)
  --transforms TRANSFORMS
                        Transforms to be applied to input dataset. options
                        (posterize,sharpness,contrast,equalize,crop,hflip). comma-separated list
                        of transforms. (default: sharpness,contrast,equalize,crop,hflip)
  --pred_model PRED_MODEL
                        Model for prediction; Default is checkpoint_model.pth; change to
                        ./best_model.pth for 1 sample best model (default:
                        ./checkpoint_model.pth)
```
5. Go to src folder; python main.py for inference

## Resources/Related Work:
[1]  Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla  “SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation” , 1511.00561.pdf (arxiv.org)

[2] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014.

[3] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in CVPR, pp. 3431–3440, 2015.

[4] C. Liang-Chieh, G. Papandreou, I. Kokkinos, K. Murphy, and A. Yuille, “Semantic image segmentation with deep convolutional nets and fully connected crfs,” in ICLR, 2015.

[5] H. Noh, S. Hong, and B. Han, “Learning deconvolution network for semantic segmentation,” in ICCV, pp. 1520–1528, 2015

[6] Olaf Ronneberger, Philipp Fischer, Thomas Brox “U-Net: Convolutional Networks for Biomedical Image Segmentation” in MICCAI, 2015

[7] https://github.com/commaai/comma10k



### The trained models can be downloaded from the below links:}

1 SegNet trained on Downsampled CamVid Data - https://gtvault-my.sharepoint.com/:f:/g/personal/pgupta353_gatech_edu/EvaBdCYGVlFOn-MoiAGIIEwBFj6g-H9TL_j4uuH_lycxRQ?e=tuzdtH

2 SegNet trained on Full Resolution CamVid Data - https://drive.google.com/file/d/1lOl3FKDaE-rXw9PYhqAPjKgB3_gFjfeZ/view?usp=sharing

3 SegNet trained on PascalVOC - https://drive.google.com/file/d/1joh4JLBdNBF_mn6JNixeDePxZOZVS8EC/view?usp=sharing
\end{enumerate}

### Team Members:
1. Shubham Soni
1. Eldho Abraham
1. Prateek Gupta
1. Abhinav Tankha

