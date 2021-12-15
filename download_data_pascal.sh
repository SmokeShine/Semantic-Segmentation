#!/bin/bash

echo -n "Downloading Full Data"
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # VOC2012 train+val set


echo -n "Unzipping Data"
tar xvf VOCtrainval_11-May-2012.tar 
