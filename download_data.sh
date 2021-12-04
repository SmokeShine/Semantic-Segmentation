#!/bin/bash

echo -n "Downloading Labelled Data"
wget -nc http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/data/LabeledApproved_full.zip
echo -n "Downloading Raw Images"
wget -nc http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip

echo -n "Unzipping Data"
unzip LabeledApproved_full.zip.3 -d data/raw/LabeledApproved_full
unzip 701_StillsRaw_full.zip -d data/raw/701_StillsRaw_full