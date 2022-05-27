# Master Thesis
Near-IR imaging and ML to estimate tooth & gum status

## Description
This project is the master thesis of Huitong Lu in FAU.
The repository includes all the contents about network training, image fusion and image processing functions.

.\teeth_dataset is the dataset for all the 320 images, 80 annotation JSON files, corresponding masks and the spreadsheet

.\dummy2 includes the dummy dataset.

.\fusion_workplace is the workplace for image fusion. You can get the fusion result here with the name 'fusion.bmp'.

.\prediction_workplace is the workplace for CNN prediction. 'origin.bmp' is the input image. 'prediction.bmp' is the direct result from
network. 'prediction_threshold.bmp' is the result after manipulation. 'masked.bmp' is the image with a mask. 

.\train includes all the python files of training. The structure of Unet is in .\unet.

'best_caries.pth' is the best weight for caries estimation.

'best_seg.pth' is the best weight for contour segmentation.

You can either explore the code by yourself or use the functions listed below.

**Note : Some documents are hidden for confidentiality reason.**

## Install
1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch](https://pytorch.org/get-started/locally/)

3. Install OpenCV
```bash
pip install opencv-python
```

4. Install pandas
```bash
pip install pandas
```

5. Install scikit-learn
```bash
pip install -U scikit-learn
```
Also, we recommend to install Anaconda for better package management and environment configuration.

## Usage

**Note : Use Python 3.6 or newer**

This project has three funtions.

### Contour Segmentation and Caries Estimation
These two functions are integrated in predict.py.
You need to change 'TESTPATH' for your desired image.
And change 'FLAG' to decide whether contour segmentation and caries estimation.
Then run predict.py
```bash
python predict.py
```
You can get the result in .\prediction_workplace
### Image Fusion
You need to change 'FUSIONPATH1' and 'FUSIONPATH2' in the code for your desired two images.
Then run image_fusion.py
```bash
python image_fusion.py
```
You can get the result in .\fusion_workplace


