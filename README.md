# Master Thesis
Near-IR imaging and ML to estimate tooth & gum status

## Description
This project is the master thesis of Huitong Lu at FAU.

The repository includes all the contents about NIR images, network training, image fusion and image processing functions.

.\teeth_dataset is the NIR dataset for all the 320 images + some new molar samples, 80 annotation JSON files, corresponding masks and the spreadsheet

.\dummy2 includes the dummy dataset.

.\fusion_workplace is the workplace for image fusion. You can get the fusion result here with the name 'fusion.bmp'.

.\prediction_workplace is the workplace for CNN prediction. 'origin.bmp' is the input image. 'prediction.bmp' is the direct prediction result from
the network. 'prediction_threshold.bmp' is the result after manipulation. 'masked.bmp' is the image with a mask. 

'best_caries.pth' is the best weights for caries estimation.

'best_seg.pth' is the best weights for contour segmentation.

.\train includes all the python files of training and evaluation. 
The structure of Unet is in .\unet. The CV implementation is in 'training_procedure.py'. The IoU test is in 'iou.py'.

You can either explore the code by yourself or use the functions listed below.

**Note: Some documents are hidden for confidentiality reasons. **

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
Also, we recommend installing Anaconda for better package management and environment configuration.

## Usage

**Note: Please use Python 3.7 or newer**

This project has three functions.

### Contour Segmentation and Caries Estimation
These two functions are integrated in 'predict.py'.
You need to change the constant 'TESTPATH' to your desired image.
And change the constant 'FLAG' to decide whether to use contour segmentation or caries estimation.
Then run predict.py
```bash
python predict.py
```
You can get the results in .\prediction_workplace
### Image Fusion
You need to change the constants 'FUSIONPATH1' and 'FUSIONPATH2' in the code to your desired two images.
Then run 'image_fusion.py'
```bash
python image_fusion.py
```
You can get the results in .\fusion_workplace


