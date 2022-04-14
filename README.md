# prep_fasterRCNN_anno
This is a quick tool for creating image masks used for training a FasterRCNN object detection module. Object detection takes several inputs including: image data, bbox dims, object labels, and image data masks. In order to quickly prepare all this data with one tool, this script will automatically create masks from a simple instance of  bbox annotations created with VGG image annotator.

# Basic Usage:

create an annotated image using VGG image annotator. https://www.robots.ox.ac.uk/~vgg/software/via/via.html
Using image annotator you can create up to 5 different annotation labels. This works best if you follow the annotation automasking machine guide found here: https://thebabellibrary.org/blog/preparing-data-for-object-detection-using-fasterrcnn.

Change any neccesary file paths, label names and associated color values, or image binarization thresholds in the mask_config.py file. The current file uses this directory structure.

# input directory structure:

pwd:

  |-myprac
  
  |   |-img1.jpg
  
  |   |-img2.jpg 
  
  |   |-img3.jpg, ect
  
  |   |-bbox_anno.csv
  
  |-mask_config.py
  
  |-mask_mker.py
  
  
# output directory structure:

pwd:

|-myprac

|  |-img1.jpg

|  |-img2.jpg

|  |-img3.jpg, ect

|  |-bbox.csv

|  |-bi_coded_masks (new folder containing binarized images)

|  |   |-bi_img1.jpg

|  |   |-bi_img2.jpg

|  |   |-bi_img3.jpg, ect

|  |-color_masks (new folder containing color blocked images)

|  |   |-color_img1.jpg

|  |   |-color_img2.jpg

|  |   |-color_img3.jpg, ect

|  |-maks (new folder containing masks that can be used for image segmentation in FasterRCNN training.)

|  |   |-img1.jpg

|  |   |-img2.jpg

|  |   |-img3.jpg, ect

# prereqs 

PIL v.1 
Numpy v.1
Pandas v.1 


 
