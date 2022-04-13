#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd


# In[3]:


# file paths

IMGFOLDER = 'myprac/'
COLORMASKFOLDER = 'myprac/color_masks/'
BIMASKFOLDER = 'myprac/bi_coded_masks/'
CSVFILE = 'myprac/bbox.csv'
IMGOUT = 'myprac/imgs/'
FINALMASKOUT = 'myprac/masks/'


# In[4]:


# setup 

# current works best for BnW images

THRESHOLD = 225
MAXVALUES = 128


# In[ ]:


# label names

LABEL1 = 'margin'
LABEL2 = 'text'
LABEL3 = 'image_capital'
LABEL4 = 'noise'
LABEL5 = 'background'


# In[ ]:


# bib ext

# https://www.tutorialspoint.com/how-to-apply-a-mask-on-the-matrix-in-matplotlib-imshow
# https://matplotlib.org/3.5.0/gallery/images_contours_and_fields/image_masked.html
# https://stackoverflow.com/questions/56766307/plotting-with-numpy-masked-arrays
# https://stackoverflow.com/questions/31877353/overlay-an-image-segmentation-with-numpy-and-matplotlib
# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_masked.html
# https://matplotlib.org/3.4.3/tutorials/intermediate/artists.html?highlight=layer%20image

# https://www.blog.pythonlibrary.org/2021/02/23/drawing-shapes-on-images-with-python-and-pillow/
# https://note.nkmk.me/en/python-pillow-composite/

