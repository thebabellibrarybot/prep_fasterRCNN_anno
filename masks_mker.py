#!/usr/bin/env python
# coding: utf-8

# In[1]:


# this will be used to make masks from images processed with a colormap


# TODO:

# find bboxed train images with labels( bk, text, image_initial, margins )
# mk anno img folder into binary   ------------------------https://note.nkmk.me/en/python-numpy-opencv-image-binarization/
# colormap used to find background, copy shape as blue -------------------------------[info]: done better w binary mask
# place color polygons over areas of importance: -----------------------------------done
#   -  (i.e) red over words, green over images, yellow over noise_margins-----------done
# paste binary maks shape over patched image ---------------------------------------done
# save as segmentation model img ---------------------------------------------------done
# make argparser and config.py
# prac on full img file and image annos


# In[8]:



import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
import mask_config
import argparse


# In[3]:


# argparser: input csv


# In[4]:


df = pd.read_csv(mask_config.CSVFILE, on_bad_lines = 'skip')

class mask_info(torch.utils.data.Dataset):
    
    def __init__(self, csv, root, transform):
        self.root = root
        self.csv = pd.read_csv(csv, on_bad_lines = 'skip')
        self.transform = transform
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # list of info to find
        imgls = []
        boxesls = []
        labelsls = []
        
        # calling info
        img = os.path.join(self.root, self.csv.iloc[idx, 0])
        boxes = self.csv.iloc[idx, 5].split(',')
        x = boxes[1].split(':')[1]
        y = boxes[2].split(':')[1]
        w = boxes[3].split(':')[1]
        h = boxes[4].split(':')[1].split('}')[0]
        boxes = x, y, w, h
        labels = self.csv.iloc[idx, 6].split(':')[1].split('"')[1]
        num_labels = self.csv.iloc[idx, 3]
        
        # append info to list so there is only one list per filename
        labelsls.append(labels)
        boxesls.append(boxes)        
        imgls.append(img)
        for i in range(num_labels):
            if i >= 1:
                img = os.path.join(self.root, self.csv.iloc[idx + i, 0])
                imgls.append(img)
                boxes = self.csv.iloc[idx + i, 5].split(',')
                x = boxes[1].split(':')[1]
                y = boxes[2].split(':')[1]
                w = boxes[3].split(':')[1]
                h = boxes[4].split(':')[1].split('}')[0]
                boxes = x, y, w, h
                boxesls.append(boxes)        
                labels = self.csv.iloc[idx + i, 6].split(':')[1].split('"')[1]
                labelsls.append(labels)
                
# return important info aka: imgpath, box dims, labels, and number of labels per file
        if self.transform:
            img = self.transform(img)
        targets = {}
        targets['img_name'] = imgls
        targets['boxes'] = boxesls
        targets['labels'] = labelsls
        targets['num_labels']= num_labels
        sample = img, targets
        return sample 
        for i in range(num_labels):
            if targets['img_name'] == targets['img_name'][i]:
                if i == len(num_labels):
                    return sample
                             
# mk obj dataset
ds = mask_info(mask_config.CSVFILE, mask_config.IMGFOLDER, transform = None)

# checker fn: plt.show 
for i, d in ds:
    # filter for good data
    num_labels = (len(d['labels']))
    realname = d['img_name'][0]
#    if (d['img_name'][num_labels - 1]) == realname:
#        print(d)
    


# In[5]:


# save binary v. of image (keep this code as one input fn)
# mk binary mask folder
os.mkdir(mask_config.BIMASKFOLDER)
# upload binary masks
for i, d in ds:
    realname = (d['img_name'][0])
    num_labels = (len(d['labels']))
    if (d['img_name'][num_labels - 1]) == realname:
        img_n = d['img_name'][0]
        maxval = 225
        thresh = 128
        img = Image.open(img_n).convert('L')
        img_i = ImageOps.invert(img)
        im_gray = np.array(img_i)
        im_bin = (im_gray > thresh) * maxval
        savetitle = img_n.split('/')
        masktitle = (mask_config.BIMASKFOLDER + 'mask_' + savetitle[1])
        Image.fromarray(np.uint8(im_bin)).save(masktitle)


# In[6]:


# make color mask folder
os.mkdir(mask_config.COLORMASKFOLDER)
# upload color mask imgs
# filter data
for i, d in ds:
    realname = (d['img_name'][0])
    num_labels = (len(d['labels']))
    if (d['img_name'][num_labels - 1]) == realname:
        # start drawing background = img.h, img.w
        img = Image.open(d['img_name'][0])
        H, W = img.height, img.width
        img.close()
        image = Image.new("RGB", (W, H), "blue")
        svfi_name = 'colormask_' + realname.split('/')[1]
        sv_name = os.path.join(mask_config.COLORMASKFOLDER, svfi_name)
        draw = ImageDraw.Draw(image)
        
        # draw label bbox as indvidual shape colors
        for i in range(num_labels):
            img, box, label = (d['img_name'][i], d['boxes'][i], d['labels'][i])
            x,y,h,w = box
            x,y,h,w = float(x), float(y), float(h), float(w)   
            if label == mask_config.LABEL1:
                color = 'red'
                x2 = x + h
                y2 = y + w
                draw.rectangle((x, y, x2, y2), fill=color)
                
        for i in range(num_labels):
            img, box, label = (d['img_name'][i], d['boxes'][i], d['labels'][i])
            x,y,h,w = box
            x,y,h,w = float(x), float(y), float(h), float(w)
            if label == mask_config.LABEL2:
                color = 'pink'
                x2 = x + h
                y2 = y + w
                draw.rectangle((x, y, x2, y2), fill = color)
        
        for i in range(num_labels):
            img, box, label = (d['img_name'][i], d['boxes'][i], d['labels'][i])
            x,y,h,w = box
            x,y,h,w = float(x), float(y), float(h), float(w)
            if label == mask_config.LABEL3:
                color = 'green'
                x2 = x + h
                y2 = y + w
                draw.rectangle((x, y, x2, y2), fill = color)
        image.save(sv_name, 'JPEG')
            


# In[7]:


from PIL import Image, ImageDraw, ImageFilter
# make color mask folder
os.mkdir(mask_config.FINALMASKOUT)
# upload combined masks and colors
color_img_dir = os.listdir(mask_config.COLORMASKFOLDER)
bi_img_dir = os.listdir(mask_config.BIMASKFOLDER)
for i in range(len(color_img_dir)):
    im1 = Image.open(os.path.join(mask_config.COLORMASKFOLDER, color_img_dir[i]))
    H, W = im1.height, im1.width
    im2 = Image.new("RGB", (W, H), "blue")
    im2.save('im2', 'JPEG')
    im2 = Image.open('im2')
    mask = Image.open(os.path.join(mask_config.BIMASKFOLDER, bi_img_dir[i]))
    im = Image.composite(im1, im2, mask)
    fi_name = color_img_dir[i].split('_')
    fi_name = fi_name[1] + fi_name[2]
    final_sv = os.path.join(mask_config.FINALMASKOUT, fi_name)
    im.save(final_sv, 'JPEG')
    print(fi_name, ' saved to: ', mask_config.FINALMASKOUT)
    


# In[ ]:




