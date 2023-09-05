#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:30:18 2023

@author: Mathew
"""


import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import filters,measure
from skimage.measure import regionprops, label
from PIL import Image
import pandas as pd
import os
from scipy import ndimage
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from skan.csr import skeleton_to_csgraph
from skimage import img_as_bool
import cv2
from skan import draw
from matplotlib.backends.backend_pdf import PdfPages
import collections
# Path to images
path="/Users/Mathew/Documents/Current analysis/Branch analysis/"
path_intensity="/Users/Mathew/Documents/Current analysis/Branch analysis/Raw/"
path_binary="/Users/Mathew/Documents/Current analysis/Branch analysis/Predicted/"




filename_contains=".tif"



# Function to load images:

def load_image(toload):
    
    image=imread(toload)
    
    
    return image

# Threshold image using otsu method and output the filtered image along with the threshold value applied:
    
def threshold_image_otsu(input_image):
    threshold_value=filters.threshold_otsu(input_image)    
    binary_image=input_image>threshold_value

    return threshold_value,binary_image


# Threshold image using otsu method and output the filtered image along with the threshold value applied:
    
def threshold_image_fixed(input_image,threshold_number):
    threshold_value=threshold_number   
    binary_image=input_image>threshold_value

    return threshold_value,binary_image

# Label and count the features in the thresholded image:
def label_image(input_image):
    labelled_image=measure.label(input_image)
    number_of_features=labelled_image.max()
 
    return number_of_features,labelled_image
    
# Function to show the particular image:
def show(input_image,color=''):
    if(color=='Red'):
        plt.imshow(input_image,cmap="Reds")
        plt.show()
    elif(color=='Blue'):
        plt.imshow(input_image,cmap="Blues")
        plt.show()
    elif(color=='Green'):
        plt.imshow(input_image,cmap="Greens")
        plt.show()
    else:
        plt.imshow(input_image)
        plt.show() 
    
        
# Take a labelled image and the original image and measure intensities, sizes etc.
def analyse_labelled_image(labelled_image,original_image):
    measure_image=measure.regionprops_table(labelled_image,intensity_image=original_image,properties=('area','perimeter','centroid','orientation','major_axis_length','minor_axis_length','mean_intensity','max_intensity'))
    measure_dataframe=pd.DataFrame.from_dict(measure_image)
    return measure_dataframe

# This is to look at coincidence purely in terms of pixels




directory=path_intensity

# print(path)

# Store data here:
    
labelled_path = os.path.join(path,'Files_Labelled')
if not os.path.isdir(labelled_path):
    os.mkdir(labelled_path)

skel_path = os.path.join(path,'Files_Skeletonised')
if not os.path.isdir(skel_path):
    os.mkdir(skel_path)



results_output=os.path.join(path,'Files_Results')
if not os.path.isdir(results_output):
    os.mkdir(results_output)

# mito_overlap=os.path.join(path,'Files_Mito_Overlap')
# mito_non_overlap=os.path.join(path,'Files_Mito_Non_Overlap')

# if not os.path.isdir(mito_path):
#     os.mkdir(mito_path)

# if not os.path.isdir(calc_path):
#     os.mkdir(calc_path)

# if not os.path.isdir(binary_calcium):
#     os.mkdir(binary_calcium)

# if not os.path.isdir(mito_overlap):
#     os.mkdir(mito_overlap)

# if not os.path.isdir(mito_non_overlap):
#     os.mkdir(mito_non_overlap)
    
    
#
# This just looks for files in the directory

for root, dirs, files in os.walk(path_intensity):
  for name in files:
    if filename_contains in name:
     if 'Files_' not in root:
     
                
                src=root+'/'+name
                print(src)
               
                
                image = load_image(path_intensity+name)
                binary = load_image(path_binary+name)
                
                
                # Label the image 
    
                number_detected,labelled=label_image(binary)
                
                labelled_im=labelled.astype('uint8')
                

                # Save image
                im = Image.fromarray(labelled_im)
                im.save(labelled_path+'/'+name)
                
                
                # Measure intensity etc. 
                                
                labelled_img_props=analyse_labelled_image(labelled,image)
                
                labelled_img_props.to_csv(results_output+'/'+name+'/'+'lengths_brightness.csv', index=False)
                               
                
                # Skeletonise
                
          
                binary_image = img_as_bool(cv2.imread(path_binary+name, cv2.IMREAD_GRAYSCALE))
             
                
                skeleton = skeletonize(binary_image)
            
                # Create a Skeleton object from the skeletonized image
                skeleton_obj = Skeleton(skeleton)
            
                # Summarize the results
                skeleton_obj = skeleton_obj
                branch_data = summarize(skeleton_obj)
                

                
                curve_ind = []
                for bd, ed in zip(branch_data["branch-distance"], branch_data["euclidean-distance"]):

                    if ed != 0.0:
                        curve_ind.append((bd - ed) / ed)
                    else:
                        curve_ind.append(bd - ed)

                branch_data["curvature-index"] = curve_ind

                grouped_branch_data_mean = branch_data.groupby(["skeleton-id"], as_index=False).mean()

                grouped_branch_data_sum = branch_data.groupby(["skeleton-id"], as_index=False).sum()

                counter = collections.Counter(branch_data["skeleton-id"])

                n_branches = []
                for i in grouped_branch_data_mean["skeleton-id"]:
                    n_branches.append(counter[i])

                branch_len = grouped_branch_data_mean["branch-distance"].tolist()
                tot_branch_len = grouped_branch_data_sum["branch-distance"].tolist()

                curv_ind = grouped_branch_data_mean["curvature-index"].tolist()
                
                
                branch_data.to_csv(results_output+'/'+name+'/'+'Branch_data.csv', index=False)
            
                # Some histograms
                
                # plt.hist(n_branches, bins = 5,range=[0,10], color='skyblue', edgecolor='black')
                # plt.xlabel('Number of branches')
                # plt.ylabel('Number of mitochondria')
                # # plt.savefig(path+'/'+'FRET_Small.pdf')
                # plt.show()
                
                # plt.hist(branch_len, bins = 20,range=[0,50], color='skyblue', edgecolor='black')
                # plt.xlabel('Branch length')
                # plt.ylabel('Number of mitochondria')
                # # plt.savefig(path+'/'+'FRET_Small.pdf')
                # plt.show()
     
                # plt.hist(tot_branch_len, bins = 50,range=[0,100], color='skyblue', edgecolor='black')
                # plt.xlabel('Total branch length')
                # plt.ylabel('Number of mitochondria')
                # # plt.savefig(path+'/'+'FRET_Small.pdf')
                # plt.show()
        
                # plt.hist(curv_ind, bins = 20,range=[0,1], color='skyblue', edgecolor='black')
                # plt.xlabel('Curvature index')
                # plt.ylabel('Number of mitochondria')
                # # plt.savefig(path+'/'+'FRET_Small.pdf')
                # plt.show()
        
    