import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio as im
from skimage import io, color
import math

from rule_based_segmentation import mask_non_skin

def ITA(image,std_elimination=True):
    """
    Calculates the individual typology angle (ITA) for a given 
    RGB image.

    Inputs:
        image - (str) RGB image file path

    Outputs:
        ITA - (float) individual typology angle
    """

    # Convert to CIE-LAB color space
    RGB = image[:,:,:3]
    #print(RGB.shape)
    CIELAB = np.array(color.rgb2lab(RGB))
    #print("foi")

    # Get L and B (subset to +- 1 std from mean)
    L = CIELAB[:, :, 0]
    L = np.where(L != 0, L, np.nan)

    

    B = CIELAB[:, :, 2]
    B = np.where(B != 0, B, np.nan)


    if std_elimination == True:
        std, mean = np.nanstd(L), np.nanmean(L)
        L = np.where(L >= mean - std, L, np.nan)
        L = np.where(L <= mean + std, L, np.nan)


        std, mean = np.nanstd(B), np.nanmean(B)
        B = np.where(B >= mean - std, B, np.nan)
        B = np.where(B <= mean + std, B, np.nan)

    #print(np.nanmean(L))
    #print(np.nanmean(B))

    # Calculate ITA
    ITA = math.atan2(np.nanmean(L) - 50, np.nanmean(B)) * (180 / np.pi)

    return ITA

def empirical_classification(ita):
    if ita > 40:
        return 1
    if ita> 23 and ita <=40:
        return 2
    if ita >12 and ita <=23:
        return 3
    if ita >0 and ita <=12:
        return 4
    if ita >-25 and ita<=0:
        return 5
    return 6

def kinyananjui_classification(ita):
    if ita > 55:
        return 1
    if ita> 41 and ita <=55:
        return 2
    if ita >28 and ita <=41:
        return 3
    if ita >19 and ita <=28:
        return 4
    if ita >10 and ita<=19:
        return 5
    return 6

#img = im.imread('white_guy.jpg')
#skin,__ = mask_non_skin(img)
#print(ITA(skin))