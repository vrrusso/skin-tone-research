import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio as im
from skimage import io, color

# this rules were suggested in Fitzpatrick17k paper


'''
#perguntar sobre rgba
def rgb_threshold(pixel):
    if pixel[0] > 95 and pixel[0] > pixel[1] and pixel[0] > pixel[2] and pixel[1]>40 and pixel[2]>20 and abs(pixel[0]-pixel[1]) > 15:
        return True
    return False

def ycrcb_threshold(pixel):
    if pixel[1] >= 35 and pixel[1] >= (0.3448*pixel[2])+76.2069 and pixel[1] >= (-45652*pixel[2])+234.5652 and pixel[1] <= (-1.15*pixel[2])+301.75 and pixel[1]<= (-2.2857*pixel[2])+432.85:
        return True
    return False

def mask_non_skin(image):
    
    image: the target image

    returns: the image with only skin pixels and the binary mask 
    

    rgb_mask = np.zeros(image.shape,dtype=np.uint8)

    
    for y in range(rgb_mask.shape[0]):
        for x in range(rgb_mask.shape[1]):
            rgb_mask[y,x] = 1 if rgb_threshold(image[y,x]) else 0

    rgb_skin = rgb_mask*image

    ycrcb_image = cv2.cvtColor(rgb_skin,cv2.COLOR_RGB2YCR_CB )

    ycrcb_mask = np.zeros(image.shape,dtype=np.uint8)

    for y in range(ycrcb_mask.shape[0]):
        for x in range(ycrcb_mask.shape[1]):
            ycrcb_mask[y,x] = 1 if ycrcb_threshold(ycrcb_image[y,x]) else 0

    return rgb_skin * ycrcb_mask, ycrcb_mask*rgb_mask 

'''

def mask_non_skin(rgb_image):
    R = rgb_image[:,:,0]
    G = rgb_image[:,:,1]
    B = rgb_image[:,:,2]

    # Reduce to skin range
    R = np.where(R < 95, 0, R)
    G = np.where(G < 40, 0, G)
    B = np.where(B < 20, 0, B)

    R = np.where(R < G, 0, R)
    R = np.where(R < B, 0, R)
    R = np.where(abs(R - G) < 15, 0, R)

    R = np.where(G == 0, 0, R)
    R = np.where(B == 0, 0, R)

    B = np.where(R == 0, 0, B)
    B = np.where(G == 0, 0, B)

    G = np.where(R == 0, 0, G)
    G = np.where(B == 0, 0, G)

    # Stack into RGB
    RGB = np.stack([R, G, B], axis = 2)

    YCBCR = color.rgb2ycbcr(RGB)
    
    Y = YCBCR[:, :, 0]
    Cb = YCBCR[:, :, 1]
    Cr = YCBCR[:, :, 2]


    # Subset to skin range
    Y = np.where(Y < 80, 0, Y)
    Cb = np.where(Cb < 85, 0, Cb)
    Cr = np.where(Cr < 135, 0, Cr)

    Cr = np.where(Cr >= (1.5862*Cb) + 20, 0, Cr)
    Cr = np.where(Cr <= (0.3448*Cb) + 76.2069, 0, Cr)
    Cr = np.where(Cr <= (-4.5652*Cb) + 234.5652, 0, Cr)
    Cr = np.where(Cr >= (-1.15*Cb) + 301.75, 0, Cr)
    Cr = np.where(Cr >= (-2.2857*Cb) + 432.85, 0, Cr)

    Y = np.where(Cb == 0, 0, Y)
    Y = np.where(Cr == 0, 0, Y)

    Cb = np.where(Y == 0, 0, Cb)
    Cb = np.where(Cr == 0, 0, Cb)

    Cr = np.where(Y == 0, 0, Cr)
    Cr = np.where(Cb == 0, 0, Cr)

    # Stack into skin region
    skinRegion = np.stack([Y, Cb, Cr], axis = 2)
    skinRegion = np.where(skinRegion != 0, 255, 0)
    skinRegion = skinRegion.astype(dtype = "uint8")

    # Apply mask to original RGB image
    mask = np.array(RGB)
    mask = np.where(skinRegion != 0, mask, 0)

    #print(mask.shape)

    return mask,None



