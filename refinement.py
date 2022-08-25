import cv2
import time
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import segmentation_refinement as refine
import numpy as np
#image = cv2.imread('test/aeroplane.jpg')
#mask = cv2.imread('test/aeroplane.png', cv2.IMREAD_GRAYSCALE)

# model_path can also be specified here
# This step takes some time to load the model
refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'

def add_mask_to_background(img, mask, color=(255,255,255)):
    d_mask = mask.astype(np.float32)
    bitwise_mask = cv2.cvtColor(d_mask, cv2.COLOR_GRAY2RGB)
    output = img | (bitwise_mask * color).astype(np.uint8)
    return output.astype(np.uint8)
def refine_images(images_path,sg_path,masks_path):
    f = h5py.File(masks_path, "r+")
    
    for image_id in sg:
        image = f[str(image_id)]['image'][...]
        for key_mask in f[str(image_id)]['objects']:
            background = f[str(image_id)]['image'][...]*(0,0,0)
            mask = f[str(image_id)]['objects'][key_mask]['mask'][...]
            mask = add_mask_to_background(background,mask)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            mask = cv2.cvtColor(np.array(mask), cv2.COLOR_BGR2GRAY)
            # Fast - Global step only.
            # Smaller L -> Less memory usage; faster in fast mode.
            output = refiner.refine(image, mask, fast=False, L=900) 

            # this line to save output
            cv2.imwrite('mask'+str(image_id)+'-'+str(key_mask)+'.png', output)
