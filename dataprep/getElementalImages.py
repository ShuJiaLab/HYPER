import torch
from imageio.v3 import imread, imwrite
from  skimage.transform import resize as imresize
import numpy as np
import sys
sys.path.append('Z:\\Xuanwen\\DLFLFM\\dl_pytorch_100x_v2\\')
from FLFMDataset import FLFMDataset
from utilities.util_getMLAcenters import get_lenslet_centers

def getElementalImages(image,psfcoordsfile,subimage_shape,data_scale=1.0):
    lenslet_coords = get_lenslet_centers(psfcoordsfile)
    ext_views = FLFMDataset.extract_views(image, lenslet_coords, subimage_shape,data_scale)
    return ext_views


if __name__=="__main__":
    # ################### set up parameters ###################
    data_scale = 0.5
    subimage_shape = (round(1024*data_scale) , round(1024*data_scale))

    # ################### load image and get elemental images ###################
    image = imread("Z:\\Keyi\\00_Project\\02_SMLM\\data\\KH230912_Mitochondria2_denoise\\individual_example\\cell6_storm_000500.tif")
    image = imresize(image.astype(np.float32),subimage_shape)
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    psfcoordsfile = "Z:\\Xuanwen\\DLFLFM\\dldatasets\\train\\TrainingExpData\\PSFFLFint_20240328_Red_Gly_10um\\lenslet_coords_red.txt"
    
    # ################### get elemental images ###################
    ext_views = getElementalImages(image,psfcoordsfile,subimage_shape,data_scale=data_scale)
    ext_views = ext_views.squeeze(0).squeeze(0).numpy()
    # ext_views = np.transpose(ext_views,(1,2,0))
    print(ext_views.shape)

    # ################### save elemental images ###################
    orients = ["Lu","Ru","Cd"] # "Lu" = left-up, "Ru" = right-up, "Cd" = center-down
    for ii in range(0,ext_views.shape[0]):
        imwrite("Z:\\Keyi\\00_Project\\02_SMLM\\data\\KH230912_Mitochondria2_denoise\\individual_example\\cell6_storm_000500_("+\
                str(ii+1)+")_"+orients[ii]+".tif",ext_views[ii].astype(np.uint16))