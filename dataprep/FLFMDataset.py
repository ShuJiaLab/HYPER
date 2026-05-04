# loading, preprocessing, and providing data for training, validation, and testing of the FLFMnet model.

import torch
import torchvision as tv
from torch.utils import data
import torch.nn.functional as F
import glob
import os
import numpy as np
from tifffile import imread
from rich.progress import track

import utils.pytorch_shot_noise as pytorch_shot_noise
from utils.misc_utils import *
from dataprep.imgRegister import *

class FLFMDataset(data.Dataset):
    def __init__(self, data_path, datavol_path,lenslet_coords_path, subimage_shape, img_shape, data_scale=[1,1],
                 images_to_use=None, lenslets_offset=0, n_depths_to_fill=64,load_vols=False,maxWorkers=10):
        # Load lenslets coordinates
        print('#====================================================#')
        self.lenslet_coords = get_lenslet_centers(lenslet_coords_path) + torch.tensor(lenslets_offset)
        self.n_lenslets = self.lenslet_coords.shape[0]
        self.images_to_use = images_to_use
        self.data_path = data_path
        self.load_vols = load_vols
        self.vol_type = torch.float16
        self.data_scale = data_scale
        self.img_shape = img_shape
        self.subimage_shape = subimage_shape
        self.half_subimg_shape = [self.subimage_shape[0]//2,self.subimage_shape[1]//2]
        self.n_images = len(images_to_use)
        n_images_to_load = max(images_to_use)-min(images_to_use) + 1
        print('# Plan to use %d images' % n_images_to_load)

        # =========================== Set up data lists ===========================
        imgs_path = data_path + '/*.tif'
        vols_path = datavol_path + '/*.tif'
        imgs_list = sorted(glob.glob(imgs_path))[images_to_use[0]:(images_to_use[-1]+1)]
        self.all_files = sorted(glob.glob(vols_path))[images_to_use[0]:(images_to_use[-1]+1)]
        print('# Loading image from: ', imgs_path, ' (', len(imgs_list), ' )')
        print('# Loading volumes from: ', vols_path, ' (', len(self.all_files), ' stacks)')
        
        # ============================= Load 2D images =============================
        self.img_dataset = imread(imgs_list, maxworkers=maxWorkers)
        print('>< Loaded %d images of size %d x %d' % self.img_dataset.shape)
                
        # ============================= Load 3D image stacks =======================
        self.vols = []
        for fname in track(self.all_files):
            stack = self.read_tiff_stack(fname, out_datatype=torch.float16)
            stack = stack.permute(2,0,1)
            self.vols.append(stack)
        self.vols = torch.stack(self.vols)
        print('>< Loaded %d image stacks of size %d x %d x %d' % self.vols.shape)

        # ========================== Load the first volume ==========================
        # if load_vols: # Load the first volume in the stack to get the volume size
        #     currVol = self.read_tiff_stack(self.all_files[0])  # load first volume in H x W x D
        #     if currVol.shape[0]==1025 or currVol.shape[1]==1025: currVol = currVol[1:,1:]
        #     currVol = currVol.permute(2,0,1)
        #     currVol = tv.transforms.functional.resize(currVol, self.subimage_shape, antialias=True)
        #     self.volStart = currVol.shape[0]//2-n_depths_to_fill//2
        #     self.volEnd = self.volStart + n_depths_to_fill
        #     print('\t> Volume start: ', self.volStart, ' end: ', self.volEnd)
        #     self.vols = torch.zeros(n_images_to_load, n_depths_to_fill, currVol.shape[-2], currVol.shape[-1], dtype=self.vol_type)
        # else:
        #     self.vols = 255*torch.ones(1)
        
        # # ========================== Create image storage ==========================
        # self.stacked_views = torch.zeros(n_images_to_load, self.img_dataset.shape[-3], self.img_dataset.shape[-2], self.img_dataset.shape[-1],dtype=torch.float16)
        # # print('# Allocated [Rescaled] stacked FLFM image size: ', self.stacked_views.shape,'and vol size: ', self.vols.shape)
        # print('# Allocated [Rescaled] stacked FLFM image size: ', self.stacked_views.shape)
        
        # for nImg in track(range(n_images_to_load), description="> Loading Volumes..."):
        #     curr_img = nImg
        #     # if load_vols:
        #     #     currVol = self.read_tiff_stack(self.all_files[curr_img],range(self.volStart,self.volEnd,1))
        #     #     assert not torch.isinf(currVol).any()
        #     #     if currVol.shape[0]==1025 or currVol.shape[1]==1025: currVol = currVol[1:,1:]
        #     #     currVol = currVol.permute(2,0,1)
        #     #     currVol = tv.transforms.functional.resize(currVol, self.subimage_shape, antialias=True)
        #     #     self.vols[nImg,:currVol.shape[0],:,:] = currVol
        #     image = torch.from_numpy(np.array(self.img_dataset[curr_img,:,:]).astype(np.float16)).type(torch.float16)
        #     image = self.pad_img_to_min(image)
        #     self.stacked_views[nImg,...] = center_crop(image.unsqueeze(0).unsqueeze(0),
        #                                                (self.img_dataset.shape[-2],self.img_dataset.shape[-1]))[0,0,...]
        # self.stacked_views = tv.transforms.functional.resize(self.stacked_views, img_shape,antialias=True)
        # # print('# Allocated [Rescaled] stacked FLFM volume size: ', self.vols.shape)
        # # print('# Loaded ' + str(self.n_images),'FLFM images: ', self.stacked_views.shape,'GT Volumes: ', self.vols.shape)
        # print('#======================================================#')

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_images
    
    def get_max(self):
        'Get max intensity from volumes and images for normalization'
        # return  self.stacked_views.float().max().type(self.stacked_views.type()),\
        #         self.stacked_views.float().max().type(self.stacked_views.type()),\
        return  self.vols.float().max().type(self.vols.type())

    def get_statistics(self):
        'Get mean and standard deviation from volumes and images for normalization'
        # return  self.stacked_views.float().mean().type(self.stacked_views.type()),\
        #         self.stacked_views.float().std().type(self.stacked_views.type()), \
        return  self.vols.float().mean().type(self.vols.type()), \
                self.vols.float().std().type(self.vols.type())

    def standarize(self, stats=None):
        mean_imgs, std_imgs, mean_imgs_s, std_imgs_s, mean_vols, std_vols = stats
        self.stacked_views[...] = (self.stacked_views[...]-mean_imgs) / std_imgs
        # self.vols = (self.vols-mean_vols) / std_vols

    def pad_img_to_min(self,image):
        min_size = min(image.shape[-2:])
        # print('>>> min size to pad: ', min_size)
        img_pad = [min_size-image.shape[-1], min_size-image.shape[-2]]
        img_pad = [img_pad[0]//2, img_pad[0]//2, img_pad[1],img_pad[1]]
        # print('>>> padding image size: ', img_pad)
        image = F.pad(image.unsqueeze(0).unsqueeze(0), img_pad)[0,0]
        return image

    def __getitem__(self, index):
        # new_index = index
        # indices = [new_index]
        # views_out = self.stacked_views[indices,...]

        # if self.load_vols is False:
        #     return views_out,0
        # # vol_out = self.vols[index,...]
        # # return views_out,vol_out
        # return views_out
        curr_img_stack = self.vols[index]  # 3D volume (64x256x256)
        curr_img = self.img_dataset[index] # 2D FLFM img (768x768)
        return curr_img_stack, curr_img

    
    @staticmethod
    def extract_views(image,vol, lenslet_coords, subimage_shape,data_scale=[1,1],isreg=False,showreginfo=False):
        subimage_shape_0 = [round(image.shape[-2]/3*data_scale[0]),round(image.shape[-1]/3*data_scale[1])] # rescaled by data_scale
        half_subimg_shape = [subimage_shape_0[0]//2,subimage_shape_0[1]//2]
        n_lenslets = lenslet_coords.shape[0]
        extracted_views = torch.zeros(size=[image.shape[0], image.shape[1], 
                                          n_lenslets, subimage_shape[0], subimage_shape[1]], 
                                          device=image.device, dtype=image.dtype)
        for nLens in range(n_lenslets):
            currCoords = lenslet_coords[nLens,:]
            # print('>>> currCoords: ', currCoords)
            lower_bounds = [currCoords[0]-half_subimg_shape[0], currCoords[1]-half_subimg_shape[1]]
            lower_bounds = [max(lower_bounds[kk],torch.tensor(0,dtype=torch.int32)) for kk in range(2)]
            # print('>>> lower_bounds: ', lower_bounds)
            subimg = image[:,:,lower_bounds[0] : currCoords[0]+half_subimg_shape[0], lower_bounds[1] : currCoords[1]+half_subimg_shape[1]]
            # print('>>> image shape: ', image.shape)
            # print('>>> subimg shape: ', subimg.shape)
            # subimg = F.interpolate(subimg.float(), scale_factor=145/65, mode='bilinear', align_corners=False).half()
            currPatch = F.pad(subimg, [subimage_shape[1]//2-subimg.shape[-1]//2, 
                                       subimage_shape[1]-subimg.shape[-1]-subimage_shape[1]//2+subimg.shape[-1]//2,
                                       subimage_shape[0]//2-subimg.shape[-2]//2,
                                       subimage_shape[0]-subimg.shape[-2]-subimage_shape[0]//2+subimg.shape[-2]//2])
            currPatch = subimg
            extracted_views[:,:,nLens,-currPatch.shape[2]:,-currPatch.shape[3]:] = currPatch
        
        if isreg is True:
            for ii in range(extracted_views.shape[0]):
                for jj in range(extracted_views.shape[1]):
                    datadevice = extracted_views.device
                    moving = extracted_views[ii,jj,2,...]
                    fixed = vol[jj,...].max(dim=0).values
                    _,outTx = imRegister(fixed, moving,showreginfo)
                    extracted_views[ii,jj,2,...] = torch.tensor(imRegisterShift(fixed, 
                                                extracted_views[ii,jj,2,...], outTx)).half().to(datadevice)
                    extracted_views[ii,jj,1,...] = torch.tensor(imRegisterShift(fixed, 
                                                extracted_views[ii,jj,1,...], outTx)).half().to(datadevice)
                    extracted_views[ii,jj,0,...] = torch.tensor(imRegisterShift(fixed, 
                                                extracted_views[ii,jj,0,...], outTx)).half().to(datadevice)
        
        return extracted_views
        
    @staticmethod
    def read_tiff_stack(filename, keyrange=None, out_datatype=torch.float16):
        out = np.clip(imread(filename, key=keyrange), 0, torch.finfo(out_datatype).max)
        return torch.from_numpy(out).permute(1,2,0).type(out_datatype)

    def add_random_shot_noise_to_dataset(self, signal_power_range=[32**2,32**2]):
        for nImg in range(self.stacked_views.shape[0]):
            signal_power = (signal_power_range[0] + (signal_power_range[1]-signal_power_range[0]) * torch.rand(1)).item()
            curr_img_stack = self.stacked_views[nImg,...].float()
            curr_max = curr_img_stack.max()
            curr_img_stack = signal_power * curr_img_stack / curr_max    
            for kk in range(curr_img_stack.shape[0]):
                curr_img_stack[kk,...] = pytorch_shot_noise.add_camera_noise(curr_img_stack[kk,...])
            curr_img_stack = curr_max * curr_img_stack.float() / signal_power
            self.stacked_views[nImg,...] = curr_img_stack
        print("Added noise to " + str(self.stacked_views.shape[0]) + " images.")

