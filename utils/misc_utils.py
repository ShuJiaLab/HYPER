import torch
import torchvision as tv
import torch.nn.functional as F
# from waveblocks.utils import complex_operations as ob
from PIL import Image
import torchvision.transforms as TF
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
# from scipy.ndimage.filters import gaussian_filter
# import h5py
# import gc
import re
import numpy as np
import findpeaks
import csv
# import pickle
import tifffile
from scipy.io import loadmat as loadmat7
from mat73 import loadmat as loadmat73

import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

# from mainTrainFLFMnet import OTF

# Prepare a volume to be shown in tensorboard as an image
def volume_2_tensorboard(vol, batch_index=0, z_scaling=2):
    vol = vol.detach()
    # expecting dims to be [batch, depth, xDim, yDim]
    xyProj = tv.utils.make_grid(vol[batch_index,...].float().unsqueeze(0).sum(1).cpu().data, normalize=True, scale_each=True)
    
    # interpolate z in case that there are not many depths
    vol = torch.nn.functional.interpolate(vol.permute(0,2,3,1).unsqueeze(1), (vol.shape[2], vol.shape[3], vol.shape[1]*z_scaling))
    yzProj = tv.utils.make_grid(vol[batch_index,...].float().unsqueeze(0).sum(3).cpu().data, normalize=True, scale_each=True)
    xzProj = tv.utils.make_grid(vol[batch_index,...].float().unsqueeze(0).sum(2).cpu().data, normalize=True, scale_each=True)

    return xzProj, yzProj, xyProj


def volume_2_projections(vol_in, proj_type=torch.max, scaling_factors=[1,1,1], depths_in_ch=False, ths=[0.0,1.0], normalize=True, border_thickness=10, add_scale_bars=False, scale_bar_vox_sizes=[40,20]):
    vol = vol_in.detach().clone()
    # Normalize sets limits from 0 to 1
    if normalize:
        vol -= vol.float().min()
        vol /= vol.float().max()
    if depths_in_ch:
        vol = vol.permute(0,2,3,1).unsqueeze(1)
    if ths[0]!=0.0 or ths[1]!=1.0:
        vol_min,vol_max = vol.min(),vol.max()
        vol[(vol-vol_min)<(vol_max-vol_min)*ths[0]] = 0
        vol[(vol-vol_min)>(vol_max-vol_min)*ths[1]] = vol_min + (vol_max-vol_min)*ths[1]

    vol_size = list(vol.shape)
    vol_size[2:] = [vol.shape[i+2] * scaling_factors[i] for i in range(len(scaling_factors))]

    if proj_type is torch.max or proj_type is torch.min:
        x_projection, _ = proj_type(vol.float().cpu(), dim=2)
        y_projection, _ = proj_type(vol.float().cpu(), dim=3)
        z_projection, _ = proj_type(vol.float().cpu(), dim=4)
    elif proj_type is torch.sum:
        x_projection = proj_type(vol.float().cpu(), dim=2)
        y_projection = proj_type(vol.float().cpu(), dim=3)
        z_projection = proj_type(vol.float().cpu(), dim=4)

    out_img = z_projection.min() * torch.ones(
        vol_size[0], vol_size[1], vol_size[2] + vol_size[4] + border_thickness, vol_size[3] + vol_size[4] + border_thickness
    )

    # def normalize_proj(proj):
    #     return (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)

    # z_projection = normalize_proj(z_projection)
    # x_projection = normalize_proj(x_projection)
    # y_projection = normalize_proj(y_projection)

    out_img[:, :, : vol_size[2], : vol_size[3]] = z_projection
    out_img[:, :, vol_size[2] + border_thickness :, : vol_size[3]] = F.interpolate(x_projection.permute(0, 1, 3, 2), size=[vol_size[-1],vol_size[-3]])
    out_img[:, :, : vol_size[2], vol_size[3] + border_thickness :] = F.interpolate(y_projection, size=[vol_size[2],vol_size[4]])

    line_color = out_img.max()
    # Draw white lines
    out_img[:, :, vol_size[2]: vol_size[2]+ border_thickness, ...] = line_color
    out_img[:, :, :, vol_size[3]:vol_size[3]+border_thickness, ...] = line_color

    if add_scale_bars:
        start = 0.02
        out_img[:, :, int(start* vol_size[2]):int(start* vol_size[2])+4, int(0.9* vol_size[3]):int(0.9* vol_size[3])+scale_bar_vox_sizes[0]] = line_color
        out_img[:, :, int(start* vol_size[2]):int(start* vol_size[2])+4, vol_size[2] + border_thickness + 10 : vol_size[2] + border_thickness + 10 + scale_bar_vox_sizes[1]*scaling_factors[2]] = line_color
        out_img[:, :, vol_size[2] + border_thickness + 10 : vol_size[2] + border_thickness + 10 + scale_bar_vox_sizes[1]*scaling_factors[2], int(start* vol_size[2]):int(start* vol_size[2])+4] = line_color

    return out_img

def imshow2D(img, blocking=False):
    plt.figure(figsize=(10,10))
    plt.imshow(img[0,0,...].float().detach().cpu().numpy())
    if blocking:
        plt.show()
def imshow3D(vol, blocking=False):
    plt.figure(figsize=(10,10))
    plt.imshow(volume_2_projections(vol.permute(0,2,3,1).unsqueeze(1), normalize=True)[0,0,...].float().detach().cpu().numpy())
    if blocking:
        plt.show()
def imshowComplex(vol, blocking=False):
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(volume_2_projections(torch.real(vol).permute(0,2,3,1).unsqueeze(1))[0,0,...].float().detach().cpu().numpy())
    plt.subplot(1,2,2)
    plt.imshow(volume_2_projections(torch.imag(vol).permute(0,2,3,1).unsqueeze(1))[0,0,...].float().detach().cpu().numpy())
    if blocking:
        plt.show()

def save_image(tensor, path='output.png'):
    if 'tif' in path:
        tifffile.imwrite(path, tensor[0,...].cpu().numpy().astype(np.float16))
        return
    if tensor.shape[1] == 1:
        imshow2D(tensor)
    else:
        imshow3D(tensor)
    plt.savefig(path)


# Aid functions for shiftfft2
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)
def batch_fftshift2d_real(x):
    out = x
    for dim in range(2, len(out.size())):
        n_shift = x.size(dim)//2
        if x.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        out = roll_n(out, axis=dim, n=n_shift)
    return out  

# FFT convolution, the kernel fft can be precomputed
def fft_conv(A,B, fullSize, Bshape=[],B_precomputed=False):
    # print("B_precomputed:", B_precomputed)
    # print("A shape: ", A.shape)  
    import torch.fft
    nDims = A.ndim-2
    # fullSize = torch.tensor(A.shape[2:]) + Bshape
    # fullSize = torch.pow(2, torch.ceil(torch.log(fullSize.float())/torch.log(torch.tensor(2.0)))-1)
    padSizeA = (fullSize - torch.tensor(A.shape[2:]))
    padSizesA = torch.zeros(2*nDims,dtype=int)
    padSizesA[0::2] = torch.floor(padSizeA/2.0)
    padSizesA[1::2] = torch.ceil(padSizeA/2.0)
    padSizesA = list(padSizesA.numpy()[::-1])

    A_padded = F.pad(A,padSizesA)
    # print("A_padded shape: ", A_padded.shape)  
     
    Afft = torch.fft.rfft2(A_padded)
    # print("Afft shape: ", Afft.shape)
    
    if B_precomputed:
        return batch_fftshift2d_real(torch.fft.irfft2( Afft * B.detach()))
    else:
        padSizeB = (fullSize - torch.tensor(B.shape[2:]))
        padSizesB = torch.zeros(2*nDims,dtype=int)
        padSizesB[0::2] = torch.floor(padSizeB/2.0)
        padSizesB[1::2] = torch.ceil(padSizeB/2.0)
        padSizesB = list(padSizesB.numpy()[::-1])
        B_padded = F.pad(B,padSizesB)
        Bfft = torch.fft.rfft2(B_padded)
        return batch_fftshift2d_real(torch.fft.irfft2( Afft * Bfft.detach())), Bfft.detach()


def reprojection_loss_camera(gt_imgs, prediction, PSF, camera, dataset, device="cpu"):
    out_type = gt_imgs.type()
    camera = camera.to(device)
    reprojection = camera(prediction.to(device), PSF.to(device))
    reprojection_views = dataset.extract_views(reprojection, prediction[0,...].unsqueeze(0),
                                               dataset.lenslet_coords, dataset.subimage_shape,isreg=False)[0,0,...]
    loss = F.mse_loss(gt_imgs.float().to(device), reprojection_views.float().to(device))

    return loss.type(out_type), reprojection_views.type(out_type), gt_imgs.type(out_type), reprojection.type(out_type)

def reprojection_loss(gt_imgs, prediction, OTF, psf_shape, dataset, n_split=20, device="cpu", loss=F.mse_loss):
    out_type = gt_imgs.type()
    batch_size = prediction.shape[0]
    reprojection = fft_conv_split(prediction[0,...].unsqueeze(0), OTF, psf_shape, n_split, B_precomputed=True, device=device)

    reprojection_views = torch.zeros_like(gt_imgs)
    reprojection_views[0,...] = dataset.extract_views(reprojection, prediction[0,...].unsqueeze(0),
                                                      dataset.lenslet_coords, dataset.subimage_shape,dataset.data_scale,isreg=False)[0,0,...]

    # full_reprojection = reprojection.detach()
    # reprojection_views = reprojection_views.unsqueeze(0).repeat(batch_size,1,1,1)
    for nSample in range(1,batch_size):
        reprojection = fft_conv_split(prediction[nSample,...].unsqueeze(0), OTF, psf_shape, n_split, B_precomputed=True, device=device)
        reprojection_views[nSample,...] = dataset.extract_views(reprojection, prediction[nSample,...].unsqueeze(0),
                                                                dataset.lenslet_coords, dataset.subimage_shape,dataset.data_scale,isreg=False)[0,0,...]
        # full_reprojection += reprojection.detach()

    # gt_imgs /= gt_imgs.float().max()
    # reprojection_views /= reprojection_views.float().max()
    # loss = F.mse_loss(gt_imgs[gt_imgs!=0].to(device), reprojection_views[gt_imgs!=0])
    # loss = (1-gt_imgs[reprojection_views!=0]/reprojection_views[reprojection_views!=0]).abs().mean()
    loss = loss(gt_imgs.float().to(device), reprojection_views.float().to(device))

    return loss.type(out_type), reprojection_views.type(out_type), gt_imgs.type(out_type), reprojection.type(out_type)

# Split an fft convolution into batches containing different depths
def fft_conv_split(A, B, psf_shape, n_split, B_precomputed=False, device = "cpu"):
    n_depths = A.shape[1]
    
    split_conv = n_depths//n_split
    depths = list(range(n_depths))
    depths = [depths[i:i + split_conv] for i in range(0, n_depths, split_conv)]

    fullSize = torch.tensor(A.shape[2:]) + psf_shape
    # print('fullSize: ', fullSize)    
    
    crop_pad = [(psf_shape[i] - fullSize[i])//2 for i in range(0,2)]
    # print('\t>>> crop_pad #1: ', crop_pad)
    crop_pad = (crop_pad[1], (psf_shape[-1]- fullSize[-1])-crop_pad[1], crop_pad[0], (psf_shape[-2] - fullSize[-2])-crop_pad[0])
    # print('\t>>> crop_pad #2: ', crop_pad)
    # Crop convolved image to match size of PSF
    img_new = torch.zeros(A.shape[0], 1, psf_shape[0], psf_shape[1], device=device)
    # print('\t>>> img_new.shape: ', img_new.shape)
    if B_precomputed == False:
        OTF_out = torch.zeros(1, n_depths, fullSize[0], fullSize[1]//2+1, requires_grad=False, dtype=torch.complex64, device=device)
        # print('\t>>> OTF_out.shape: ', OTF_out.shape)
    for n in range(n_split):
        # print(n)
        curr_psf = B[:,depths[n],...].to(device)
        # print('\t>>> curr_psf.shape: ', curr_psf.shape)
        img_curr = fft_conv(A[:,depths[n],...].to(device), curr_psf, fullSize, psf_shape, B_precomputed)
        # print('\t>>> img_curr.shape: ', [a.shape for a in img_curr])
        if B_precomputed == False:
            OTF_out[:,depths[n],...] = img_curr[1]
            img_curr = img_curr[0]
        img_curr = F.pad(img_curr, crop_pad)
        # print('\t>>> img_curr.shape #2: ', [a.shape for a in img_curr])
        img_new += img_curr[:,:,:psf_shape[0],:psf_shape[1]].sum(1).unsqueeze(1).abs()
    
    if B_precomputed == False:
        return img_new, OTF_out
    return img_new


def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    mask = (y>0).float()
    y = torch.mul(y,mask)
    return y

# # Apply different normalizations to volumes and images
# def normalize_type(LF_views, vols, id=0, mean_imgs=0, std_imgs=1, mean_vols=0, std_vols=1, max_imgs=1, max_vols=1, inverse=False):
#     if inverse:
#         if id==-1: # No normalization
#             return LF_views, vols
#         if id==0: # baseline normlization
#             return (LF_views) * (2*std_imgs), vols * std_vols + mean_vols
#         if id==1: # Standarization of images and volume normalization
#             return LF_views * std_imgs + mean_imgs, vols * std_vols
#         if id==2: # normalization of both
#             return LF_views * max_imgs, vols * max_vols
#         if id==3: # normalization of both
#             return LF_views * std_imgs, vols * std_vols
#     else:
#         if id==-1: # No normalization
#             return LF_views, vols
#         if id==0: # baseline normlization
#             return (LF_views) / (2*std_imgs), (vols - mean_vols) / std_vols
#         if id==1: # Standarization of images and volume normalization
#             return (LF_views - mean_imgs) / std_imgs, vols / std_vols
#         if id==2: # normalization of both
#             return LF_views / max_imgs, vols / max_vols
#         if id==3: # normalization of both
#             return LF_views / std_imgs, vols / std_vols

# Apply different normalizations to volumes and images (self_supervised)
def  normalize_type(LF_views, id=0, mean_imgs=0, std_imgs=1, max_imgs=1, inverse=False):
    if inverse:
        if id==-1: # No normalization
            return LF_views
        if id==0: # baseline normlization
            return (LF_views) * (2*std_imgs)
        if id==1: # Standarization of images and volume normalization
            return LF_views * std_imgs + mean_imgs
        if id==2: # normalization of both
            return LF_views * max_imgs
        if id==3: # normalization of both
            return LF_views * std_imgs
    else:
        if id==-1: # No normalization
            return LF_views
        if id==0: # baseline normlization
            return (LF_views) / (2*std_imgs)
        if id==1: # Standarization of images and volume normalization
            return (LF_views - mean_imgs) / std_imgs
        if id==2: # normalization of both
            return LF_views / max_imgs
        if id==3: # normalization of both
            return LF_views / std_imgs

# Random transformation of volume, for augmentation
def transform_volume(currVol, transformParams=None, maxZRoll=180):
    # vol format [B,Z,X,Y]
    if transformParams==None:
        angle, transl, scale, shear = TF.RandomAffine.get_params((-180,180), (0.1,0.1), (0.9,1.1), (0,0), currVol.shape[2:4])
        zRoll = int(maxZRoll*torch.rand(1)-maxZRoll//2)
        transformParams = {'angle':angle, 'transl':transl, 'scale':scale, 'shear':shear, 'zRoll':zRoll}
    
    zRoll = transformParams['zRoll']
    for nVol in range(currVol.shape[0]):
        for nDepth in range(currVol.shape[1]):
            currDepth = TF.functional.to_pil_image(currVol[nVol,nDepth,...].float())
            currDepth = TF.functional.affine(currDepth, transformParams['angle'], transformParams['transl'], transformParams['scale'], transformParams['shear'])
            currVol[nVol,nDepth,...] = TF.functional.to_tensor(currDepth)
    currVol = currVol.roll(zRoll, 1)
    if zRoll>=0:
        currVol[:,0:zRoll,...] = 0
    else:
        currVol[:,zRoll:,...] = 0
    return currVol, transformParams

def plot_param_grads(writer, net, curr_it, prefix=""):
    for tag, parm in net.named_parameters():
        if parm.grad is not None:
            writer.add_histogram(prefix+tag, parm.grad.data.cpu().numpy(), curr_it)
            assert not torch.isnan(parm.grad.sum()), print("NAN in: " + str(tag) + "\t\t")

def compute_histograms(gt, pred, input_img, n_bins=1000):
    volGTHist = torch.histc(gt, bins=n_bins, max=gt.max().item())
    volPredHist = torch.histc(pred, bins=n_bins, max=pred.max().item())
    inputHist = torch.histc(input_img, bins=n_bins, max=input_img.max().item())
    return volGTHist,volPredHist,inputHist


def match_histogram(source, reference):
    isTorch = False
    source = source / source.max() * reference.max()
    if isinstance(source, torch.Tensor):
        source = source.cpu().numpy()
        isTorch = True
    if isinstance(reference, torch.Tensor):
        reference = reference[:source.shape[0],...].cpu().numpy()

    matched = match_histograms(source, reference, multichannel=False)
    if isTorch:
        matched = torch.from_numpy(matched)
    return matched

def load_PSF(filename, n_depths=120,data_scale=[1,1]):
    # Load PSF
    try:
        mat_data = loadmat7(filename)
        # Get first data key (ignore MATLAB metadata)
        psf_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
        psfIn = torch.from_numpy(mat_data[psf_key]).permute(2,0,1).unsqueeze(0)
    except:
        try:
            import h5py
            psfFile = h5py.File(filename,'r')
            psf_key = [k for k in psfFile.keys() if not k.startswith('__')][0]
            psfIn = torch.from_numpy(psfFile.get(psf_key)[:]).permute(0,2,1).unsqueeze(0)
        except:
            mat_data = loadmat73(filename)
            psf_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
            psfIn = torch.from_numpy(mat_data[psf_key]).permute(2,0,1).unsqueeze(0)

    if psfIn.shape[2]==1025:
            psfIn = psfIn[:,:,1:,1:]
    psfIn = tv.transforms.functional.resize(psfIn, (round(psfIn.shape[-2]*data_scale[0]),
                                                    round(psfIn.shape[-1]*data_scale[1])),
                                                    antialias=True)

    # Make a square PSF
    min_psf_size = min(psfIn.shape[-2:])
    psf_pad = [min_psf_size-psfIn.shape[-1], min_psf_size-psfIn.shape[-2]]
    psf_pad = [psf_pad[0]//2, psf_pad[0]//2, psf_pad[1],psf_pad[1]]
    psfIn = F.pad(psfIn, psf_pad)

    # Grab only needed depths, 96:160 --> 97th to 160th from 1-256
    psfIn = psfIn[:, psfIn.shape[1]//2- n_depths//2 : psfIn.shape[1]//2+n_depths//2, ...]
    
    # # Normalize psfIn such that each depth sum is equal to 1
    # for nD in range(psfIn.shape[1]):
    #     psfIn[:,nD,...] = psfIn[:,nD,...] / psfIn[:,nD,...].sum()
    
    psf_sum = psfIn.sum()
    psfIn = psfIn / psf_sum
    
    return psfIn

def get_lenslet_centers(filename):
    x,y = [], []
    with open(filename,'r') as f:
        reader = csv.reader(f,delimiter='\t')
        for row in reader:
            x.append(int(row[0]))
            y.append(int(row[1]))
    lenslet_coords = torch.cat((torch.IntTensor(x).unsqueeze(1),
                                torch.IntTensor(y).unsqueeze(1)),1)
    return lenslet_coords

def load_PSF_OTF(filename, vol_size, n_split=1, n_depths=120, data_scale=[1,1], device="cpu",
                 calc_max=False, psfIn=None, compute_transpose=False,
                 n_lenslets=9, lenslet_centers_file_out='lenslet_centers_python.txt',
                 recalc_lenslet_centers=False):
    
    import os 
    # import glob
    # if os.path.isdir(filename):
    #     mat_files = glob.glob(os.path.join(filename, "*.mat"))
    #     mat_files = sorted(mat_files)[:5]  # Take first 5 files
    #     print(f"Loading 5 PSF files from folder: {filename}")
                    
    if not os.path.exists(filename):
        print(f"File not found: {filename}")  
    
    # Load PSF
    if psfIn is None:
        psfIn = load_PSF(filename, n_depths, data_scale)
        print(">>> PSF loaded", psfIn.shape)

    psf_shape = torch.tensor(psfIn.shape[2:])
    print(">>> PSF shape: " + str(psf_shape))
    vol = torch.rand(1,psfIn.shape[1], vol_size[0], vol_size[1], device=device)
    print(">>> Vol rand shape: " + str(vol.shape))
    img, OTF = fft_conv_split(vol, psfIn.float().detach().to(device), psf_shape, n_split=n_split, device=device)
    
    OTF = OTF.detach()
    return OTF, psf_shape, psfIn

    # # Load all 5 PSFs
    # psfs = []
    # for i, mat_file in enumerate(mat_files):
    #     print(f"Loading PSF {i+1}: {os.path.basename(mat_file)}")
    #     psf = load_PSF(mat_file, n_depths, data_scale)
    #     psfs.append(psf) # Shape: (5, n_depths, H, W)

    # first_psf = psfs[0]
    # n_psfs = len(psfs)  # Use len() instead of .shape[0]
    # psf_shape = torch.tensor(first_psf.shape[2:])  # Get shape from first PSF
    # vol = torch.rand(1, first_psf.shape[1], vol_size[0], vol_size[1], device=device)
    
    # print(f"PSF slice values: {psfIn[0, n_depths//2, ...]}")

    # if len(lenslet_centers_file_out)>0 and recalc_lenslet_centers:
    #     print(">>> Recalculating lenslet centers, original coordinates: ",get_lenslet_centers(lenslet_centers_file_out))
    #     find_lenslet_centers(psfIn[0,n_depths//2 - 1,...].numpy(), n_lenslets=n_lenslets, file_out_name=lenslet_centers_file_out)
    
    # if not os.path.exists(lenslet_centers_file_out):
    #     print(f"Lenslet centers file not found: {lenslet_centers_file_out}")
        
    # lenscoords = get_lenslet_centers(lenslet_centers_file_out)
    # print(">>> Using", lenscoords.shape[0], "lenslets, Current coords:", lenscoords)

    # if calc_max:
    #     psfMaxCoeffs = torch.amax(psfIn, dim=[0,2,3])

    # psfIn0 = psfIn
    
    # if psfIn.shape[1] > vol_size[0] or psfIn.shape[2] > vol_size[1]:
    #     psfIn = extract_psf_patches(psfIn, lenscoords, vol_size[0] // 3)
    #     print(">>> PSF shape after extraction: " + str(psfIn.shape))
        
    # psf_shape = torch.tensor(psfIn.shape[2:])
    # print(">>> PSF shape: " + str(psf_shape))
    # vol = torch.rand(1,psfIn.shape[1], vol_size[0], vol_size[1], device=device)
    # print(">>> Vol rand shape: " + str(vol.shape))
    
    # all_otfs = []
    # for psf_idx in range(n_psfs):
    #     current_psf = psfs[psf_idx]   
    #     img, OTF = fft_conv_split(vol, current_psf.float().detach().to(device), psf_shape, n_split=n_split, device=device)
    #     OTF = OTF.detach()
    #     all_otfs.append(OTF)
           
    # return all_otfs, psf_shape

    # if compute_transpose:
    #     OTFt = torch.real(OTF) - 1j * torch.imag(OTF)
    #     OTF = torch.cat((OTF.unsqueeze(-1), OTFt.unsqueeze(-1)), 4)
    # if calc_max:
    #     return OTFc, psfMaxCoeffs
    # else:
    #     return OTF,psf_shape,psfIn0

def find_lenslet_centers(img, n_lenslets=9, file_out_name='lenslet_centers_python.txt'):
    fp2 = findpeaks.findpeaks()
    
    image_divisor = 1 # To find the centers faster
    img = findpeaks.stats.resize(img, size=(img.shape[0]//image_divisor,img.shape[1]//image_divisor))
    print(f"Image size: {img.shape}")
    # img_normalized = (img - img.min()) / (img.max() - img.min())
   
    results_s = peak_local_max(img, min_distance=50, num_peaks = n_lenslets)
    def sorting_key(center):
        x, y = center
        # Group x into ranges [0, 512), [512, 1024), [1024, 1536)
        x_group = x // (img.shape[0]/3)
        return (x_group, y)  # Sort by x_group first, then by y
    results_s = sorted(results_s, key = sorting_key)   
    results_s = np.array(results_s)  # Convert back to a NumPy array 
    # print(f"peak localized: {results}")
   
    results = results_s * image_divisor
    # print(f"Scaled peaks: {results}")
    
    # # Show image with peaks
    # plt.imshow(img, cmap='gray')
    # plt.plot(results_s[:, 1], results_s[:, 0], 'r.', markersize=5)
    # plt.title('Peaks in the Image')
    # plt.axis('off')
    # plt.show()

    # fp2 = findpeaks.findpeaks(method='gradient')
    # results_2 = fp2.fit(img)

    # # results_2 = fp2.fit(img)
    # limit_min = fp2.results['persistence'][0:n_lenslets+1]['score'].min()

    # # Initialize topology
    # fp = findpeaks.findpeaks(method='topology', limit=limit_min)
    # # make the fit
    # results = fp.fit(img)
    # # Make plot
    # # fp.plot_persistence()
    # # fp.plot()    
    # results = np.ndarray([n_lenslets,2], dtype=int)
    # for ix,data in enumerate(fp.results['groups0']):
    #     results[ix] = np.array(data[0], dtype=int) * image_divisor
    
    if len(file_out_name) > 0:
        print(">>> Fitted lenslet centers to " + file_out_name)
        np.savetxt(file_out_name, results, fmt='%d', delimiter='\t')

    return results

# Aid functions for getting information out of directory names
def get_intensity_scale_from_name(name):
    intensity_scale_sparse = re.match(r"^.*_(\d*)outScaleSp",name)
    if intensity_scale_sparse is not None:
        intensity_scale_sparse = int(intensity_scale_sparse.groups()[0])
    else:
        intensity_scale_sparse = 1

    intensity_scale_dense = re.match(r"^.*_(\d*)outScaleD",name)
    if intensity_scale_dense is not None:
        intensity_scale_dense = int(intensity_scale_dense.groups()[0])
    else:
        intensity_scale_dense = 1
    return intensity_scale_dense,intensity_scale_sparse

def get_number_of_frames(name):
    n_frames = re.match(r"^.*_(\d*)timeF",name)
    if n_frames is not None:
        n_frames = int(n_frames.groups()[0])
    else:
        n_frames = 1
    return n_frames


def net_get_params(net):
    if hasattr(net, 'module'):
        return net.module
    else:
        return net


def center_crop(layer, target_size, pad=0):
    _, _, layer_height, layer_width = layer.size()
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    return layer[
        :, :, (diff_y - pad) : (diff_y + target_size[0] - pad), (diff_x - pad) : (diff_x + target_size[1] - pad)
    ]

def inputbkg_est(curr_img_stack):
    bkg_btmright = curr_img_stack[...,-5:,-5:].flatten(start_dim = -2)
    bkg_btmleft = curr_img_stack[...,-5:,:5].flatten(start_dim = -2)
    bkg_topright = curr_img_stack[...,:5,-5:].flatten(start_dim = -2)
    bkg_topleft = curr_img_stack[...,:5,:5].flatten(start_dim = -2)
    bkg = torch.cat((bkg_btmright,bkg_btmleft,bkg_topright,bkg_topleft),dim=-1).topk(25,dim=-1).values.mean(-1)
    return bkg

def extract_psf_patches(psf, lenslet_coords, patch_size=171):
    """
    Extract patches from the PSF based on lenslet centers and stitch them into a 3x3 grid.

    Args:
        psf (torch.Tensor): The PSF tensor of shape (B, D, H, W) or (D, H, W).
        lenslet_coords (torch.Tensor): Tensor of shape (n_lenslets, 2) containing lenslet centers.
        patch_size (int): The size of each patch to extract (default is 171 for 512/3).

    Returns:
        torch.Tensor: Extracted patches stitched into a smaller PSF of shape (B, D, 512, 512) or (D, 512, 512).
    """
    # Handle batch dimension
    if psf.ndim == 4:  # Shape: (B, D, H, W)
        batch_size, psf_depth, psf_height, psf_width = psf.shape
    elif psf.ndim == 3:  # Shape: (D, H, W)
        batch_size = 1
        psf_depth, psf_height, psf_width = psf.shape
        psf = psf.unsqueeze(0)  # Add batch dimension
    else:
        raise ValueError("PSF must have 3 or 4 dimensions (D, H, W) or (B, D, H, W).")

    half_patch = patch_size // 2

    # Initialize a tensor to store the stitched PSF
    stitched_psf = torch.zeros((batch_size, psf_depth, patch_size*3, patch_size*3), device=psf.device, dtype=psf.dtype)

    # Ensure lenslet_coords has exactly 9 centers (3x3 grid)
    assert lenslet_coords.shape[0] == 9, "lenslet_coords must contain exactly 9 centers for a 3x3 grid."

    # Iterate over the 3x3 grid of lenslet coordinates
    for i, (center_x, center_y) in enumerate(lenslet_coords):
        # Calculate the patch boundaries
        start_x = center_x - half_patch
        end_x = start_x + patch_size 
        start_y = center_y - half_patch
        end_y = start_y + patch_size 

        # Extract the patch for each batch
        patch = psf[:, :, start_x:end_x, start_y:end_y]
        # if i == 4:
        #     plt.figure(figsize=(8, 8))
        #     plt.imshow(patch[0, 32, :, :].cpu().numpy(), cmap='viridis')
        #     plt.show()
            
        # Determine the position in the stitched PSF
        grid_x = i % 3  # Column index (0, 1, 2)
        grid_y = i // 3  # Row index (0, 1, 2)

        # Calculate the placement boundaries in the stitched PSF
        stitched_start_x = grid_x * patch_size
        stitched_end_x = stitched_start_x + patch_size 
        stitched_start_y = grid_y * patch_size
        stitched_end_y = stitched_start_y + patch_size 

        # Place the patch in the stitched PSF
        stitched_psf[:, :, stitched_start_y:stitched_end_y, stitched_start_x:stitched_end_x] = patch
      
    # # Remove batch dimension if it was added
    # if batch_size == 1:
    #     stitched_psf = stitched_psf.squeeze(0)

    return stitched_psf

def crop_img(img, lenslet_coords, patch_size=171):
    """
    Extract patches from the FLFM imgs based on lenslet centers and stitch them into a 3x3 grid.
    """
    # Handle batch dimension
    if img.ndim == 3:  # Shape: (D, H, W)
        img_depth, img_height, img_width = img.shape
    else:
        raise ValueError("img must have 2 or 3 dimensions.")

    half_patch = patch_size // 2

    # Initialize a tensor to store the stitched PSF
    stitched_img = np.zeros((img_depth, patch_size*3, patch_size*3))

    # Ensure lenslet_coords has exactly 9 centers (3x3 grid)
    assert lenslet_coords.shape[0] == 9, "lenslet_coords must contain exactly 9 centers for a 3x3 grid."

    # Iterate over the 3x3 grid of lenslet coordinates
    for i, (center_x, center_y) in enumerate(lenslet_coords):
        # Calculate the patch boundaries
        start_x = center_x - half_patch
        end_x = start_x + patch_size
        start_y = center_y - half_patch
        end_y = start_y + patch_size

        # Extract the patch for each batch
        patch = img[:, start_x:end_x, start_y:end_y]
        # if i == 4:
        #     plt.figure(figsize=(8, 8))
        #     plt.imshow(patch[0, 32, :, :].cpu().numpy(), cmap='viridis')
        #     plt.show()
            
        # Determine the position in the stitched PSF
        grid_x = i % 3  # Column index (0, 1, 2)
        grid_y = i // 3  # Row index (0, 1, 2)

        # Calculate the placement boundaries in the stitched PSF
        stitched_start_x = grid_x * patch_size
        stitched_end_x = stitched_start_x + patch_size
        stitched_start_y = grid_y * patch_size
        stitched_end_y = stitched_start_y + patch_size

        # Place the patch in the stitched PSF
        stitched_img[:, stitched_start_y:stitched_end_y, stitched_start_x:stitched_end_x] = patch
      
    # # Remove batch dimension if it was added
    # if batch_size == 1:
    #     stitched_psf = stitched_psf.squeeze(0)

    return stitched_img