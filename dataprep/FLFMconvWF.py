from glob import glob
from sys import path as sys_path
from os.path import basename
from imageio.v3 import imread, imwrite
from scipy.io import loadmat
import numpy as np
# import matplotlib.pyplot as plt
from cupyx.scipy.ndimage import zoom, convolve
import cupy as cp
# from rich.progress import Progress, track
# from cupy.fft import rfft2, irfft2, ifftshift

def loadWFPSF(psffolder,windowsize=32):
    psfmat = loadmat(psffolder)
    psf = cp.asarray(psfmat['FLFPSF']).astype(cp.float32).transpose(2,0,1)
    if psf.shape[1]//2!=psf.shape[1]/2:
        psf_iso = zoom(psf[0:,0:-1,0:-1],(65.0/65,1,1))
    else:
        psf_iso = zoom(psf[0:,:,:],(65.0/65,1,1))
    if windowsize>0:
        psfIn_midindex_0 = int(psf_iso.shape[0]/2)
        psfIn_midindex_1 = int(psf_iso.shape[1]/2)
        psfIn_midindex_2 = int(psf_iso.shape[2]/2)
        psfIn_center = psf_iso[psfIn_midindex_0-windowsize:psfIn_midindex_0+windowsize,
                            psfIn_midindex_1-windowsize:psfIn_midindex_1+windowsize,
                            psfIn_midindex_2-windowsize:psfIn_midindex_2+windowsize]
        print(">>> psf loaded in shape:", psfIn_center.shape)
        return psfIn_center
    else:
        print(">>> psf loaded in shape:", psf_iso.shape)
        return psf_iso

def addAperture(vol,radius,depth):
    vol_midindex_0 = int(vol.shape[0]/2)
    vol_midindex_1 = int(vol.shape[1]/2)
    vol_midindex_2 = int(vol.shape[2]/2)
    circle_aperture = cp.zeros_like(vol[0,:,:])
    X,Y = cp.meshgrid(cp.arange(-vol_midindex_1,vol_midindex_1),cp.arange(-vol_midindex_2,vol_midindex_2))
    circle_aperture[cp.sqrt(X**2+Y**2)<radius] = 1.0
    depth = int(depth/2)
    volout = cp.zeros_like(vol)
    volout[vol_midindex_0-depth:vol_midindex_0+depth,:,:] = vol[vol_midindex_0-depth:vol_midindex_0+depth,:,:]*circle_aperture[None,:,:]
    return volout

def convolveWFM(currVol,psfIn,windowsize = 32):
    psfIn_midindex_0 = int(psfIn.shape[0]/2)
    psfIn_midindex_1 = int(psfIn.shape[1]/2)
    psfIn_midindex_2 = int(psfIn.shape[2]/2)
    psfIn_center = psfIn[psfIn_midindex_0-windowsize:psfIn_midindex_0+windowsize,
                         psfIn_midindex_1-windowsize:psfIn_midindex_1+windowsize,
                         psfIn_midindex_2-windowsize:psfIn_midindex_2+windowsize]
    currVol_conv = convolve(currVol,psfIn_center)
    return currVol_conv

def convolveWFM_fft(currVol,psfIn):
    # convolution
    currVol_conv = cp.fft.ifftshift(cp.fft.irfftn((cp.fft.rfftn(currVol))*(cp.fft.rfftn(psfIn))))
    return currVol_conv



if __name__=="__main__":
    from os import system
    from os.path import basename, exists
    system('cls')

    # load widefield PSF
    psffolder = "Z:\\Xuanwen\\DLFLFM\\dldatasets\\train\\TrainingExpData\\PSFFLFint_20240428_Red_Gly_10um\\PSFGAUint_20231215_Red_gly_10um.tif"
    windowsize = -1
    # psfIn_center = loadWFPSF(psffolder,windowsize)
    psfIn_center = cp.asarray(imread(psffolder)).astype(np.float32)
    print(">>> PSF loaded in shape:", psfIn_center.shape)

    # convolve gtc to cfc
    gtcfolder = "Z:\\Xuanwen\\DLFLFM\\dldatasets\\RawDatasets\\syn_gtc_333FOVcluster_aug\\"
    cfcfolder = "Z:\\Xuanwen\\DLFLFM\\dldatasets\\RawDatasets\\syn_cfc_333FOVcluster_aug\\"
    gtclist = glob(gtcfolder+"*.tif")
    isreplace = False
    for ii in range(0,len(gtclist)):
    # for ii in range(0,1):
        filename = basename(gtclist[ii])
        print(">>> Processing:", filename,end="\t")
        # if not(isreplace) and exists(cfcfolder+"\\"+filename):
        #     print("--- File already exists, skip:", filename)
        # else:
        vol = cp.asarray(imread(gtclist[ii])).astype(cp.float32)
        if vol.shape[1]//2!=vol.shape[1]/2:
            vol = vol[:,0:-1,0:-1]
        # vol = addAperture(vol,500,100)
        cfc = convolveWFM_fft(vol,psfIn_center).get()
        imwrite(cfcfolder+"\\"+filename, (cfc/np.max(cfc)*40000).astype(np.uint16))
        print("*** Saved:", filename,end="\n")
