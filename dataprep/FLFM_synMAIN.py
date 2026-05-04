from os import system,mkdir,environ
from os.path import exists
system('cls')
from FLFMgenWFSYN_333FOVcluster import *
from FLFMgenWFSYN_333FOVspline import genRandSpline
from FLFMdataAug import addAperture,transform_volume #,transform_points,aperturePoints
from FLFMconvWF import convolveWFM_fft

import numpy as np
import cupy as cp
from torch import from_numpy
from imageio.v3 import imread, imwrite
from argparse import ArgumentParser
from clij2fft.richardson_lucy import richardson_lucy
from cupyx.scipy.ndimage import gaussian_filter

from sys import path as sys_path
sys_path.append('E:\\24_Largefield_flfm\\DLFLFM\\dl_pytorch_16x_v4\\')
# from utilities.util_setupSystem import setupParams

print(">>> Start generating synthetic data for FLFM training")
# #############################################################################################################
gtcfolder = "E:\\24_Largefield_flfm\\DLFLFM\\expdata16x\\syn_gtc_cluster3_WF445_big_dense_s2\\"
gtc_augfolder = "E:\\24_Largefield_flfm\\DLFLFM\\expdata16x\\syn_gtc_cluster3_WF445_big_dense_aug_s2\\"
cfc_augfolder = "E:\\24_Largefield_flfm\\DLFLFM\\expdata16x\\syn_cfc_cluster3_WF445_big_dense_aug_s2\\"
cfc_rld_augfolder = "E:\\24_Largefield_flfm\\DLFLFM\\expdata16x\\syn_cfc_clusterRLD3_WF445_big_dense_aug_s2\\"

# YMLFILENAME = 'Z:/Xuanwen/DLFLFM/dl_pytorch_100x_v2/paraymls/Train_useBeads_20240429r.yml' # set up parameter file
# args = setupParams(YMLFILENAME,default=False) # set up parameters, default=True to use default parameters
# flf_folder = args.train_folder_in

if not exists(gtcfolder):mkdir(gtcfolder)
if not exists(gtc_augfolder):mkdir(gtc_augfolder)
if not exists(cfc_augfolder):mkdir(cfc_augfolder)
if not exists(cfc_rld_augfolder):mkdir(cfc_rld_augfolder)
# if not exists(flf_folder):mkdir(flf_folder)
# #############################################################################################################
psffolder = r"E:\24_Largefield_flfm\DLFLFM\expdata16x\wfPSF_xy256z256_px1800z1600nm_BD.tif"
# psffolder = "Z:\\Xuanwen\\DLFLFM\\expdata100x\\Simu20231212Wv512\\PSFGAUint_20231215_Blue_gly_10um_deconv.tif"
# psffolder = "Z:\\Xuanwen\\DLFLFM\\expdata100x\\Simu20231212Wv599\\PSFGAUint_20231215_Green_gly_10um_deconv.tif"
# psffolder = "Z:\\Xuanwen\\DLFLFM\\expdata100x\\Simu20231212Wv680\\PSFGAUint_20231215_Red_gly_10um_deconv.tif"
psfIn_center = cp.asarray(imread(psffolder)).astype(np.float32)
psfIn_center = psfIn_center/cp.max(psfIn_center)
print(">>> PSF for WF CONV loaded in shape:", psfIn_center.shape)
psffolder = r"E:\24_Largefield_flfm\DLFLFM\expdata16x\wfPSF_xy256z256_px1800z1600nm.tif"
# psffolder = "Z:\\Xuanwen\\DLFLFM\\expdata100x\\Simu20231212Wv512\\PSFGAUint_20231215_Blue_gly_10um.tif"
# psffolder = "Z:\\Xuanwen\\DLFLFM\\expdata100x\\Simu20231212Wv599\\PSFGAUint_20231215_Green_gly_10um.tif"
# psffolder = "Z:\\Xuanwen\\DLFLFM\\expdata100x\\Simu20231212Wv680\\PSFGAUint_20231215_Red_gly_10um.tif"
psfIn_rld = imread(psffolder).astype(np.float32)
psfIn_rld = psfIn_rld/np.max(psfIn_rld)
print(">>> PSF for RL DCONV loaded in shape:", psfIn_rld.shape)
# #############################################################################################################
# #############################################################################################################

rld_iters = 10

def getSynData(dataargs,iters = 50):
    # ####################### generate synthetic data #######################
    if not(dataargs.isreplace) and exists(gtcfolder+dataargs.gtcfilename):
        print(">>> File already exists, skip:", dataargs.gtcfilename)
    else:
        if dataargs.isSpline:
            print(">>> Generating spline data")
            volout = genRandSpline(dataargs.vol_shape,
                                   global_num=20,
                                   global_z_sparsity = 1)
        else:
            volout,_ = genRandPSF(dataargs.vol_shape,
                                global_num=dataargs.global_num,
                                local_density = dataargs.local_density,
                                cluster_num = dataargs.cluster_num,
                                global_z_sparsity = dataargs.global_z_sparsity,
                                local_z_sparsity = dataargs.local_z_sparsity,
                                cluster_z_sparsity = dataargs.cluster_z_sparsity,
                                density=dataargs.density)
        imwrite(gtcfolder+dataargs.gtcfilename, volout)
        print(">>> Saved:", dataargs.gtcfilename)
    # ####################### generate augmented data #######################
    if not(dataargs.isreplace) and exists(gtc_augfolder+'/'+dataargs.gtcfilename):
        print(">>> File already exists, skip:", dataargs.gtcfilename)
    else:
        # ####################### process initial data #######################
        currVol = addAperture(volout.astype(np.float32),256,256)
        cfc = convolveWFM_fft(cp.asarray(currVol).astype(cp.float32),psfIn_center)
        imwrite(gtc_augfolder+'/'+dataargs.gtcfilename, (currVol*1.0/np.max(currVol)*40000).astype(np.uint16))
        print(">>> Saved original file 0:", dataargs.gtcfilename, " in shape:", currVol.shape)
        imwrite(cfc_augfolder+"\\"+dataargs.gtcfilename, (cfc/np.max(cfc)*40000).get().astype(np.uint16))
        print("*** Saved original WF file 0:", dataargs.gtcfilename,end="\n")
        cfc = (cfc/np.max(cfc)*40000).get()
        imdeconv = richardson_lucy(cfc, psfIn_rld, numiterations=iters)
        imdeconv = imdeconv/np.max(imdeconv)*np.max(cfc)
        imwrite(cfc_rld_augfolder+'/'+dataargs.gtcfilename, imdeconv.astype(np.uint16))
        print("*** Saved original RLD file 0:", dataargs.gtcfilename,end="\n")
        # ####################### process augmented data #######################
        currVol = from_numpy(currVol).unsqueeze(0)#.type(torch_float)
        print(">>> Max of currVol:", currVol.max())
        print(">>> currVol in Tensor:", dataargs.gtcfilename, " in shape:", currVol.shape)
        if dataargs.augfactor>1:
            for nAug in range(dataargs.augfactor-1):
                dataargs.gtc_augfilename = dataargs.gtcfilename.replace('.tif','_'+str(nAug+1)+'.tif')
                currVolAug,transformParams = transform_volume(currVol)
                currVolAug = currVolAug.squeeze(0).numpy().astype(np.float32)
                currVolAug = addAperture(currVolAug,256,256).astype(np.uint16)
                cfc = convolveWFM_fft(cp.asarray(currVolAug),psfIn_center)
                imwrite(gtc_augfolder+'/'+dataargs.gtc_augfilename, currVolAug)
                print(">>> Saved augmented file ",nAug+1,":", dataargs.gtc_augfilename, " in shape:", currVolAug.shape)
                imwrite(cfc_augfolder+'/'+dataargs.gtc_augfilename, (cfc/np.max(cfc)*40000).get().astype(np.uint16))
                print(">>> Saved augmented WF file ",nAug+1,":", dataargs.gtc_augfilename)
                cfc = (cfc/np.max(cfc)*40000).get()
                imdeconv = richardson_lucy(cfc, psfIn_rld, numiterations=iters)
                imdeconv = imdeconv/np.max(imdeconv)*np.max(cfc)
                imwrite(cfc_rld_augfolder+'/'+dataargs.gtc_augfilename, imdeconv.astype(np.uint16))
                print("*** Saved original RLD file ",nAug+1,":", dataargs.gtc_augfilename,end="\n")

if __name__=="__main__":
    parser = ArgumentParser()
    dataargs = parser.parse_args()
    dataargs.vol_shape = (256,256,256)
    # ############################################################
    params = [
        [25, 0.15, 8,  1, 1, 1, 0.15],  # Moderate tissue density
        [40, 0.12, 6,  1, 1, 1, 0.12],  # Higher cell count, balanced
        [35, 0.18, 10, 1, 1, 1, 0.18],  # Dense tissue areas
        [15, 0.24, 12, 1, 1, 1, 0.20],  # Small tissue patches, dense
        [50, 0.09, 8,  1, 1, 1, 0.10],  # Large tissue, moderate density
        [30, 0.15, 6,  1, 1, 1, 0.14],  # Balanced distribution
    ]
    # ############################################################
    # dataargs.global_num = 10
    # dataargs.local_density = 0.5
    # dataargs.cluster_num = 25
    # dataargs.global_z_sparsity = 1
    # dataargs.local_z_sparsity = 1
    # dataargs.cluster_z_sparsity = 1
    # dataargs.density = 0.05

    dataargs.isreplace = True
    dataargs.augfactor = 10

    for ii in range(0,60):
        param_select_ind = np.random.randint(0,len(params))
        dataargs.global_num = params[param_select_ind][0]
        dataargs.local_density = params[param_select_ind][1]
        dataargs.cluster_num = params[param_select_ind][2]
        dataargs.global_z_sparsity = params[param_select_ind][3]
        dataargs.local_z_sparsity = params[param_select_ind][4]
        dataargs.cluster_z_sparsity = params[param_select_ind][5]
        dataargs.density = params[param_select_ind][6]
        dataargs.gtcfilename = "wf_"+str(ii).zfill(4)+".tif"
        if ii>180:
            dataargs.isSpline = True
        else:
            dataargs.isSpline = False
        getSynData(dataargs,iters = rld_iters)

    # ###########################################################
    # dataargs.global_num = 30
    # dataargs.local_density = 0.2
    # dataargs.cluster_num = 8
    # dataargs.global_z_sparsity = 1
    # dataargs.local_z_sparsity = 1
    # dataargs.cluster_z_sparsity = 1
    # dataargs.density = 0.03

    # for ii in range(40,80):
    #     dataargs.gtcfilename = "wf_"+str(ii).zfill(4)+".tif"
    #     if ii>180:
    #         dataargs.isSpline = True
    #     else:
    #         dataargs.isSpline = False
    #     getSynData(dataargs,iters = rld_iters)

    # ############################################################
    # dataargs.global_num = 50
    # dataargs.local_density = 0.2
    # dataargs.cluster_num = 10
    # dataargs.global_z_sparsity = 1
    # dataargs.local_z_sparsity = 1
    # dataargs.cluster_z_sparsity = 1
    # dataargs.density = 0.01

    # for ii in range(80,120):
    #     dataargs.gtcfilename = "wf_"+str(ii).zfill(4)+".tif"
    #     if ii>180:
    #         dataargs.isSpline = True
    #     else:
    #         dataargs.isSpline = False
    #     getSynData(dataargs,iters = rld_iters)

