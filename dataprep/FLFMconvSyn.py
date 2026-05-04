from glob import glob
from sys import path as sys_path
from os.path import basename, exists
from os import mkdir
from imageio.v3 import imread, imwrite
from torch import from_numpy
from torch.fft import rfft2, irfft2, ifftshift
from torch.nn.functional import interpolate,relu
from torchvision.transforms.functional import resize
import numpy as np

try:
    from utilities.util_setupPSFOTF import load_PSF
except ModuleNotFoundError:
    sys_path.append('Z:\\Xuanwen\\DLFLFM\\dl_pytorch_100x_v2\\')
    # print(">>> sys.path:", sys_path)
    from utilities.util_setupPSFOTF import load_PSF      

def convolveFLFM(currVol_padded,psfIn):
    synFLFMimg3d = irfft2(rfft2(currVol_padded)*rfft2(psfIn),
                          s=currVol_padded.shape[-2:])
    synFLFMimg3d = ifftshift(synFLFMimg3d,(-2,-1))
    return synFLFMimg3d.sum(1)

def padCurrVol(currVol,currVol_padded,bkgnoise):
    currVol = relu(currVol-bkgnoise) + bkgnoise
    x_start = (currVol_padded.shape[2] - currVol.shape[2]) // 2
    x_end = x_start + currVol.shape[2]
    y_start = (currVol_padded.shape[3] - currVol.shape[3]) // 2
    y_end = y_start + currVol.shape[3]
    print(">>> Replace currVol_padded center with currVol:",x_start, x_end, y_start, y_end)
    currVol_padded[:, :, x_start:x_end, y_start:y_end] = currVol
    return currVol_padded

def getFLFMSyn(args,psfIn=None,augfactor=1,isreplace = False):
    train_folder_gt = args.train_folder_gt
    train_folder_gt_noaug = args.train_folder_gt_noaug
    synSaveFolder = args.train_folder_in
    if not exists(synSaveFolder): mkdir(synSaveFolder)
    clipvalue = 1.0
    if psfIn is None:
        psfIn = load_PSF(args.psf_file, 100,1)
    print(">>> PSF loaded", psfIn.shape)   
    psfIn = interpolate(psfIn.unsqueeze(0), (round(psfIn.shape[1]*65/65),
                                             round(psfIn.shape[2]*145/65),
                                             round(psfIn.shape[3]*145/65)),
                                             mode='nearest-exact').squeeze(0)
    print(">>> PSF resized", psfIn.shape)

    # Load training ground truths
    all_filelist = sorted(glob(train_folder_gt+'/*.tif'))
    all_filelist_noaug = sorted(glob(train_folder_gt_noaug+'/*.tif'))
    print('# Loading ground truths from: ', train_folder_gt, ' (', len(all_filelist), ' stacks)')
    for nFile in all_filelist_noaug:
        filename = basename(nFile)
        savefilename = filename.replace('wf','lf')
        if not(isreplace) and exists(synSaveFolder+'/'+savefilename):
            print(">>> File already exists, skip:", savefilename)
        else:
            currVol = imread(train_folder_gt+'/'+filename).astype(np.float32)
            currVol = np.floor(currVol/clipvalue)*clipvalue
            if currVol.shape[1]//2!=currVol.shape[1]/2:
                currVol = currVol[:,0:-1,0:-1]
            if currVol.shape[0] > psfIn.shape[1]:
                currVol = currVol[(currVol.shape[0]//2- psfIn.shape[1]//2+1) : (currVol.shape[0]//2+psfIn.shape[1]//2+1),:,:] # crop the volume if it is larger than the PSF
            bkgnoise = currVol[:,1020:1024,1020:1024].mean()
            currVol_padded0 = psfIn*0+bkgnoise+np.random.rand(psfIn.shape[0],psfIn.shape[1],psfIn.shape[2],psfIn.shape[3])*np.minimum(bkgnoise*0.01,10)
            # processing strategy: pad currVol to the same size as psfIn, then convolve
            currVol_padded = padCurrVol(from_numpy(currVol).unsqueeze(0),currVol_padded0,bkgnoise)
            synFLFMimg = convolveFLFM(currVol_padded,psfIn).unsqueeze(0)
            synFLFMimg = resize(synFLFMimg,(currVol.shape[1],currVol.shape[2])).squeeze(0).squeeze(0).numpy()
            imwrite(synSaveFolder+'/'+savefilename,(synFLFMimg/np.max(synFLFMimg)*40000).astype(np.uint16))
            print(">>> Saved: ", savefilename)
            # ------------------- for augmentation -------------------
            for nAug in range(augfactor-1):
                nAugfilename = filename.replace('.tif','_'+str(nAug+1)+'.tif')
                savefilename = nAugfilename.replace('wf','lf')
                currVol = imread(train_folder_gt+'/'+nAugfilename).astype(np.float32)
                currVol = np.floor(currVol/clipvalue)*clipvalue
                if currVol.shape[1]//2!=currVol.shape[1]/2:
                    currVol = currVol[:,0:-1,0:-1]
                if currVol.shape[0] > psfIn.shape[1]:
                    currVol = currVol[currVol.shape[0]//2- psfIn.shape[1]//2+1 : currVol.shape[0]//2+psfIn.shape[1]//2+1,:,:] # crop the volume if it is larger than the PSF
                currVol_padded = padCurrVol(from_numpy(currVol).unsqueeze(0),currVol_padded0,bkgnoise)
                synFLFMimg = convolveFLFM(currVol_padded,psfIn).unsqueeze(0)
                synFLFMimg = resize(synFLFMimg,(currVol.shape[1],currVol.shape[2])).squeeze(0).squeeze(0).numpy()
                imwrite(synSaveFolder+'/'+savefilename,(synFLFMimg/np.max(synFLFMimg)*40000).astype(np.uint16))
                print(">>> Saved: ", savefilename)

if __name__=="__main__":
    from os import system
    from os import name as system_name
    sys_path.append('Z:\\Xuanwen\\DLFLFM\\dl_pytorch_100x_v2\\')
    from utilities.util_setupSystem import setupParams

    system('cls' if system_name=='nt' else 'clear')
    YMLFILENAME = 'Z:/Xuanwen/DLFLFM/dl_pytorch_100x_v3/paraymls/Train_useBeads_20250220g.yml' # set up parameter file
    args = setupParams(YMLFILENAME,default=False) # set up parameters, default=True to use default parameters
    args.train_folder_gt = "Z:/Xuanwen/DLFLFM/dldatasets/RawDatasets/syn_cfc_clusterRLD3_WF599_aug"
    args.train_folder_gt_noaug = "Z:/Xuanwen/DLFLFM/dldatasets/RawDatasets/syn_gtc_cluster3_WF599"
    # args.train_folder_gt = "Z:/Xuanwen/DLFLFM/dldatasets/RawDatasets/syn_gtc_diverse10_aug"
    # args.train_folder_gt_noaug = "Z:/Xuanwen/DLFLFM/dldatasets/RawDatasets/syn_gtc_diverse10"
    getFLFMSyn(args,augfactor=5,isreplace=False)