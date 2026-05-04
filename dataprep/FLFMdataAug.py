from torch import rand as torchRand
from torch import from_numpy,max
# from torch import float as torch_float
from numpy import uint16 as np_uint16
from numpy import float32 as np_float32
from numpy import max as np_max
from torchvision.transforms import RandomAffine
from torchvision.transforms.functional import to_pil_image, to_tensor, affine
from glob import glob
from imageio.v3 import imread, imwrite
from os.path import basename, exists
import matplotlib.pyplot as plt
import numpy as np

# Random transformation of volume, for augmentation
def transform_volume(currVol, transformParams=None, maxZRoll=10):
    # vol format [B,Z,X,Y]
    currVol = currVol.cuda()
    if transformParams==None:
        angle, transl, scale, shear = RandomAffine.get_params((-180,180), (0.08,0.08), (0.9,1.1), (0,0), currVol.shape[2:4])
        zRoll = int(maxZRoll*torchRand(1)-maxZRoll//2)
        transformParams = {'angle':angle, 'transl':transl, 'scale':scale, 'shear':shear, 'zRoll':zRoll}

    zRoll = transformParams['zRoll']
    # print(">>> Max of currVol:", currVol.max())
    for nVol in range(currVol.shape[0]):
        for nDepth in range(currVol.shape[1]):
            currDepth = currVol[nVol,nDepth,...].float().unsqueeze(0).unsqueeze(0)
            # print(">>> shape of currDepth:", currDepth.shape)
            # print(">>> Max of currDepth:", currDepth.max())
            currDepth = affine(currDepth, transformParams['angle'], transformParams['transl'], transformParams['scale'], transformParams['shear'])
            currVol[nVol,nDepth,...] = currDepth
    currVol = currVol.roll(zRoll, 1)
    if zRoll>=0:
        currVol[:,0:zRoll,...] = 0
    else:
        currVol[:,zRoll:,...] = 0
    currVol = currVol.cpu()
    return currVol, transformParams

def transform_points(total_points,volsize=(155,1024,1024),transformParams=None):
    # transform points
    # print("transformParams: ", transformParams)
    t_angle = transformParams['angle'] * np.pi / 180.0
    t_trnsl = transformParams['transl']
    t_scale = transformParams['scale']
    t_zRoll = transformParams['zRoll']
    transformed_points = total_points
    transformed_points[:,1] = t_scale * total_points[:,2] * np.sin(t_angle) +\
        t_scale * total_points[:,1] * np.cos(t_angle) + t_trnsl[0] # y' = k*x*sin(theta) + k*y*cos(theta) + n
    transformed_points[:,2] = t_scale * total_points[:,2] * np.cos(t_angle) -\
        t_scale * total_points[:,1] * np.sin(t_angle) + t_trnsl[1] # z' = k*x*cos(theta) - k*y*sin(theta) + m
    transformed_points[:,0] = (total_points[:,0] + t_zRoll) % volsize[0]
    # omit the points outside the volume
    transformed_points = transformed_points[transformed_points[:,0]>=0]
    transformed_points = transformed_points[transformed_points[:,0]<volsize[0]]
    transformed_points = transformed_points[transformed_points[:,1]>=0]
    transformed_points = transformed_points[transformed_points[:,1]<volsize[1]]
    transformed_points = transformed_points[transformed_points[:,2]>=0]
    transformed_points = transformed_points[transformed_points[:,2]<volsize[2]]
    print(">>> transformed_points size:", transformed_points.shape)
    return transformed_points,transformParams

def addAperture(vol,radius,depth):
    vol_midindex_0 = int(vol.shape[0]/2)
    vol_midindex_1 = int(vol.shape[1]/2)
    vol_midindex_2 = int(vol.shape[2]/2)
    circle_aperture = np.zeros_like(vol[0,:,:])
    X,Y = np.meshgrid(np.arange(-vol_midindex_1,vol_midindex_1),np.arange(-vol_midindex_2,vol_midindex_2))
    circle_aperture[np.sqrt(X**2+Y**2)<radius] = 1.0
    depth = int(depth/2)
    volout = np.zeros_like(vol)
    volout[vol_midindex_0-depth:vol_midindex_0+depth,:,:] = vol[vol_midindex_0-depth:vol_midindex_0+depth,:,:]*circle_aperture[None,:,:]
    return volout

def aperturePoints(total_points,volsize=(155,1024,1024),radius=500,depth=100):
    vol_midindex_1 = int(volsize[1]/2)
    vol_midindex_2 = int(volsize[2]/2)
    total_points = total_points[((total_points[:,1]-vol_midindex_1)**2+(total_points[:,2]-vol_midindex_2)**2)<(radius**2)]
    total_points = total_points[total_points[:,0]>=volsize[0]-depth]
    total_points = total_points[total_points[:,0]<volsize[0]+depth]
    print(">>> total_apertured_points size:", total_points.shape)
    return total_points

def getFLFMdataAug(args,augSaveFolder,augfactor=1,isreplace = False):
    train_folder_gt = args.train_folder_gt_noaug
    all_filelist = sorted(glob(train_folder_gt+'/*.tif'))
    print('# Loading ground truths from: ', train_folder_gt, ' (', len(all_filelist), ' stacks)')
    for nFile in all_filelist:
        filename = basename(nFile)
        if not(isreplace) and exists(augSaveFolder+'/'+filename):
            print(">>> File already exists, skip:", filename)
        else:
            currVol = imread(nFile)
            if currVol.shape[1]//2!=currVol.shape[1]/2:
                currVol = currVol[:,0:-1,0:-1]
            currVol = addAperture(currVol.astype(np_float32),500,100)
            imwrite(augSaveFolder+'/'+filename, (currVol*1.0/np_max(currVol)*60000).astype(np_uint16))
            print(">>> Saved original file:", filename, " in shape:", currVol.shape)
            currVol = currVol.astype(np_float32)
            currVol = from_numpy(currVol).unsqueeze(0)#.type(torch_float)
            print(">>> Max of currVol:", currVol.max())
            print(">>> currVol in Tensor:", filename, " in shape:", currVol.shape)
            plt.ion()
            fig, ax = plt.subplots(num=1)
            imshowHDL = ax.imshow(max(currVol,dim=1).values.squeeze(0))
            fig.canvas.draw()
            fig.canvas.flush_events()
            if augfactor>1:
                for nAug in range(augfactor-1):
                    filenameAug = filename.replace('.tif','_'+str(nAug+1)+'.tif')
                    currVolAug,_ = transform_volume(currVol)

                    imshowHDL.set_data(max(currVolAug,dim=1).values.squeeze(0))
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    print(">>> Max of currVolAug:", currVolAug.max())
                    currVolAug = currVolAug.squeeze(0).numpy().astype(np_float32)
                    currVolAug = addAperture(currVolAug,500,100).astype(np_uint16)
                    imwrite(augSaveFolder+'/'+filenameAug, currVolAug)
                    print(">>> Saved augmented file:", filenameAug, " in shape:", currVolAug.shape)



if __name__=="__main__":
    from os import system
    from os import name as system_name
    from sys import path as sys_path
    sys_path.append('Z:\\Xuanwen\\DLFLFM\\dl_pytorch_100x_v2\\')
    from utilities.util_setupSystem import setupParams

    system('cls' if system_name=='nt' else 'clear')
    YMLFILENAME = 'Z:/Xuanwen/DLFLFM/dl_pytorch_100x_v2\/paraymls/Train_useBeads_20240429.yml' # set up parameter file
    args = setupParams(YMLFILENAME,default=False) # set up parameters, default=True to use default parameters
    args.train_folder_gt_noaug = "Z:/Xuanwen/DLFLFM/dldatasets/RawDatasets/syn_gtc_333FOVcluster"
    getFLFMdataAug(args,"Z:\\Xuanwen\\DLFLFM\\dldatasets\\RawDatasets\\syn_gtc_333FOVcluster_aug",5,isreplace=True)


    
