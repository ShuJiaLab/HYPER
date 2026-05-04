import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter
from scipy import interpolate as interp
from imageio.v3 import imwrite
from rich.progress import track
import time

def destfunc(x,y,z,sigmaX=1,sigmaY=1,sigmaZ=1):
    return np.exp(((-x**2)/(2*sigmaX**2))**1)*np.exp(((-y**2)/(2*sigmaY**2))**1)*np.exp(((-z**2)/(2*sigmaZ**2))**1)

def rejection_sampling(iter=1000,xylimit=[-512,512],zlimit=[-77,77],
                       cell_sigmaX=300,cell_sigmaY=300,cell_sigmaZ=64,maxval=1):
    # samples = []
    x = np.random.uniform(xylimit[0],xylimit[1],size=iter)
    y = np.random.uniform(xylimit[0],xylimit[1],size=iter)
    z = np.random.uniform(zlimit[0],zlimit[1],size=iter)
    w = np.random.uniform(0, maxval,size=iter)
    destdist = destfunc(x,y,z,cell_sigmaX,cell_sigmaY,cell_sigmaZ)
    z = np.delete(z, np.where(w > destdist))
    y = np.delete(y, np.where(w > destdist))
    x = np.delete(x, np.where(w > destdist))
    samples = np.stack((z,y,x),axis=1)

    while samples.shape[0] < iter:
        itera_remain = iter - samples.shape[0]
        x = np.random.uniform(xylimit[0],xylimit[1],size=itera_remain)
        y = np.random.uniform(xylimit[0],xylimit[1],size=itera_remain)
        z = np.random.uniform(zlimit[0],zlimit[1],size=itera_remain)
        w = np.random.uniform(0, maxval,size=itera_remain)
        destdist = destfunc(x,y,z,cell_sigmaX,cell_sigmaY,cell_sigmaZ)
        z = np.delete(z, np.where(w > destdist))
        y = np.delete(y, np.where(w > destdist))
        x = np.delete(x, np.where(w > destdist))
        samples = np.concatenate((samples,np.stack((z,y,x),axis=1)),axis=0)

    return samples

def random_z_fill(z_sparse,num_centers):
    z_filled = z_sparse[np.random.randint(0,z_sparse.shape[0],size=num_centers)]
    return z_filled

def getGlobalClusterCenters(volsize,num_centers,z_sparsity = 1):
    # This function returns the global cluster centers for the given volume size
    # and number of centers. The centers are returned as a list of tuples.
    # The centers are chosen to be evenly spaced in the image.
    xylimit = [-volsize[1]/2,volsize[1]/2]
    zlimit = [-volsize[0]/2,volsize[0]/2]
    x = np.around(np.random.uniform(xylimit[0],xylimit[1],size = num_centers))
    y = np.around(np.random.uniform(xylimit[0],xylimit[1],size = num_centers))
    num_centers_z = int(num_centers*1.0/z_sparsity)
    z_sparse = np.around(np.random.uniform(zlimit[0],zlimit[1],size = num_centers_z))
    z = random_z_fill(z_sparse,num_centers)
    centers = np.stack((z,y,x),axis=1)
    return centers

def getLocalClusterCenters(volsize,init_center,num_subcenters,subcenters_sigma,z_sparsity = 1):
    # This function returns the local cluster centers for the given initial
    # center and number of subcenters. The centers are returned as a list of
    # tuples. The centers are chosen to be normally spaced centered on the initial center.
    ylimit = [-init_center[1],volsize[1]-init_center[1]]
    xlimit = [-init_center[2],volsize[2]-init_center[2]]
    xylimit = [np.maximum(xlimit[0],ylimit[0]),np.minimum(xlimit[1],ylimit[1])]
    zlimit = [-init_center[0],volsize[0]-init_center[0]]
    subcenters = rejection_sampling(iter=num_subcenters,
                                    xylimit=xylimit,zlimit=zlimit,
                                    cell_sigmaX=subcenters_sigma[2],
                                    cell_sigmaY=subcenters_sigma[1],
                                    cell_sigmaZ=subcenters_sigma[0],maxval=1)
    num_centers_z = int(num_subcenters*1.0/z_sparsity)
    subcenters_z = subcenters[:,0]
    z_sparse = subcenters_z[np.random.randint(0,subcenters_z.shape[0],size=num_centers_z)]
    subcenters_z = random_z_fill(z_sparse,num_subcenters)
    subcenters[:,0] = subcenters_z
    subcenters = subcenters + init_center
    return subcenters

def getSplineCoords(volsize,num_centers,z_sparsity = 1,smmoothness=0.0):
    xylimit = [-volsize[1]/2,volsize[1]/2]
    zlimit = [-volsize[0]/2,volsize[0]/2]
    x = np.around(np.random.uniform(xylimit[0],xylimit[1],size = num_centers))
    y = np.around(np.random.uniform(xylimit[0],xylimit[1],size = num_centers))
    num_centers_z = int(num_centers*1.0/z_sparsity)
    z_sparse = np.around(np.random.uniform(zlimit[0],zlimit[1],size = num_centers_z))
    z = random_z_fill(z_sparse,num_centers)
    tck,u = interp.splprep([x,y,z] ,s = smmoothness)
    xnew,ynew,znew = interp.splev( np.linspace( 0, 1, int(1e6)), tck,der = 0)
    centers = np.stack((znew,ynew,xnew),axis=1)
    return centers


def sphericalize(subvol,sigma_default=200/65/2.355):
    sigma_random = np.random.uniform(sigma_default*0.1,sigma_default*1.1)
    subvol = gaussian_filter(subvol, sigma=sigma_random)
    return subvol

def getKeyPt(centers,volsize):
    hum_of_centers = centers.shape[0]
    keypt_index = np.random.randint(0,hum_of_centers)
    keypt = centers[keypt_index]
    key_dist = np.sqrt((keypt[0]-volsize[0]/2)**2+(keypt[1]-volsize[1]/2)**2+(keypt[2]-volsize[2]/2)**2)
    key_distlimit = np.sqrt((volsize[0]/2)**2+(volsize[1]/2)**2+(volsize[2]/2)**2)/2
    while key_dist>key_distlimit:
        keypt_index = np.random.randint(0,hum_of_centers)
        keypt = centers[keypt_index]
        key_dist = np.sqrt((keypt[0]-volsize[0]/2)**2+(keypt[1]-volsize[1]/2)**2+(keypt[2]-volsize[2]/2)**2)
    return keypt

def genRandSpline(volsize,
               global_num=20,
               global_z_sparsity = 2.5):
    global_center_range_factor = 2.0/3.0
    global_center_range = (80,
                           volsize[1]*global_center_range_factor,
                           volsize[2]*global_center_range_factor)
    '''Here changes the density of the global centers'''
    centers = getSplineCoords(global_center_range,global_num,z_sparsity=global_z_sparsity)
    centers[:,0] = centers[:,0] + volsize[0]/2
    centers[:,1] = centers[:,1] + volsize[1]/2
    centers[:,2] = centers[:,2] + volsize[2]/2
    centers = np.around(centers).astype(int)
    print("Global center size:", centers.shape)
    
    total_points = centers
    vol = cp.zeros(volsize)
    
    vol[total_points[:,0],total_points[:,1],total_points[:,2]] = 1
    vol = sphericalize(vol)
    vol = cp.array(vol*60000).astype(cp.uint16).get()
    return vol



if __name__=="__main__":
    from os import system,mkdir
    from os.path import exists
    system('cls')

    destfolder = "Z:\\Xuanwen\\DLFLFM\\dldatasets\\RawDatasets\\syn_gtc_333FOVcluster\\"
    if not exists(destfolder):
        mkdir(destfolder)

    vol_shape = (155,1024,1024)
    # test diverseVol function
    for ii in range(120,201):
        volout = genRandSpline(vol_shape,
                            global_num=6,
                            global_z_sparsity = 1)
        filename = "wf_"+str(ii).zfill(4)+".tif"
        imwrite(destfolder+filename, volout)
        print(">>> saved:", filename)

    # #######################################
    # test z sparse-filled function
    # #######################################
    # z_sparse = cp.array([0,1,2,3,4,5,6,7,8,9])
    # print("z_sparse shape:", z_sparse.shape[0])
    # num_centers = 30
    # z_filled = random_z_fill(z_sparse,num_centers)
    # print(z_filled)