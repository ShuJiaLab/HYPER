import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter
# from scipy.ndimage import gaussian_filter
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
    # z_sparse = np.around(np.random.uniform(zlimit[0],zlimit[1],size = num_centers_z))
    z_loc = np.random.uniform((zlimit[0]+zlimit[1])/2,(zlimit[0]+zlimit[1])/2,size=1)
    z_sparse = np.around(np.random.normal(z_loc,(zlimit[1]-zlimit[0])/2,size = num_centers_z))
    z = random_z_fill(z_sparse,num_centers)
    centers = np.stack((z,y,x),axis=1)
    return centers

def getLocalClusterCenters(global_center_range,volsize,init_center,num_subcenters,subcenters_sigma,z_sparsity = 1):
    # This function returns the local cluster centers for the given initial
    # center and number of subcenters. The centers are returned as a list of
    # tuples. The centers are chosen to be normally spaced centered on the initial center.
    ylimit = [-global_center_range[1]/2,global_center_range[1]/2]
    xlimit = [-global_center_range[2]/2,global_center_range[2]/2]
    xylimit = [np.maximum(xlimit[0],ylimit[0]),np.minimum(xlimit[1],ylimit[1])]
    zlimit = [-global_center_range[0]/2,global_center_range[0]/2]
    subcenters = rejection_sampling(iter=num_subcenters,
                                    xylimit=xylimit,zlimit=zlimit,
                                    cell_sigmaX=subcenters_sigma[2],
                                    cell_sigmaY=subcenters_sigma[1],
                                    cell_sigmaZ=subcenters_sigma[0],maxval=1)
    num_centers_z = int(np.maximum(num_subcenters*1.0/z_sparsity,1.0))
    subcenters_z = subcenters[:,0]
    # print("num_centers_z size",num_centers_z,"subcenters_z size:", subcenters_z.shape)
    z_sparse = subcenters_z[np.random.randint(0,subcenters_z.shape[0],size=num_centers_z)]
    subcenters_z = random_z_fill(z_sparse,num_subcenters)
    subcenters[:,0] = subcenters_z
    subcenters = subcenters + init_center
    xylimit = [np.maximum(xylimit[0],0),np.minimum(xylimit[1],volsize[1]-1)]
    zlimit = [np.maximum(zlimit[0],0),np.minimum(zlimit[1],volsize[0]-1)]
    return subcenters

def getneighbors(init_centers,contdensity,contlevel):
    nbcenters = []
    for ii in range(0,init_centers.shape[0]):
        if np.random.rand() <= contdensity:
            nbcenter = init_centers[ii,:]
            contdim = np.random.randint(0,27)
            while contdim == 13:
                    contdim = np.random.randint(0,27)
            contz = contdim//9-1
            conty = (contdim%9)//3-1
            contx = contdim%3-1
            nbcenter = nbcenter + np.array([contz,conty,contx])
            nbcenter = [int(nbcenter[0]),int(nbcenter[1]),int(nbcenter[2])]
            nbcenters.append(nbcenter)
            smallvector0 = np.array([contz,conty,contx])

            for _ in range(1,np.random.randint(round(0.5*contlevel),round(1.5*contlevel))):
                isforward = False
                while not isforward:
                    contdim = np.random.randint(0,27)
                    while contdim == 13:
                        contdim = np.random.randint(0,27)
                    contz = contdim//9-1
                    conty = (contdim%9)//3-1
                    contx = contdim%3-1
                    smallvector = np.array([contz,conty,contx])
                    smallangle = np.dot(smallvector0,smallvector)/(np.linalg.norm(smallvector0)*np.linalg.norm(smallvector))
                    if smallangle >= 0.75:
                        isforward = True
                        nbcenter = nbcenter + np.array([contz,conty,contx])
                        nbcenter = [int(nbcenter[0]),int(nbcenter[1]),int(nbcenter[2])]
                        nbcenters.append(nbcenter)
                        smallvector0 = smallvector
    return nbcenters

def sphericalize(subvol,sigma_default=0.6):
    sigma_random = np.random.uniform(sigma_default*0.5,sigma_default*0.75)
    subvol = gaussian_filter(subvol, sigma=np.random.uniform(sigma_random*0.9,sigma_random*1.1))
    return subvol

def random_ellipsoid(center, radii, shape, irregularity=0.1, flatten_z_frac=0.1):
    zz, yy, xx = np.indices(shape)
    cx, cy, cz = center
    rx, ry, rz = radii
    rx *= np.random.uniform(1-irregularity, 1+irregularity)
    ry *= np.random.uniform(1-irregularity, 1+irregularity)
    rz *= np.random.uniform(1-irregularity, 1+irregularity)
    ellipsoid = (((xx-cx)/rx)**2 + ((yy-cy)/ry)**2 + ((zz-cz)/rz)**2) <= 1
    # Flatten top and bottom along z
    zmin = int(flatten_z_frac * shape[0])
    zmax = shape[0] - zmin
    ellipsoid[:zmin,:,:] = False
    ellipsoid[zmax:,:,:] = False
    return ellipsoid

def assignIntensity(vol,total_points,neighbor_density=1):
    volin = vol.get()
    volout = vol.get()
    for ii in range(0,total_points.shape[0]):
        pointz_start = np.maximum(total_points[ii,0]-neighbor_density,0)
        pointz_end = np.minimum(total_points[ii,0]+neighbor_density,volin.shape[0])
        pointy_start = np.maximum(total_points[ii,1]-neighbor_density,0)
        pointy_end = np.minimum(total_points[ii,1]+neighbor_density,volin.shape[1])
        pointx_start = np.maximum(total_points[ii,2]-neighbor_density,0)
        pointx_end = np.minimum(total_points[ii,2]+neighbor_density,volin.shape[2])
        weight = np.sum(volin[pointz_start:pointz_end,pointy_start:pointy_end,pointx_start:pointx_end])
        original_value = volin[total_points[ii,0],total_points[ii,1],total_points[ii,2]]
        new_value = original_value*1.0/(weight**1.0+1.0)
        volout[total_points[ii,0],total_points[ii,1],total_points[ii,2]] = new_value
        # print(">>> ",total_points[ii,:], original_value, new_value,original_value*1.0/(weight**2.0+1.0),
        #       "new_value:", volout[total_points[ii,0],total_points[ii,1],total_points[ii,2]])
    return cp.asarray(volout)

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

def genRandPSF(volsize,
               global_num=20,
               local_density = 0.25,
               cluster_num = 6,
               global_z_sparsity = 2.5,
               local_z_sparsity = 1,
               cluster_z_sparsity = 1,
               density=0.02):   
    """
    Generate a synthetic 3D volume with hierarchical clustering structure.
    
    Parameters:
    -----------
    volsize : tuple
        3D volume dimensions (z, y, x) in voxels
    global_num : int
        Number of global cluster centers (main anchor points)
    local_density : float
        Multiplier for number of local clusters around each global center
        Higher values = more local clusters per global center
    cluster_num : int
        Number of small clusters generated around each selected local center
    global_z_sparsity : float
        Controls z-distribution of global centers
        >1 = more centers share same z-levels (layered)
        =1 = each center can have unique z
    local_z_sparsity : float
        Controls z-distribution of local centers around each global center
    cluster_z_sparsity : float
        Controls z-distribution of small clusters around each local center
    density : float
        Fraction (0-1) of local centers that will get additional small clusters
        Higher values = more fine-scale detail
    
    Returns:
    --------
    vol : numpy.ndarray (uint16)
        Generated synthetic volume scaled to 0-40000 range
    total_points : numpy.ndarray
        All generated point coordinates [z, y, x]
    
    Process Overview:
    ----------------
    1. Generate global cluster centers distributed in volume
    2. For each global center, generate local clusters around it
    3. For subset of local centers, generate small clusters for fine detail
    4. Assign intensities based on neighborhood density
    5. Apply smoothing to simulate microscope blur
    6. Add realistic background clouds
    7. Normalize and convert to uint16
    """    
    
    global_center_range_factor = 0.9
    global_center_range = (volsize[0]*global_center_range_factor,
                           volsize[1]*global_center_range_factor,
                           volsize[2]*global_center_range_factor)
    '''Here changes the density of the global centers'''
    centers = getGlobalClusterCenters(global_center_range,global_num,z_sparsity=global_z_sparsity)
    centers[:,0] = centers[:,0] + volsize[0]/2
    centers[:,1] = centers[:,1] + volsize[1]/2
    centers[:,2] = centers[:,2] + volsize[2]/2
    centers = np.around(centers).astype(int)
    print("Global center size:", centers.shape)
    # print("Global centers:\n", centers)
    keypt = getKeyPt(centers,volsize)
    # print("keypt:", keypt)
    total_points = []
    vol = cp.zeros(volsize)
        
    # ##############################################################################
    for center in track(centers,description="2. Adding local centers..."):
        '''Here changes the density of the local centers'''
        ringrange = [np.sqrt((volsize[0]/2)**2+(volsize[1]/2)**2+(volsize[2]/2)**2)*1/3,
                     np.sqrt((volsize[0]/2)**2+(volsize[1]/2)**2+(volsize[2]/2)**2)*3/4]
        ringsize = np.random.uniform(ringrange[0],ringrange[1])
        key_dist = np.sqrt((keypt[0]-volsize[0]/2)**2+(keypt[1]-volsize[1]/2)**2+(keypt[2]-volsize[2]/2)**2)
        ringorient = [(volsize[0]-keypt[0])/key_dist, (volsize[1]-keypt[1])/key_dist, (volsize[2]-keypt[2])/key_dist]
        ringcenter = [keypt[0]+ringorient[0]*ringsize*np.random.uniform(0.8,1.2), 
                      keypt[1]+ringorient[1]*ringsize*np.random.uniform(0.8,1.2), 
                      keypt[2]+ringorient[2]*ringsize*np.random.uniform(0.8,1.2)]
        ringdist = np.sqrt(np.sqrt((center[0]-keypt[0])**2+(center[1]-keypt[1])**2+(center[2]-keypt[2])**2) * \
                           np.absolute(np.sqrt((center[0]-ringcenter[0])**2+(center[1]-ringcenter[1])**2+\
                                               (center[2]-ringcenter[2])**2)-ringsize))
        subclusterdensity = np.absolute(100.0 - ringdist/np.random.uniform(10,15.0))
        subclusterdensity = int(np.maximum(subclusterdensity*local_density,1.0))
        local_center_range = (global_center_range[0]*0.75,global_center_range[1],global_center_range[2])
        subcenters = getLocalClusterCenters(local_center_range,volsize,center,
                                            subclusterdensity,
                                            [10,30,30],
                                            z_sparsity=local_z_sparsity)
        total_points.extend(subcenters)
    total_points = np.array(total_points).astype(int)
    total_points_intmed = total_points
    print("total_points size w/ local centers:", total_points.shape)
    subsub_index = np.random.randint(0,total_points.shape[0],size=int(total_points.shape[0]*density))
    print("subsub_index size:", subsub_index.shape)
    # ##############################################################################
    subsubcenters_list = []
    for point in track(subsub_index,description="3. Adding small clusters..."):
        subsubcenters = getLocalClusterCenters(local_center_range,volsize,total_points[point],
                                               cluster_num,
                                               [5,3,3],
                                               z_sparsity=cluster_z_sparsity)
        subsubcenters = np.around(subsubcenters).astype(int)
        subsubcenters_list.append(subsubcenters)
    if len(subsubcenters_list) > 0:
        subsubcenters_all = np.concatenate(subsubcenters_list, axis=0)
        total_points = np.concatenate((total_points, subsubcenters_all), axis=0).astype(int)
    print("total_points size:", total_points.shape)
    # ##############################################################################
    # nbcenters = getneighbors(total_points_intmed,0.2,np.random.uniform(10,50))
    # total_points = np.concatenate((total_points,nbcenters),axis=0).astype(int)
    total_points = total_points[total_points[:,0]>=0]
    total_points = total_points[total_points[:,0]<volsize[0]]
    total_points = total_points[total_points[:,1]>=0]
    total_points = total_points[total_points[:,1]<volsize[1]]
    total_points = total_points[total_points[:,2]>=0]
    total_points = total_points[total_points[:,2]<volsize[2]]
    # ###############################################################################
    vol[total_points[:,0],total_points[:,1],total_points[:,2]] = cp.random.uniform(0.5,1,1)
    # print(">>> ",vol[total_points[:,0],total_points[:,1],total_points[:,2]])
    vol = assignIntensity(vol,total_points,neighbor_density=5)
    vol = np.clip(vol,0,1)
    # vol = sphericalize(vol,3)
    vol = sphericalize(vol,1)
    print("vol max min:", cp.max(vol), cp.min(vol))
    # clipvalue = valueClip(vol,total_points)
    # vol[vol>clipvalue] = clipvalue
    
    num_cloud_centers = np.random.randint(2, min(8, centers.shape[0]))  # 2-8 clouds, but not more than available centers
    cloud_indices = np.random.choice(centers.shape[0], num_cloud_centers, replace=False)   
    original_vol_max = cp.max(vol).get()
    for i in cloud_indices:
        # Mix of small and medium background features
        if np.random.rand() < 0.7:  # 70% chance of small clouds
            radii = [int(np.random.randint(2, 8)) for _ in range(3)]
        elif np.random.rand() < 0.9:  # 20% chance of medium clouds
            radii = [int(np.random.randint(8, 20)) for _ in range(3)]
        else:
            radii = [int(np.random.randint(20, 80)) for _ in range(3)]
            
        cloud = cp.asarray(random_ellipsoid(centers[i,:], radii, volsize, irregularity=0.1).astype(float))
        cloud = gaussian_filter(cloud, sigma=np.random.uniform(6, 15))
        cloud = cloud / (cp.max(cloud) if cp.max(cloud) > 0 else 1) * np.random.uniform(original_vol_max/1000, original_vol_max/100)
        vol = vol + cloud
            
    vol = (vol/cp.max(vol)*40000).get().astype(np.uint16)
    # vol = (vol/np.max(vol)*40000).astype(np.uint16)
    print("total_points max in z:", np.max(total_points[:,0]))
    return vol, total_points



# if __name__=="__main__":
#     from os import system,mkdir
#     from os.path import exists
#     system('cls')
#     import matplotlib.pyplot as plt

#     destfolder = "Z:\\Xuanwen\\DLFLFM\\dldatasets\\RawDatasets\\syn_gtc_333FOVcluster\\"
#     if not exists(destfolder):
#         mkdir(destfolder)

#     vol_shape = (155,1024,1024)
#     # test diverseVol function
#     for ii in range(0,201):
#         volout,_ = genRandPSF(vol_shape,
#                             global_num=60,
#                             local_density = 0.2,
#                             cluster_num = 10,
#                             global_z_sparsity = 1,
#                             local_z_sparsity = 1,
#                             cluster_z_sparsity = 1,
#                             density=0.01)
#         volout_xy = np.max(volout,axis=0)
#         # plt.imshow(volout_xy)
#         # plt.show()
#         filename = "wf_"+str(ii).zfill(4)+".tif"
#         imwrite(destfolder+filename, volout)
#         print(">>> saved:", filename)

    # #######################################
    # test z sparse-filled function
    # #######################################
    # z_sparse = cp.array([0,1,2,3,4,5,6,7,8,9])
    # print("z_sparse shape:", z_sparse.shape[0])
    # num_centers = 30
    # z_filled = random_z_fill(z_sparse,num_centers)
    # print(z_filled)