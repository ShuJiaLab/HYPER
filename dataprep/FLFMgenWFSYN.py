from glob import glob
from imageio.v3 import imread, imwrite

import cupy as cp

def destfunc(x,y,z,sigmaX=1,sigmaY=1,sigmaZ=1):
    return cp.exp(((-x**2)/(2*sigmaX**2))**3)*\
           cp.exp(((-y**2)/(2*sigmaY**2))**3)*\
           cp.exp(((-z**2)/(2*sigmaZ**2))**3)

def rejection_sampling(iter=1000,xylimit=[-512,512],zlimit=[-77,77],
                       cell_sigmaX=300,cell_sigmaY=300,cell_sigmaZ=64,maxval=1):
    samples = []
    # xlimit = [-windowsize/2,windowsize/2]
    for i in range(iter):
        x = cp.random.uniform(xylimit[0],xylimit[1])
        y = cp.random.uniform(xylimit[0],xylimit[1])
        z = cp.random.uniform(zlimit[0],zlimit[1])
        w = cp.random.uniform(0, maxval)

        while w > destfunc(x,y,z,cell_sigmaX,cell_sigmaY,cell_sigmaZ):
            x = cp.random.uniform(xylimit[0],xylimit[1])
            y = cp.random.uniform(xylimit[0],xylimit[1])
            z = cp.random.uniform(zlimit[0],zlimit[1])
            w = cp.random.uniform(0, maxval)
        samples.append([cp.around(z),cp.around(y),cp.around(x)])

    return cp.array(samples)

def genRandPSF(volsize,density):
    # x,y,z = np.indices(volsize)
    vol = cp.zeros(volsize).astype(cp.uint16)
    total_N = int(volsize[0]*volsize[1]*volsize[2])
    total_index = int(total_N*density)
    total_point = cp.around(cp.random.rand(1,total_index)*total_N).astype(int)
    print("total_index:", total_index, "max of total_point:", cp.max(total_point))
    total_indices = cp.unravel_index(total_point,volsize)
    # print("total_indices:", total_indices)
    vol[total_indices] = 60000
    return vol

def genRandPSFnorm(volsize,density):
    # x,y,z = np.indices(volsize)
    vol = cp.zeros(volsize)
    total_N = int(volsize[0]*volsize[1]*volsize[2])
    total_index = cp.around(cp.random.uniform(int(total_N*density*0.8),
                                              int(total_N*density*1.2))).get().astype(int)
    # x_point = rejection_sampling(iter=total_index,
    #                              xlimit=[-volsize[2]/2,volsize[2]/2],
    #                              cell_sigma=cp.random.uniform(10, 30)/0.065,
    #                              maxval=1) + volsize[2]/2
    # y_point = rejection_sampling(iter=total_index,
    #                              xlimit=[-volsize[1]/2,volsize[1]/2],
    #                              cell_sigma=cp.random.uniform(10, 30)/0.065,
    #                              maxval=1) + volsize[1]/2
    # z_point = rejection_sampling(iter=int(total_index*1.0/2),
    #                              xlimit=[-volsize[0]/2,volsize[0]/2],
    #                              cell_sigma=cp.random.uniform(8,12)/0.065,
    #                              maxval=1) + volsize[0]/2
    # z_point = cp.concatenate((z_point,z_point))
    # cp.random.shuffle(z_point)
    # print('shape of x_point:',x_point.shape)
    # print('shape of y_point:',y_point.shape)
    # print('shape of z_point:',z_point.shape)
    total_point = rejection_sampling(iter=total_index,
                                     xylimit=[-volsize[2]/2,volsize[2]/2],
                                     zlimit=[-volsize[0]/2,volsize[0]/2],
                                     cell_sigmaX=cp.random.uniform(10, 30)/0.065,
                                     cell_sigmaY=cp.random.uniform(10, 30)/0.065,
                                     cell_sigmaZ=cp.random.uniform(8,12)/0.065)
    print('shape of total_point:',total_point.shape)
    total_indices = (cp.array(total_point[:,0].T+volsize[0]/2).astype(int).get(),
                     cp.array(total_point[:,1].T+volsize[1]/2).astype(int).get(),
                     cp.array(total_point[:,2].T+volsize[2]/2).astype(int).get())
    # total_indices = (z_point.astype(int),y_point.astype(int),x_point.astype(int))
    # print('shape of total_indices:', cp.shape(total_indices))
    # total_point = cp.around(cp.random.rand(1,total_index)*total_N).astype(int)
    # print("total_index:", total_index, "max of total_point:", cp.max(total_point))
    # total_indices = cp.unravel_index(total_point,volsize)
    # print("total_indices:", total_indices)
    print('shape of vol:', vol.shape)
    vol[total_indices] = 60000
    vol = cp.around(vol * (0.5+0.5*cp.random.rand(volsize[0],volsize[1],volsize[2]))).astype(cp.uint16)
    return vol

if __name__=="__main__":
    from os import system,mkdir
    from os.path import exists
    system('cls')

    ## load data
    # volfolder = "L:\\dldatasets\\RawDatasets\\syn_gtc\\"
    # vollist = glob(volfolder+"*.tif")
    # vol = cp.asarray(imread(vollist[1]))
    # vol = vol[:,0:-1,0:-1]
    # print(">>> vol shape:", vol.shape)

    destfolder = "Z:\\Xuanwen\\DLFLFM\\dldatasets\\RawDatasets\\syn_gtc_norm10\\"
    if not exists(destfolder):
        mkdir(destfolder)

    vol_shape = (155,1024,1024)
    # test diverseVol function
    for ii in range(202):
        volout = genRandPSFnorm(vol_shape,5e-6).get()
        filename = "wf_"+str(ii).zfill(4)+".tif"
        imwrite(destfolder+filename, volout)
        print(">>> saved:", filename)
