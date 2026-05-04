import numpy as np
from glob import glob
from imageio.v3 import imread, imwrite
from clij2fft.richardson_lucy import richardson_lucy
from rich.progress import track
from os.path import basename

# if __name__ == '__main__':
'''Load PSF into GPU'''
# psfstack = loadWFPSF("Z:\\Xuanwen\\DLFLFM\\expdata100x\\Simu20231212Wv680\\PSFGAUint_20231215_Red_gly_10um.mat",-1)
# psfstack = psfstack.get()
psfstack = imread("Z:\\Xuanwen\\DLFLFM\\dldatasets\\train\\TrainingExpData\\PSFFLFint_20240428_Red_Gly_10um\\PSFGAUint_20231215_Red_gly_10um.tif")
psfstack = psfstack.astype(np.float32)

imstacklist = glob("Z:\\Xuanwen\\DLFLFM\\dldatasets\\RawDatasets\\syn_cfc_333FOVcluster_aug\\*.tif")

for i in track(range(5), description='Deconvolving images...'):
    # print('Deconvolving image ' + str(i) + ': ' + imstacklist[i] + '...')
    imstack = imread(imstacklist[i])
    filename = basename(imstacklist[i])
    imstack = np.asarray(imstack).astype(np.float32)
    imstackmaxv = np.max(imstack)
    # imstack = imstack/imstackmaxv
    imdeconv = richardson_lucy(imstack, psfstack, numiterations=10, regularizationfactor=0.005)
    imdeconv = imdeconv/np.max(imdeconv)*imstackmaxv
    imwrite("Z:\\Xuanwen\\DLFLFM\\dldatasets\\RawDatasets\\syn_cfc_333FOVclusterRLD_aug\\" + filename, 
            imdeconv.astype(np.uint16),extension='.tif')
    print('Image ' + str(i) + ' deconvolved!')
