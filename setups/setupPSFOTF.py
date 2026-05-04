import time
import gc
from torch.cuda import empty_cache

def setupPSFOTF(load_PSF_OTF, args, device, n_depths,recalcPSFcenters=False):
    if len(args.gpu_repro)>0:
        S = time.time()
        # Load PSF and compute OTF
        n_split = args.n_split
        OTF,psf_shape, psfIn = load_PSF_OTF(args.psf_file, args.output_shape, n_depths=n_depths,
                                    data_scale=args.data_scale,n_lenslets=9, device="cpu",
                                    lenslet_centers_file_out=args.lenslet_file,
                                    recalc_lenslet_centers=recalcPSFcenters)
        OTF = OTF.to(device)
        gc.collect()
        empty_cache()
        E = time.time()
        print("PSF OTF loading time: ",round(E - S,2),"s. PSF shape: ",psf_shape)
        gc.collect()
        empty_cache()
        return OTF, psf_shape, psfIn