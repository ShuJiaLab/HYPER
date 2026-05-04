# Performs inference and reconstruction using the FLFMnet model, with visualization and 
# saving capabilities.

import torch
import os
import time
import matplotlib.pyplot as plt
from torch.utils import data
from torch.utils.data.sampler import SequentialSampler
from torch.cuda.amp import autocast
import torch.nn as nn
import os
from datetime import datetime

from nets.FLFMnet import FLFMnet
from rich.console import Console
from dataprep.FLFMDataset import FLFMDataset
from utils.misc_utils import *
from imageio import volwrite

from setups.setupParams import setupParams
from setups.setupDevices import setupDevices
from setups.setupPSFOTF import setupPSFOTF
from nets.unet3d import UNet3d
from nets.FiLMnet_v4 import PSFConditionedRecon
from utils.psf_shift_utils import hybrid_psf

XMLFILENAME = './testparameters/useWFsynLFdeep_20251218_16X.xml'

console = Console(color_system="truecolor",style=None)
os.system('cls' if os.name == 'nt' else 'clear')

args = setupParams(XMLFILENAME,default=False)
device, _, n_threads = setupDevices(args)

# Load previous checkpoints
if len(args.checkpoint)>0:
    checkpoint_XLFMNet = torch.load(args.checkpoint, map_location=device, weights_only=False)
    args_deconv = checkpoint_XLFMNet['args']

# Get commit number 
training_id = "FLFMNet_Recon__" + datetime.now().strftime('%Y%m%d_%H%M%S') + "__" + args.prefix
save_folder = args.output_path + '/' + training_id

subimage_shape = [256,256] # original elemental image size
subimage_shape = [round(subimage_shape[0]*args.data_scale[0]),round(subimage_shape[1]*args.data_scale[1])]
img_shape = [768,768] # original ground truth image size
img_shape = [round(img_shape[0]*args.data_scale[0]),round(img_shape[1]*args.data_scale[1])]
# OTF, psf_shape = load_PSF_OTF(filename=args.psf_file, vol_size=[768,768,128], n_depths=128)
# PSF = load_PSF(filename = r'F:\DLFLFM\expdata16x\FLFPSF_xy768z128_px1800z1600nm.mat', n_depths = 128, data_scale = [1,1])
PSF = load_PSF(filename=args.psf_file, n_depths = 128, data_scale = [1,1])

# # Get displacement arrays from arguments
# if hasattr(args, 'corrected_Xc_file') and args.corrected_Xc_file:
#     # Load from files specified in arguments
#     try:
#         # First try scipy.io.loadmat for older MATLAB formats
#         import scipy.io
#         displacement_data = scipy.io.loadmat(args.corrected_Xc_file)
#         print(f"Loaded displacement arrays using scipy.io from {args.corrected_Xc_file}")
#     except NotImplementedError:
#         # If it's a MATLAB v7.3 file, use h5py
#         import h5py
#         print(f"MATLAB v7.3 format detected, using h5py to load {args.corrected_Xc_file}")
#         with h5py.File(args.corrected_Xc_file, 'r') as f:
#             # h5py loads data in different format, need to transpose
#             displacement_data = {}
#             for key in ['corrected_Xc', 'corrected_Yc', 'Xc_center', 'Yc_center']:
#                 if key in f:
#                     # h5py loads arrays transposed compared to scipy.io
#                     displacement_data[key] = f[key][()].T
#                 else:
#                     print(f"Warning: {key} not found in file")
                    
#     corrected_Xc = torch.from_numpy(displacement_data['corrected_Xc']).float()
#     corrected_Yc = torch.from_numpy(displacement_data['corrected_Yc']).float()
#     Xc_center = torch.from_numpy(displacement_data['Xc_center']).float()
#     Yc_center = torch.from_numpy(displacement_data['Yc_center']).float()

# batch_psf_input = hybrid_psf(
#     PSF[0, ...], corrected_Xc, corrected_Yc, Xc_center, Yc_center, 0
# ).unsqueeze(0)
      
imagerangemax = 20000
if (args.images_to_use_end-args.images_to_use_start+1)>imagerangemax:
    print(">>> Splitting reconstruction into multiple batches",imagerangemax,"images each")
    imgs2use = [range(20000,30000),range(30000,40000),range(40000,45000)]
else:
    imgs2use = [range(args.images_to_use_start-1,args.images_to_use_end)]

for imgsubrange in imgs2use:
    print("# Reconstructing images: ", imgsubrange[0]+1,"-",imgsubrange[-1]+1,"/")
    # Create dataloaders
    # dataset = FLFMDataset(args.data_folder, args.data_folder_vol,args.lenslet_file,subimage_shape, img_shape, args.psf_label,
    #                     args.data_scale,images_to_use=imgsubrange, n_depths_to_fill=64,load_vols=True)
    dataset = FLFMDataset(args.data_folder, args.data_folder_vol, args.lenslet_file, 
                      subimage_shape, img_shape,args.data_scale, 
                      images_to_use=args.images_to_use, n_depths_to_fill=128,load_vols=args.load_vols)

    dataset_size = len(dataset)
    test_indices = list(range(dataset_size))
    train_sampler = SequentialSampler(dataset)
    test_loader = data.DataLoader(dataset,sampler=train_sampler, num_workers=0, shuffle=False)
    print(f"Type of test_loader: {type(test_loader)}")
         
    # Get normalization values 
    # max_images,max_images_sparse,max_volumes = dataset.get_max() 
    stats = checkpoint_XLFMNet['statistics'] #dataset.get_statistics()

    # Create net
    # net = FLFMnet(dataset.n_lenslets, args_deconv.output_shape, dataset=dataset, 
    #             use_bias=args_deconv.use_bias, unet_settings=args_deconv.unet_settings).to(device)

    net = PSFConditionedRecon(psf_depth=args_deconv.n_depths, emb_dim=args_deconv.psf_emb_dim, 
                              base_ch=args_deconv.base_ch, z_channels=args_deconv.z_channels, dropout_rate=args_deconv.dropout_rate).to(device)
            
    # timers
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start_epoch = 0
    if len(args.checkpoint)>0:
        net.load_state_dict(checkpoint_XLFMNet['model_state_dict'], strict=False)

    if args.writeVolsToStack>0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    # Update noramlization stats for SLNet inside network
    net.stats = stats
    net = net.eval()

    plt.ion()
    fig, ax = plt.subplots()
    imshowHDL = ax.imshow(dataset.stacked_views[0,:,:])
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(5)

    with torch.no_grad():
        for ix,(curr_img_stack, label) in enumerate(test_loader):
            print(f"Label for batch {ix}")
            curr_img_stack = curr_img_stack.half().to(device)
            current_psf = PSF.half().to(device)
            # current_psf = batch_psf_input.half().to(device)
            # current_otf = OTF.to(device)
            # rayOptics_vol = rayOptics_vol.half().to(device)
            
            # curr_img_stack = curr_img_stack - args.dark_current
            curr_img_stack = F.relu(curr_img_stack).detach()
            # rayOptics_vol = F.relu(rayOptics_vol).detach()
            # curr_img_stack, rayOptics_vol = normalize_type(curr_img_stack, rayOptics_vol, stats['norm_type'], 
            #                                 stats['mean_imgs'], stats['std_images'], 
            #                                 stats['mean_vols'], stats['std_vols'], 
            #                                 stats['max_images'], stats['max_vols'])   
            curr_img_stack = normalize_type(curr_img_stack, stats['norm_type'], 
                                            stats['mean_imgs'], stats['std_images'], 
                                            stats['max_images'])   

            with autocast():
            # if True:
                start.record()
        
                # Run batch of predicted images in discriminator
                # networkinput = {'curr_img_stack':curr_img_stack,'OTF':current_otf}
                networkinput = {'curr_img_stack':curr_img_stack,'PSF':current_psf}
                prediction = net(networkinput)

                if not all([prediction.shape[i] == subimage_shape[i-2] for i in range(2,4)]):
                    diffY = (subimage_shape[0] - prediction.size()[2])
                    diffX = (subimage_shape[1] - prediction.size()[3])

                    prediction = F.pad(prediction, (diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2))

                pred_proj = prediction[0,...].max(dim=0).values
                pred_proj = pred_proj/pred_proj.max()*60000
                print(pred_proj.shape, pred_proj.min(), pred_proj.max())
                imshowHDL.set_data(pred_proj.cpu().numpy())
                fig.canvas.draw()
                fig.canvas.flush_events()
                # time.sleep(1)

                # Record training time
                end.record()
                torch.cuda.synchronize()
                end_time = start.elapsed_time(end)
                print(ix, "--" ,prediction[0,...].shape, " time:", round(end_time,2), "ms | Freq:", round(1000/end_time,2), "Hz")
                
                if args.writeVolsToStack>0:
                    stack_to_save = prediction[0,...].cpu().numpy().squeeze()
                    # stack_to_save = (stack_to_save - stack_to_save.min())/(stack_to_save.max()-stack_to_save.min())*60000
                    stack_to_save = (stack_to_save - stack_to_save.min())/(stack_to_save.max())*60000
                    stack_to_save = stack_to_save.astype(np.uint16)
                    print(ix, "--" ,stack_to_save.shape, " time:", round(end_time,2), "ms | Freq:", round(1000/end_time,2), "Hz")
                    volwrite(save_folder + '/FLFM_stack_'+ "%05d" % (ix+1+imgsubrange[0]) + '.tif', stack_to_save)


