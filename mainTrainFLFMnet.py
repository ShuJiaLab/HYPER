# 
import torch
import os
import sys
import time
import subprocess
import h5py

from torch.cuda.amp import autocast
import torchvision as tv

from datetime import datetime
from pytorch_msssim import SSIM, MS_SSIM
from nets.FiLMnet import setupNetwork
import utils.pytorch_shot_noise as pytorch_shot_noise
from dataprep.FLFMDataset import FLFMDataset
from utils.misc_utils import *
from utils.psf_shift_utils import hybrid_psf
from losses.lossfunctions import lossfunc
from rich.console import Console
from utils.notice import send_notice

from setups.setupParams import setupParams
from setups.setupDevices import setupDevices
from setups.setupPSFOTF import setupPSFOTF
from setups.setupDataset import setupDataset
from setups.setupWriter import setupWriter

XMLFILENAME = './trainingparameters/useWFsynLFdeep_20251213_16X.xml'

console = Console(color_system="truecolor",style=None)
os.system('cls' if os.name == 'nt' else 'clear')

args = setupParams(XMLFILENAME,default=False)
print("Parameters loaded successfully.")

device, device_repro,n_threads = setupDevices(args)
print(f"Devices set up: device={device}, device_repro={device_repro}, n_threads={n_threads}")
print(f"Checkpoint exists: {os.path.exists(args.checkpoint_FLFMnet)}")

#===============================================================================
# Get training #ID
training_id = "FLFMNet_train__" + datetime.now().strftime('%Y%m%d_%H%M%S') + "__" + \
                                  XMLFILENAME.split('/')[-1].split('.')[0] + "_" + args.prefix
save_folder = args.output_path + '/' + training_id

print("# Saving to: ", save_folder)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Save folder created: {save_folder}")

#===============================================================================
# Set up data shape and scale
subimage_shape = [256,256] # original elemental image size
subimage_shape = [round(subimage_shape[0]*args.data_scale[0]),round(subimage_shape[1]*args.data_scale[1])]
img_shape = [768,768] # original ground truth image size
img_shape = [round(img_shape[0]*args.data_scale[0]),round(img_shape[1]*args.data_scale[1])]
args.output_shape = img_shape + [args.n_depths]
OTF, psf_shape, PSF = setupPSFOTF(load_PSF_OTF, args, device, args.n_depths,recalcPSFcenters=True)

PSF_test = load_PSF(filename = r'I:\expdata16x\FLFPSF_xy768z128_px1800z3200nm.mat', n_depths = 128, data_scale = [1,1])
vol_shape = [psf_shape[0].item(), psf_shape[1].item()]
dummy_vol = torch.rand(1, args.n_depths, vol_shape[0], vol_shape[1], device=device)
_, test_otf = fft_conv_split(dummy_vol, PSF_test.float().to(device), 
                    psf_shape, n_split=16, B_precomputed=False, device=device)
with h5py.File('I:\expdata16x\psfslopestest.mat', 'r') as f:
    psf_slope_test = f['psf_params'][:]   # read all data into NumPy
print(psf_slope_test)    
psf_slope_test = torch.from_numpy(psf_slope_test).float().flatten()

#===============================================================================
# Setup displacement arrays for PSF shifting from arguments
print("Setting up PSF displacement arrays from arguments...")

# Get displacement arrays from arguments
if hasattr(args, 'corrected_Xc_file') and args.corrected_Xc_file:
    # Load from files specified in arguments
    try:
        import scipy.io
        displacement_data = scipy.io.loadmat(args.corrected_Xc_file)
        print(f"Loaded displacement arrays using scipy.io from {args.corrected_Xc_file}")
    except NotImplementedError:
        print(f"MATLAB v7.3 format detected, using h5py to load {args.corrected_Xc_file}")
        with h5py.File(args.corrected_Xc_file, 'r') as f:
            # h5py loads data in different format, need to transpose
            displacement_data = {}
            for key in ['corrected_Xc', 'corrected_Yc', 'Xc_center', 'Yc_center','psf_slopes']:
                if key in f:
                    # h5py loads arrays transposed compared to scipy.io
                    displacement_data[key] = f[key][()].T
                else:
                    print(f"Warning: {key} not found in file")
                    
    corrected_Xc = torch.from_numpy(displacement_data['corrected_Xc']).float()
    corrected_Yc = torch.from_numpy(displacement_data['corrected_Yc']).float()
    Xc_center = torch.from_numpy(displacement_data['Xc_center']).float()
    Yc_center = torch.from_numpy(displacement_data['Yc_center']).float()
    psf_slopes = torch.from_numpy(displacement_data['psf_slopes']).float()
    print(f"Displacement arrays loaded successfully. Shapes: Xc={corrected_Xc.shape}, Yc={corrected_Yc.shape}")
else:
    print("No displacement file provided.")

#===============================================================================
# Set up the datasets
console.rule('[bold red]# Loading dataset #')
dataset, dataset_test, data_loaders, stats = setupDataset(FLFMDataset,args,subimage_shape,img_shape,n_threads)
print("Dataset loaded successfully.")
send_notice('Dataset loaded')
#===============================================================================
# Create net
console.rule('[bold red]# Creating network #')
NETWORK_OBJ_set = setupNetwork(args, stats, device)
net,checkpoint_FLFMnet,optimizer,lr,scaler,lr_sched,params = NETWORK_OBJ_set.values2return
print("# Network created, trainable parameters: ", params)

total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

#===============================================================================
# Create summary writer to log stuff
sample_psf_input = torch.rand(1, 18)  # 18D parameter vector instead of full PSF stack
writer = setupWriter(args, save_folder, params, net,
                    {
                         'curr_img_stack':torch.rand(1, 1, img_shape[0], img_shape[1]).to(device),
                         'PSF':sample_psf_input.to(device)
                    },
                    XMLFILENAME)
#===============================================================================
# Create loss function and optimizer
ssim_module = SSIM(data_range=1, size_average=True, channel=dataset.n_lenslets).to(device_repro)
#---------------------------------------------------
ssimloss_module = None
if args.loss_type[:4] == 'ssim':
    ssimloss_module = SSIM(data_range=1, size_average=True, channel=dataset.n_depths).to(device)
elif args.loss_type[:6] == 'ms_ssim':
    ssimloss_module = MS_SSIM(data_range=1, size_average=True, channel=dataset.n_depths).to(device)

loss, loss_img = lossfunc(args.loss_type,args.use_img_loss)
print("# Loss function created")
#===============================================================================
if len(args.gpu_repro)>0:
    OTF_options =   {'OTF':OTF,
                    'psf_shape':psf_shape,
                    'dataset':dataset,
                    'n_split':args.n_split,
                    'loss_img':loss_img}
    net.OTF_options = OTF_options
#===============================================================================
start_epoch = 0
if checkpoint_FLFMnet is not None:    
    net.load_state_dict(checkpoint_FLFMnet['model_state_dict'], strict=False)
    # loading optimizer will overwrite LR in xml file
    # optimizer.load_state_dict(checkpoint_FLFMnet['optimizer_state_dict'])
    start_epoch = checkpoint_FLFMnet['epoch']-1
    # save_folder += '_C'
    print("# Loaded checkpoint from ", args.checkpoint_FLFMnet)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
print('# No previous checkpoint found, starting from scratch')
time.sleep(5)
#===============================================================================
os.system('start powershell -NoExit -command "tensorboard --logdir=' + save_folder + ' --port=6006"')
#===============================================================================
'''Train the network
    Loop over epochs'''
send_notice('Training loop starts now!')
console.rule('[bold red]Training loop starts here[/bold red]')
for epoch in range(start_epoch, args.max_epochs):
    for curr_train_stage in ['train','val','test']:
        # Grab current data_loader
        curr_loader = data_loaders[curr_train_stage]
        curr_loader_len = curr_loader.sampler.num_samples \
                          if curr_train_stage=='test' else len(curr_loader.batch_sampler.sampler.indices)

        if curr_train_stage=='train':
            net.train()
            # net.tempConv.eval()
            torch.set_grad_enabled(True)
        if curr_train_stage=='val' or curr_train_stage=='test':
            if epoch%args.eval_every!=0:
                continue
            net.eval()
            torch.set_grad_enabled(False)

        # Store loss
        mean_volume_loss = 0 
        max_grad = 0
        mean_psnr = 0
        mean_time = 0
        mean_repro = 0
        mean_repro_ssim = 0

        # ============================================
        # Training
        # curr_img_stack: light-field image stack, local_volumes: ground truth volumes
        for ix,(curr_img_stack, label) in enumerate(curr_loader):
            # Generate batch-specific PSF for network input
            # Get shifted PSF as conditional input to the network
            if curr_train_stage=='test':
                batch_psf_input = PSF_test
                psf_emb = psf_slope_test.unsqueeze(0)  # Shape: [1, 18]
            else:
                batch_psf_input = hybrid_psf(
                    PSF[0, ...], corrected_Xc, corrected_Yc, Xc_center, Yc_center, ix
            ).unsqueeze(0)      
                psf_emb = psf_slopes[ix].unsqueeze(0)  # Shape: [1, 18]         
 
            if curr_img_stack.float().sum()==0 or torch.isnan(curr_img_stack.float().max()):
                print(f"Skipping batch {ix} due to zero sum or NaN values in current image stack.")
                continue
            
            curr_img_stack = curr_img_stack.half().to(device)
            psf_emb = psf_emb.half().to(device)
            print(f"Batch {ix}, PSF input: {psf_emb}")
            
            # Prepare PSF input for the network (conditional input)
            batch_size = curr_img_stack.size(0)
            batch_psf_input = batch_psf_input.half().to(device)          
            curr_img_stack = F.relu(curr_img_stack).detach()
            
            if args.add_noise==1 and curr_train_stage!='test':
                curr_max = curr_img_stack.max()
                curr_min = curr_img_stack.min()
                # Update new signal power
                signal_power = (args.signal_power_min + (args.signal_power_max-args.signal_power_min) * torch.rand(1)).item()
                curr_img_stack = signal_power/(curr_max-curr_min) * (curr_img_stack-curr_min)
                # Add noise
                curr_img_stack = pytorch_shot_noise.add_camera_noise(curr_img_stack)
                print('# Added noise signal_power: ', signal_power)
                curr_img_stack = curr_img_stack.float().to(device)
                print('# Added noise, curr_img_stack shape: ', curr_img_stack.shape)

            # Images are already normalized from mainCreateDataset.py
            curr_img_stack = normalize_type(curr_img_stack, stats['norm_type'], 
                                            stats['mean_imgs'], stats['std_images'], 
                                            stats['max_images'])
                                 
            # ================== start recording time ==================
            start.record()

            if curr_train_stage=='train':
                net.zero_grad()
                optimizer.zero_grad()
            # 
            with autocast():
                # ======================================================                             
                networkinput = {'curr_img_stack':curr_img_stack, 'PSF':psf_emb}
                prediction = net(networkinput)
                intermediate_result = dataset.extract_views(curr_img_stack, prediction,
                                                    dataset.lenslet_coords, 
                                                    dataset.subimage_shape,
                                                    dataset.data_scale,isreg=False)[:,0,...]
                                           
                diffY = (img_shape[0] - prediction.size()[2])
                diffX = (img_shape[1] - prediction.size()[3])
                prediction_padded = F.pad(prediction.float(), (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
               
                # Get batch size and initialize reprojection views
                out_type = prediction.type()
                batch_size = prediction.shape[0]
                reprojection_views = torch.zeros_like(intermediate_result)

                # Create dummy volume for OTF computation - match the expected input shape for fft_conv_split
                vol_shape = [psf_shape[0].item(), psf_shape[1].item()]
                dummy_vol = torch.rand(1, args.n_depths, vol_shape[0], vol_shape[1], device=device)
                
                # Compute OTF from shifted PSF using the existing fft_conv_split function
                _, batch_otf = fft_conv_split(dummy_vol, batch_psf_input[0, ...].unsqueeze(0).float(), 
                                    psf_shape, n_split=16, B_precomputed=False, device=device)
                                 
                # Use the batch-specific OTF for all samples in this batch
                # All samples in the batch use the same PSF/OTF
                for nSample in range(batch_size):
                    # Perform reprojection for this sample using batch OTF
                    reprojection = fft_conv_split(
                        prediction_padded[nSample,...].unsqueeze(0), 
                        batch_otf, 
                        psf_shape, 
                        n_split=16, 
                        B_precomputed=True, 
                        device=device
                    )
                    
                    # Extract views for this sample
                    reprojection_views[nSample,...] = dataset.extract_views(
                        reprojection, 
                        prediction[nSample,...].unsqueeze(0),
                        dataset.lenslet_coords, 
                        dataset.subimage_shape,
                        dataset.data_scale,
                        isreg=False
                    )[0,0,...]
                #======================================================
                volume_loss = loss(intermediate_result.float().to(device), reprojection_views.float().to(device), prediction, args.loss_z_smooth_lambda)
                print(f"Volume loss: {volume_loss.item()}")
                                                
                if curr_train_stage=='test' and len(args.gpu_repro)>0:
                    with torch.no_grad():
                        diffY = (img_shape[0] - prediction.size()[2])
                        diffX = (img_shape[1] - prediction.size()[3])
                        prediction_padded = F.pad(prediction, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

                        # Process each sample in batch with its corresponding OTF (same as training)
                        batch_size = prediction.shape[0]
                        reproj_list = []
                        curr_views_list = []
                        reproj_loss_total = 0
                        
                        for nSample in range(batch_size):
                            # Use the same batch OTF for test phase
                            # Perform reprojection for this sample with batch-specific OTF
                            sample_reproj_loss, sample_reproj, sample_curr_views, _ = reprojection_loss(
                                intermediate_result[nSample:nSample+1],  # Single sample
                                prediction_padded[nSample:nSample+1].float(),  # Single sample
                                test_otf,  # Batch-specific OTF for test
                                psf_shape, 
                                dataset, 
                                n_split=16, 
                                device=device_repro
                            )
                        
                            reproj_list.append(sample_reproj)
                            curr_views_list.append(sample_curr_views)
                            reproj_loss_total += sample_reproj_loss.item()
                            
                        # Stack results back into batch format
                        reproj = torch.cat(reproj_list, dim=0)
                        curr_views = torch.cat(curr_views_list, dim=0)
                        reproj_loss_mean = reproj_loss_total / batch_size
                    mean_repro += reproj_loss_mean                                    
                    mean_repro_ssim += ssim_module((intermediate_result/intermediate_result.max()).to(device_repro).float(), 
                                                   (reproj/reproj.max()).float().to(device_repro)).cpu().item()
                
            mean_volume_loss += volume_loss.mean().detach().item()

            if curr_train_stage=='train':
                # print("Before backward pass")
                scaler.scale(volume_loss).backward()
                # print("After backward pass")
                scaler.step(optimizer)
                scaler.update()                   

            # Record training time
            end.record()
            torch.cuda.synchronize()
            end_time = start.elapsed_time(end)
            mean_time += end_time

            # detach tensors
            prediction = prediction.detach().cpu().float()
            curr_img_stack = curr_img_stack.detach()

            if ix % 10 == 0:
                torch.cuda.empty_cache()
                
            # Normalize back
            # curr_img_stack, local_volumes = normalize_type(curr_img_stack, local_volumes, args.norm_type, mean_imgs, std_images, mean_vols, std_vols, max_images, max_volumes, inverse=True)
            # _, prediction = normalize_type(curr_img_stack, prediction, args.norm_type, mean_imgs, std_images, mean_vols, std_vols, max_images, max_volumes, inverse=True)

            if torch.isinf(torch.tensor(mean_volume_loss)):
                print('inf')
        
        mean_volume_loss /= curr_loader_len
        # mean_psnr = 20 * torch.log10(stats['max_vols'] / torch.sqrt(torch.tensor(mean_volume_loss))) #/= curr_loader_len
        mean_time /= curr_loader_len
        mean_repro /= curr_loader_len
        mean_repro_ssim /= curr_loader_len

        # Update learning rate
        if curr_train_stage=='val':
            lr_sched.step(mean_volume_loss)
            lr = optimizer.param_groups[0]['lr']

        if epoch % 5 == 0:
            writer.add_image('max_prediction_'+curr_train_stage, 
                            tv.utils.make_grid(volume_2_projections(prediction.permute(0,2,3,1).unsqueeze(1))[0,...], 
                            normalize=True, scale_each=True), epoch)
            writer.add_image('sum_prediction_'+curr_train_stage, 
                             tv.utils.make_grid(volume_2_projections(prediction.permute(0,2,3,1).unsqueeze(1), 
                             proj_type=torch.sum)[0,...], normalize=True, scale_each=True), epoch)
            writer.add_image('input_image_'+curr_train_stage,
                             curr_img_stack[0, 0].unsqueeze(0),  # shape [1, H, W]
                             epoch)
            
            if curr_train_stage=='test' and len(args.gpu_repro)>0:
                repro_grid = tv.utils.make_grid(reproj[0,...].sum(0).float().unsqueeze(0).cpu().data.detach(), 
                             normalize=True, scale_each=False)
                writer.add_image('reproj_'+curr_train_stage, repro_grid, epoch)
                # repro_grid = tv.utils.make_grid(curr_img_sparse[0,...].sum(0).float().unsqueeze(0).cpu().data.detach(), normalize=True, scale_each=False)
                repro_grid = tv.utils.make_grid(curr_views[0,...].sum(0).float().unsqueeze(0).cpu().data.detach(), 
                             normalize=True, scale_each=False)
                writer.add_image('reproj_GT_'+curr_train_stage, repro_grid, epoch)
                repro_grid = tv.utils.make_grid((curr_views-reproj)[0,4,...].abs().float().unsqueeze(0).cpu().data.detach(), 
                             normalize=True, scale_each=False)
                writer.add_image('reproj_error_'+curr_train_stage, repro_grid, epoch)
                writer.add_scalar('reproj/ssim/'+curr_train_stage, mean_repro_ssim, epoch)
                writer.add_scalar('reproj/Loss/'+curr_train_stage, mean_repro, epoch)

            writer.add_scalar('Loss/'+curr_train_stage, mean_volume_loss, epoch)
            # writer.add_scalar('psnr/'+curr_train_stage, mean_psnr, epoch)
            writer.add_scalar('times/'+curr_train_stage, mean_time, epoch)
            writer.add_scalar('lr/'+curr_train_stage, lr, epoch)
        
        if curr_train_stage=='train':
            console.print('[white]'+str(epoch) + ' ' + curr_train_stage + " loss: " + str(mean_volume_loss) + " time: " + str(round(end_time,5)) + "ms.")
        elif curr_train_stage=='val':
            console.print('[yellow]'+str(epoch) + ' ' + curr_train_stage + " loss: " + str(mean_volume_loss) + " time: " + str(round(end_time,5)) + "ms.")
            # send_notice('Epoch Number '+str(epoch) + ' at ' + curr_train_stage + " stage, loss is" + str(round(mean_volume_loss,5)))
        elif curr_train_stage=='test':
            console.print('[green]'+str(epoch) + ' ' + curr_train_stage + " loss: " + str(mean_volume_loss) + " time: " + str(round(end_time,5)) + "ms.")

        if os.path.isfile('./'+'exit_file.txt'):
            torch.cuda.empty_cache()
            sys.exit(0)

        if epoch%50==0 and epoch!=0:
            if curr_train_stage=='val':
                send_notice('Epoch '+str(epoch) + ' at ' + curr_train_stage + ' stage, '+\
                            'loss reaches' + str(round(mean_volume_loss,5)))
            torch.save({
            'epoch': epoch,
            'args' : args,
            'args_SLNet' : 'No argsSLNet',
            'statistics' : stats,
            'model_state_dict': net_get_params(net).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict' : scaler.state_dict(),
            'loss': mean_volume_loss},
            save_folder + '/model_'+str(epoch))

send_notice('Training loop has completed!')