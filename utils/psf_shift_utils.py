import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import shift as imshift

def translate_tensor_2d(tensor, shift_x, shift_y):
    """
    Translate a 2D tensor using PyTorch's grid_sample (equivalent to MATLAB's imtranslate).
    
    Args:
        tensor: 2D tensor to translate [H, W]
        shift_x: Translation in x direction (positive = right)
        shift_y: Translation in y direction (positive = down)
    
    Returns:
        Translated tensor [H, W]
    """
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D")
    
    H, W = tensor.shape
    device = tensor.device
    
    # Add batch and channel dimensions for grid_sample: [1, 1, H, W]
    tensor_4d = tensor.unsqueeze(0).unsqueeze(0)
    
    # Create identity affine matrix
    theta = torch.tensor([[1, 0, -2*shift_x/W],  # Normalize shift_x to [-1,1] range Positive dx → image content moves right.
                         [0, 1, -2*shift_y/H]],  # Normalize shift_y to [-1,1] range Positive dy → image content moves down.
                        dtype=tensor.dtype, device=device).unsqueeze(0)
    
    # Generate sampling grid
    grid = F.affine_grid(theta, tensor_4d.size(), align_corners=False)
    
    # Sample using bilinear interpolation
    shifted = F.grid_sample(tensor_4d, grid, mode='bilinear', 
                           padding_mode='zeros', align_corners=False)
    
    # Remove batch and channel dimensions
    return shifted.squeeze(0).squeeze(0)

# # ======================psf 768*768*128 xy 1800 nm z 1600 nm=====================
# def hybrid_psf(FLF_SimPSF_rez, corrected_Xc, corrected_Yc, Xc_center, Yc_center, batch_idx):
#     """
#     Python port of:
#       FLF_HyRPSF_Rez(:,:, 129-idxfr) = imtranslate( FLF_SimPSF_rez(:,:, idxfr),
#                                                      [dX, dY] )

#     Parameters
#     ----------
#     FLF_SimPSF_rez : torch.Tensor
#         Simulated PSF stack to be shifted per depth & elemental.
#         Shape expected: (768, 768, 128)
#     corrected_Xc, corrected_Yc : np.ndarray
#         Centroid fits over z for each elemental and trial.
#         Shape: (3, 3, >=190, T)  -- must include indices 63..190 (1-based in MATLAB).
#     Xc_center, Yc_center : np.ndarray
#         Center values for each elemental and trial (used as reference).
#         Shape: (3, 3, T)

#     Returns
#     -------
#     FLF_HyRPSF: torch.Tensor
#         Hybrid PSF volume.
#         Shape: (128,768,768)
#     """

#     Z, H, W = FLF_SimPSF_rez.shape
#     assert (H, W, Z) == (768, 768, 128), "Expect FLF_SimPSF_rez to be (768,768,128)"

#     # Create output tensor with same device and dtype as input
#     FLF_HyRPSF = torch.zeros((Z, H, W), dtype=FLF_SimPSF_rez.dtype, device=FLF_SimPSF_rez.device)

#     tile_size = 256

#     for z_src in range(Z):  # 0..127 (MATLAB idxfr = 1..128)
#         # target z index (MATLAB: 129-idxfr → 1..128; Python 0-based):
#         z_dst = 127 - z_src

#         for ei_row in range(3):
#             r0 = tile_size * ei_row
#             r1 = r0 + tile_size

#             for ei_col in range(3):
#                 c0 = tile_size * ei_col
#                 c1 = c0 + tile_size

#                 # pull the 256x256 tile at this elemental from source PSF
#                 tile_src = FLF_SimPSF_rez[z_src, r0:r1, c0:c1]

#                 # MATLAB used (idxfr + 62) 1-based → Python 0-based is (z_src + 61)
#                 z_for_centroid = z_src + 61  # corresponds to MATLAB (idxfr+62)

#                 # compute translation (same signs as your MATLAB):
#                 # dX = corrected_Xc - Xc_center
#                 # dY = -corrected_Yc + Yc_center
#                 dX = (
#                     corrected_Xc[ei_row, ei_col, z_for_centroid, batch_idx]
#                     - Xc_center[ei_row, ei_col, batch_idx]
#                 )
#                 dY = (
#                     -corrected_Yc[ei_row, ei_col, z_for_centroid, batch_idx]
#                     + Yc_center[ei_row, ei_col, batch_idx]
#                 )

#                 # Use PyTorch-based translation instead of scipy
#                 tile_shifted = translate_tensor_2d(tile_src, dX, dY)

#                 # write into destination z-slice at the same elemental location
#                 FLF_HyRPSF[z_dst, r0:r1, c0:c1] = tile_shifted

#     return FLF_HyRPSF

# ======================psf 768*768*128 xy 1800 nm z 3200 nm=====================
def hybrid_psf(FLF_SimPSF_rez, corrected_Xc, corrected_Yc, Xc_center, Yc_center, batch_idx):
    """
    Memory-efficient version:
    1. Avoids double interpolation (128→251→128)
    2. Processes in Z-chunks to reduce peak memory
    3. Clears intermediate tensors explicitly
    """
    Z, H, W = FLF_SimPSF_rez.shape
    device = FLF_SimPSF_rez.device
    dtype = FLF_SimPSF_rez.dtype
    
    # Create output tensor
    FLF_HyRPSF = torch.zeros((Z, H, W), dtype=dtype, device=device)
    
    for z_src in range(Z):     
        _process_z_slice_direct(FLF_SimPSF_rez[z_src], FLF_HyRPSF, z_src,
                                corrected_Xc, corrected_Yc, Xc_center, Yc_center, batch_idx)
    
    return FLF_HyRPSF

def _process_z_slice_direct(src_slice, output_tensor, z_output, corrected_Xc, corrected_Yc, Xc_center, Yc_center, batch_idx):
    """
    Process a single Z-slice directly without storing intermediate full volumes.
    
    Args:
        z_centroid_index: Pre-computed index into the centroid arrays (0-250)
    """
    tile_size = 256   
    z_orig = np.arange(1, 252)          # 1..251
    z_new  = np.linspace(1, 251, 128)     # 128 points
                       
    adjust_ratio = np.ones(9) * 7/8   # initialize 9 elements with 7/8
    for k in [1, 3, 5, 7]:            # Python is 0-indexed → shift by -1
        adjust_ratio[k] = 7/9
    
    for ei_row in range(3):
        r0, r1 = tile_size * ei_row, tile_size * (ei_row + 1)
        for ei_col in range(3):
            cx_traj = corrected_Xc[ei_row, ei_col, :, batch_idx]   # [251]
            cy_traj = corrected_Yc[ei_row, ei_col, :, batch_idx]   # [251]   
            
            dX = np.interp(z_new, z_orig, cx_traj)  # [128]
            dY = np.interp(z_new, z_orig, cy_traj)  # [128]
                     
            c0, c1 = tile_size * ei_col, tile_size * (ei_col + 1)
            
            # Extract tile
            tile_src = src_slice[r0:r1, c0:c1]
            idx_z = 128 - 1 - z_output

            # linear index 1..9 in MATLAB → 0..8 in Python
            adj_idx = ei_row * 3 + ei_col

            dx = (dX[idx_z] - Xc_center[ei_row, ei_col, batch_idx]) * adjust_ratio[adj_idx]
            dy = (-dY[idx_z] + Yc_center[ei_row, ei_col, batch_idx]) * adjust_ratio[adj_idx]  
            
            # Apply shift and store
            tile_shifted = translate_tensor_2d(tile_src, dx, dy)
            output_tensor[z_output, r0:r1, c0:c1] = tile_shifted

# def _hybrid_psf_original(FLF_SimPSF_rez, corrected_Xc, corrected_Yc, Xc_center, Yc_center, batch_idx):
#     """
#     Original implementation (kept for comparison/fallback).
#     """
#     FLF_SimPSF_rez = F.interpolate(FLF_SimPSF_rez.unsqueeze(0).unsqueeze(0), size=(251,768,768), mode='trilinear', align_corners=False).squeeze(0).squeeze(0)
#     Z, H, W = FLF_SimPSF_rez.shape    
    
#     # Create output tensor with same device and dtype as input
#     FLF_HyRPSF = torch.zeros((Z, H, W), dtype=FLF_SimPSF_rez.dtype, device=FLF_SimPSF_rez.device)

#     tile_size = 256

#     for z_src in range(Z):  # 0..250 (MATLAB idxfr = 1..251)
#         for ei_row in range(3):
#             r0 = tile_size * ei_row
#             r1 = r0 + tile_size
#             for ei_col in range(3):
#                 c0 = tile_size * ei_col
#                 c1 = c0 + tile_size

#                 # pull the 256x256 tile at this elemental from source PSF
#                 tile_src = FLF_SimPSF_rez[z_src, r0:r1, c0:c1]

#                 # 0-250 -> 250-0
#                 z_for_centroid = 250 - z_src

#                 # compute translation (same signs as your MATLAB):
#                 # dX = corrected_Xc - Xc_center
#                 # dY = -corrected_Yc + Yc_center
#                 dX = (
#                     corrected_Xc[ei_row, ei_col, z_for_centroid, batch_idx]
#                     - Xc_center[ei_row, ei_col, batch_idx]
#                 )
#                 dY = (
#                     -corrected_Yc[ei_row, ei_col, z_for_centroid, batch_idx]
#                     + Yc_center[ei_row, ei_col, batch_idx]
#                 )

#                 # Use PyTorch-based translation instead of scipy
#                 tile_shifted = translate_tensor_2d(tile_src, dX, dY)

#                 # write into destination z-slice at the same elemental location
#                 FLF_HyRPSF[z_src, r0:r1, c0:c1] = tile_shifted

#     FLF_HyRPSF = F.interpolate(FLF_HyRPSF.unsqueeze(0).unsqueeze(0), size=(128,768,768), mode='trilinear', align_corners=False).squeeze(0).squeeze(0)

#     return FLF_HyRPSF