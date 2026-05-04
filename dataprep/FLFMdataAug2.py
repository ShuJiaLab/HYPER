import os
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from scipy.ndimage import affine_transform
from pathlib import Path
import tifffile as tiff

def random_affine_3d_matrix(scale_range=(0.8, 1.2), shear_range=0, trans_range=(10, 10, 0)):
    # Create random rotation matrix
    angle_x = np.deg2rad(np.random.uniform(-45, 45))
    angle_y = np.deg2rad(np.random.uniform(-45, 45))
    # angle_z = np.deg2rad(np.random.uniform(-45, 45))
    angle_z = 0
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx

    # Scaling
    scale = np.random.uniform(scale_range[0], scale_range[1])
    S = np.eye(3) * scale

    # Shear (X-Y-Z)
    shear = np.deg2rad(np.random.uniform(-shear_range, shear_range, size=(3,)))
    Sh = np.eye(3)
    Sh[0, 1] = np.tan(shear[0])
    Sh[1, 2] = 0
    Sh[1, 2] = 0
    # Sh[1, 2] = np.tan(shear[1])
    # Sh[0, 2] = np.tan(shear[2])

    # Combine all
    A = R @ S @ Sh

    # Translation
    tx = np.random.uniform(-trans_range[0], trans_range[0])
    ty = np.random.uniform(-trans_range[1], trans_range[1])
    tz = np.random.uniform(-trans_range[2], trans_range[2])
    t = np.array([tx, ty, tz])

    return A, t

def apply_affine_3d(volume, A, t):
    """Apply affine transform using scipy with trilinear interpolation."""
    center = np.array(volume.shape) / 2
    offset = center - A @ center + t
    return affine_transform(volume, A, offset=offset, order=3, mode='constant', cval=0.0)

# Configuration
augment_factor = 20
input_folder = r"..\expdata16x\colonWF16x_512denoised_z256"
output_folder = r"..\expdata16x\colonWFaug16x_512denoised_Z256_10"
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Load all files
volume_files = sorted(Path(input_folder).glob("*.tif"))

for fileidx, filepath in enumerate(volume_files):
    filename = filepath.stem
    vol = tiff.imread(filepath).astype(np.float32)

    for ii in range(augment_factor):
        save_name = f"{filename}_{ii+1}.tif"
        save_path = os.path.join(output_folder, save_name)

        if ii == 0:
            vol_aug = vol
        else:
            A, t = random_affine_3d_matrix()
            vol_aug = apply_affine_3d(vol, A, t)

        tiff.imwrite(save_path, vol_aug, dtype=np.float32)
        print(f"[{fileidx+1}] ({ii+1}/{augment_factor}): Written {save_path}")