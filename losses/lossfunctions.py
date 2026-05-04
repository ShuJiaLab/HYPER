import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch

def ssimlossfunc(x,y,ssim_module= None):
    if ssim_module == None:
        ssim_module = SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    ssim_loss = 1 - ssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    return ssim_loss

def msssimlossfunc(x,y,msssim_module= None):
    if msssim_module == None:
        msssim_module = MS_SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    msssim_loss = 1 - msssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    return msssim_loss

def ssiml1lossfunc(x,y,ssim_module= None):
    if ssim_module == None:
        ssim_module = SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    ssim_loss = 1 - ssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    l1_loss = nn.L1Loss()(x,y)
    return ssim_loss + l1_loss

def ssiml2lossfunc(x,y,ssim_module= None):
    if ssim_module == None:
        ssim_module = SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    ssim_loss = 1 - ssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    l2_loss = nn.MSELoss()(x,y)
    return ssim_loss + l2_loss

def ssimsmoothl1lossfunc(x,y,ssim_module= None):
    # The Smooth L1 Loss is a loss function that combines the benefits of L1 loss (absolute error) and L2 loss (squared error). 
    # It is less sensitive to outliers compared to L2 loss and smoother than L1 loss, making it a good choice for regression tasks where robustness to outliers is important.
    if ssim_module == None:
        ssim_module = SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    ssim_loss = 1 - ssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    smoothl1_loss = nn.SmoothL1Loss()(x,y)
    return ssim_loss + smoothl1_loss

def msssiml1lossfunc(x,y,msssim_module= None):
    if msssim_module == None:
        msssim_module = MS_SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    msssim_loss = 1 - msssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    l1_loss = nn.L1Loss()(x,y)
    return msssim_loss + l1_loss

def msssiml2lossfunc(x,y,msssim_module= None):
    if msssim_module == None:
        msssim_module = MS_SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    msssim_loss = 1 - msssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    l2_loss = nn.MSELoss()(x,y)
    return msssim_loss + l2_loss

def msssimsmoothl1lossfunc(x,y,msssim_module= None):
    if msssim_module == None:
        msssim_module = MS_SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    msssim_loss = 1 - msssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    smoothl1_loss = nn.SmoothL1Loss()(x,y)
    return msssim_loss + smoothl1_loss

def ssiml1l2lossfunc(x,y,ssim_module):
    if ssim_module == None:
        ssim_module = SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    ssim_loss = 1 - ssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    l1_loss = nn.L1Loss()(x,y)
    l2_loss = nn.MSELoss()(x,y)
    return ssim_loss + l1_loss + l2_loss

def msssiml1l2lossfunc(x,y,msssim_module):
    if msssim_module == None:
        msssim_module = MS_SSIM(data_range=1, size_average=True, channel=3).to('cuda:0')
    msssim_loss = 1 - msssim_module((x/x.max()).to('cuda:0').float(), (y/y.max()).to('cuda:0').float())
    l1_loss = nn.L1Loss()(x,y)
    l2_loss = nn.MSELoss()(x,y)
    return msssim_loss + l1_loss + l2_loss

def l1lossfunc(x,y,ssim_module= None):
    l1_loss = nn.L1Loss()(x,y)
    return l1_loss

def l2lossfunc(x,y,ssim_module= None):
    l2_loss = nn.MSELoss()(x,y)
    return l2_loss

# def z_smooth_loss(x, y, volume, λ=0.01):
#     l2_loss = nn.MSELoss()(x,y)
#     z_smooth_loss = torch.mean((volume[:, 1:, :, :] - volume[:, :-1, :, :]) ** 2)
#     total_loss = l2_loss + λ * z_smooth_loss  # Try λ = 0.01 ~ 0.1
#     return total_loss

def z_smooth_loss(x, y, volume, λ=0.01):
    l2_loss = nn.MSELoss()(x,y)
    zloss = torch.mean(torch.abs(2*volume[:,1:-1,...]-volume[:,:-2,:,:]-volume[:,2:,:,:]))
    total_loss = l2_loss + λ * zloss  
    print(f"L2 Loss: {l2_loss.item()}, Z Smoothness Loss: {zloss.item()}")
    return total_loss 

def l2lossfunc_norm(x, y):
    mse = torch.sum((x - y) ** 2)
    norm = torch.sum(x ** 2) 
    return mse / norm

def smoothl1lossfunc(x,y,ssim_module= None):
    smoothl1_loss = nn.SmoothL1Loss()(x,y)
    return smoothl1_loss

def compute_hessian(volume):
    """
    Compute the Hessian matrix for a 3D volume.
    Args:
        volume (torch.Tensor): Predicted volume of shape [batch, depth, height, width].
    Returns:
        hessian_norm (torch.Tensor): Frobenius norm of the Hessian matrix.
    """
    # Compute first-order gradients
    grad_x = torch.diff(volume, dim=3, append=volume[:, :, :, -1:])
    grad_y = torch.diff(volume, dim=2, append=volume[:, :, -1:, :])
    grad_z = torch.diff(volume, dim=1, append=volume[:, -1:, :, :])

    # Compute second-order gradients
    hess_xx = torch.diff(grad_x, dim=3, append=grad_x[:, :, :, -1:])
    hess_yy = torch.diff(grad_y, dim=2, append=grad_y[:, :, -1:, :])
    hess_zz = torch.diff(grad_z, dim=1, append=grad_z[:, -1:, :, :])

    # Frobenius norm of the Hessian matrix
    hessian_norm = (hess_xx**2 + hess_yy**2 + hess_zz**2).mean()
    return hessian_norm

def compute_hessian2(volume):
    """
    Compute the Hessian matrix for a 3D volume and return the weighted sum of L1 norms.
    Args:
        volume (torch.Tensor): Predicted volume of shape [batch, depth, height, width].
        lamda (float): Weight for the z-axis continuity and mixed gradients involving z.
    Returns:
        hessian_norm (torch.Tensor): Weighted sum of L1 norms of the Hessian components.
    """
    # Compute first-order gradients
    grad_x = torch.diff(volume, dim=3, append=volume[:, :, :, -1:])
    grad_y = torch.diff(volume, dim=2, append=volume[:, :, -1:, :])
    grad_z = torch.diff(volume, dim=1, append=volume[:, -1:, :, :])

    # Compute second-order gradients (diagonal terms)
    hess_xx = torch.diff(grad_x, dim=3, append=grad_x[:, :, :, -1:])
    hess_yy = torch.diff(grad_y, dim=2, append=grad_y[:, :, -1:, :])
    hess_zz = torch.diff(grad_z, dim=1, append=grad_z[:, -1:, :, :])

    # Compute mixed second-order gradients (off-diagonal terms)
    hess_xy = torch.diff(grad_x, dim=2, append=grad_x[:, :, -1:, :])
    hess_xz = torch.diff(grad_x, dim=1, append=grad_x[:, -1:, :, :])
    hess_yz = torch.diff(grad_y, dim=1, append=grad_y[:, -1:, :, :])

    # Compute the weighted sum of L1 norms
    hessian_norm = (
        hess_xx.pow(2).mean() +
        hess_yy.pow(2).mean() +
        hess_zz.pow(2).mean() +
        2 * hess_xy.pow(2).mean() +
        2 * hess_xz.pow(2).mean() +
        2 * hess_yz.pow(2).mean()
    ).sqrt()

    return hessian_norm

def l3lossfunc(x,y,z,alpha=1):
    """
    Combine MSE loss with Hessian-based continuity loss.
    Args:
        x (torch.Tensor): input light field image
        y (torch.Tensor): reprojected image from the predicted volume
        ssim_module (torch.nn.Module): SSIM module.
        alpha (float): Weight for the continuity loss.
    Returns:
        loss (torch.Tensor): Combined loss.
    """
    # MSE loss
    mse_loss = nn.MSELoss()(x,y)

    # Hessian-based continuity loss
    hessian_norm = compute_hessian2(z)

    # print(f"MSE: {mse_loss.item()}, Hessian: {hessian_norm.item()}")
    
    # Combine losses
    loss = mse_loss + alpha * hessian_norm
    return loss 

def ratiolossfunc(x,y,epsilon=1e-8):
    """
    Compute the loss as the sum of squares of (x - y) / y.
    Args:
        x (torch.Tensor): Input light field image.
        y (torch.Tensor): Projected image from the predicted volume.
        epsilon (float): Small value to avoid division by zero.
    Returns:
        loss (torch.Tensor): Scalar loss value.
    """
    ratio = (y - x) / (x + epsilon)
    loss = torch.sum(ratio ** 2) 
    return loss       

def ratiolossfunc2(x,y,epsilon=1e-8):
    """
    Compute the loss as the sum of squares of (x - y) / y.
    Args:
        x (torch.Tensor): sparse prediction (light field image)
        y (torch.Tensor): Projected image from the predicted volume.
        epsilon (float): Small value to avoid division by zero.
    Returns:
        loss (torch.Tensor): Scalar loss value.
    """
    ratio = (y - x) / (x + epsilon) 
    weight = (x + epsilon)
    loss = torch.sum(weight * ratio ** 2) / torch.sum(weight) / torch.sum(x ** 2)   
    return loss 

# ============================================================

def lossfunc(losstype,use_img_loss=1):   
    if losstype == 'l1':
        loss = l1lossfunc
    elif losstype == 'l2':
        loss = l2lossfunc
    elif losstype == 'l2_n':
        loss = l2lossfunc_norm
    elif losstype == 'z_smooth':
        loss = z_smooth_loss        
    elif losstype == 'l3':
        loss = l3lossfunc
    elif losstype == 'xy_ratio':
        loss = ratiolossfunc
    elif losstype == 'xy_ratio2':
        loss = ratiolossfunc2               
    elif losstype == 'smoothl1':
        loss = smoothl1lossfunc
    elif losstype == 'ssim':
        loss = ssimlossfunc
    elif losstype == 'msssim':
        loss = msssimlossfunc
    elif losstype == 'ssiml1':
        loss = ssiml1lossfunc
    elif losstype == 'ssiml2':
        loss = ssiml2lossfunc
    elif losstype == 'ssimsmoothl1':
        loss = ssimsmoothl1lossfunc
    elif losstype == 'msssiml1':
        loss = msssiml1lossfunc
    elif losstype == 'msssiml2':
        loss = msssiml2lossfunc
    elif losstype == 'msssimsmoothl1':
        loss = msssimsmoothl1lossfunc
    elif losstype == 'ssiml1l2':
        loss = ssiml1l2lossfunc
    elif losstype == 'msssiml1l2':
        loss = msssiml1l2lossfunc

    if use_img_loss>0:
        loss_img = loss
    else:
        loss_img = nn.MSELoss()
    return loss, loss_img

