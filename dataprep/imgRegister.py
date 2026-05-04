import numpy as np
import SimpleITK as sitk


def tensor2ndarray(nparray):
    datatype = type(nparray)
    if datatype == np.ndarray:
        return nparray.astype(np.uint16)
    else:
        return nparray.float().numpy().astype(np.uint16)

def imshowfalsecolor(img1, img2, k=1):
    img1 = img1/img1.max()
    img2 = img2/img2.max()
    img = img1.unsqueeze(-1).repeat(1,1,3)
    img[...,1] = img2*k
    img = img/img.max()*255*2
    img = img.numpy().astype(np.uint8)
    return img

def command_iteration(method):
    print(f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}")
    
def imRegister(fixed, moving, showinfo=False):
    movingmax = moving.max().float().cpu().numpy()
    fixed = sitk.GetImageFromArray(fixed.float().cpu().numpy())
    moving = sitk.GetImageFromArray(moving.float().cpu().numpy())
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, 0.001, 300)
    R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)
    # if showinfo:
    #     R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)
    if showinfo:
        print("-------")
        print(outTx)
        # print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
        # print(f" Iteration: {R.GetOptimizerIteration()}")
        # print(f" Metric value: {R.GetMetricValue()}")

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)
    out = resampler.Execute(moving)
    out = sitk.GetArrayFromImage(out)
    out = out/out.max()*movingmax
    return out, outTx

def imRegisterShift(fixed,moving,outTx):
    movingmax = moving.max().float().cpu().numpy()
    fixed = sitk.GetImageFromArray(fixed.float().cpu().numpy())
    moving = sitk.GetImageFromArray(moving.float().cpu().numpy())
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)
    out = resampler.Execute(moving)
    out = sitk.GetArrayFromImage(out)
    out = out/out.max()*movingmax
    return out
