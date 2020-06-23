import argparse
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
from cloudvolume import CloudVolume
import tinybrain
from joblib import Parallel, delayed

from util import imgResample, tqdm_joblib

def get_bias_field(img, mask=None, scale=1.0, niters=[50, 50, 50, 50]):
    """Correct bias field in image using the N4ITK algorithm (http://bit.ly/2oFwAun)

    Parameters:
    ----------
    img : {SimpleITK.SimpleITK.Image}
        Input image with bias field.
    mask : {SimpleITK.SimpleITK.Image}, optional
        If used, the bias field will only be corrected within the mask. (the default is None, which results in the whole image being corrected.)
    scale : {float}, optional
        Scale at which to compute the bias correction. (the default is 0.25, which results in bias correction computed on an image downsampled to 1/4 of it's original size)
    niters : {list}, optional
        Number of iterations per resolution. Each additional entry in the list adds an additional resolution at which the bias is estimated. (the default is [50, 50, 50, 50] which results in 50 iterations per resolution at 4 resolutions)

    Returns
    -------
    SimpleITK.SimpleITK.Image
        Bias-corrected image that has the same size and spacing as the input image.
    """

    # do in case image has 0 intensities
    # add a small constant that depends on
    # distribution of intensities in the image
    minmaxfilter = sitk.MinimumMaximumImageFilter()
    minmaxfilter.Execute(img)
    minval = minmaxfilter.GetMinimum()
    img_rescaled = sitk.Cast(img, sitk.sitkFloat32) - minval + 1.0

    spacing = np.array(img_rescaled.GetSpacing())/scale
    img_ds = imgResample(img_rescaled, spacing=spacing)
    img_ds = sitk.Cast(img_ds, sitk.sitkFloat32)

    # Calculate bias
    if mask is None:
        mask = sitk.Image(img_ds.GetSize(), sitk.sitkUInt8)+1
        mask.CopyInformation(img_ds)
    else:
        if type(mask) is not sitk.SimpleITK.Image:
            mask_sitk = sitk.GetImageFromArray(mask)
            mask_sitk.CopyInformation(img)
            mask = mask_sitk
        mask = imgResample(mask, spacing=spacing)

    img_ds_bc = sitk.N4BiasFieldCorrection(img_ds, mask, 0.001, niters)
    bias_ds = img_ds_bc / sitk.Cast(img_ds, img_ds_bc.GetPixelID())

    # Upsample bias
    bias = imgResample(bias_ds, spacing=img.GetSpacing(), size=img.GetSize())

    return bias



def process_slice(bias_slice,z,data_orig_path,data_bc_path):
    data_vol = CloudVolume(data_orig_path,parallel=False,progress=False,fill_missing=True)
    data_vol_bc = CloudVolume(data_bc_path,parallel=False,progress=False,fill_missing=True)
    data_vols_bc = [CloudVolume(data_bc_path,mip=i,parallel=False) for i in range(len(data_vol_bc.scales))]
    # convert spcing rom nm to um
    new_spacing = np.array(data_vol.scales[0]['resolution'][:2])/1000
    bias_upsampled_sitk = imgResample(bias_slice,new_spacing,size=data_vol.scales[0]['size'][:2])
    bias_upsampled = sitk.GetArrayFromImage(bias_upsampled_sitk)
    data_native = np.squeeze(data_vol[:,:,z]).T
    data_corrected = data_native * bias_upsampled
    img_pyramid = tinybrain.downsample_with_averaging(data_corrected.T[:,:,None], factor=(2,2,1), num_mips=len(data_vol_bc.scales)-1)
    data_vol_bc[:,:,z] = data_corrected.T.astype('uint16')[:,:,None]
    for i in range(len(data_vols_bc)-1):
        data_vols_bc[i+1][:,:,z] = img_pyramid[i].astype('uint16')


def correct_stitched_data(data_s3_path, out_s3_path, num_procs=12):
    # create vol
    vol = CloudVolume(data_s3_path)
    mip = 0
    for i in range(len(vol.scales)):
        # get low res image smaller than 10 um
        if vol.scales[i]['resolution'][0] < 10000:
            mip = i
    vol_ds = CloudVolume(data_s3_path,mip,parallel=True,fill_missing=True)

    # create new vol if it doesnt exist
    vol_bc = CloudVolume(out_s3_path,info=vol.info.copy())
    vol_bc.commit_info()

    # download image at low res
    data = sitk.GetImageFromArray(np.squeeze(vol_ds[:,:,:]).T)
    data.SetSpacing(np.array(vol_ds.scales[mip]['resolution'])/1000)

    bias = get_bias_field(data,scale=0.125)
    bias_slices = [bias[:,:,i] for i in range(bias.GetSize()[-1])]
    try: 
        with tqdm_joblib(tqdm(desc=f"Uploading bias corrected data...", total=len(bias_slices))) as progress_bar:
                Parallel(num_procs, timeout=7200)(
                    delayed(process_slice)(
                        bias_slice,
                        z,
                        data_s3_path,
                        out_s3_path
                    ) for z,bias_slice in enumerate(bias_slices)
                )
    except:
        print("timed out on bias correcting slice. moving to next step.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Correct whole brain bias field in image at native resolution.')
    parser.add_argument('data_s3_path',help='full s3 path to data of interest as precomputed volume. must be of the form `s3://bucket-name/path/to/channel`')
    parser.add_argument('out_s3_path',help='S3 path to save output results')
    parser.add_argument('--num_procs',help='number of processes to use', default=15, type=int)
    args = parser.parse_args()

    correct_stitched_data(
        args.data_s3_path,
        args.out_s3_path,
        args.num_procs
    )
