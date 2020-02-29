try:
    import unzip_requirements
except ImportError:
    pass

import os
import time
import math
from io import BytesIO

import boto3
import numpy as np
from PIL import Image
import tifffile as tf
import SimpleITK as sitk

# brightness of tiles to fix
# hardcoding arbitrary  value for now, might be a heuristic that is better
BRIGHTNESS = 6000


def imgResample(img, spacing, size=[], useNearest=False, origin=None, outsideValue=0):
    """Resample image to certain spacing and size.

    Parameters:
    ----------
    img : {SimpleITK.SimpleITK.Image}
        Input 3D image.
    spacing : {list}
        List of length 3 indicating the voxel spacing as [x, y, z]
    size : {list}, optional
        List of length 3 indicating the number of voxels per dim [x, y, z] (the default is [], which will use compute the appropriate size based on the spacing.)
    useNearest : {bool}, optional
        If True use nearest neighbor interpolation. (the default is False, which will use linear interpolation.)
    origin : {list}, optional
        The location in physical space representing the [0,0,0] voxel in the input image. (the default is [0,0,0])
    outsideValue : {int}, optional
        value used to pad are outside image (the default is 0)

    Returns
    -------
    SimpleITK.SimpleITK.Image
        Resampled input image.
    """

    if origin is None: origin = [0]*3
    if len(spacing) != img.GetDimension():
        raise Exception(
            "len(spacing) != " + str(img.GetDimension()))

    # Set Size
    if size == []:
        inSpacing = img.GetSpacing()
        inSize = img.GetSize()
        size = [int(math.ceil(inSize[i] * (inSpacing[i] / spacing[i])))
                for i in range(img.GetDimension())]
    else:
        if len(size) != img.GetDimension():
            raise Exception(
                "len(size) != " + str(img.GetDimension()))

    # Resample input image
    interpolator = [sitk.sitkLinear, sitk.sitkNearestNeighbor][useNearest]
    identityTransform = sitk.Transform()
    identityDirection = list(
        sitk.AffineTransform(
            img.GetDimension()).GetMatrix())

    return sitk.Resample(
        img,
        size,
        identityTransform,
        interpolator,
        origin,
        spacing,
        img.GetDirection(),
        outsideValue)

def correct_bias_field(image, mask=None, scale=1.0, niters=[50, 50, 50, 50]):
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
    image_min = image.min()
    image2 = image - image_min + 1
    img = sitk.GetImageFromArray(image2)
    img_rescaled = sitk.Cast(img, sitk.sitkFloat32)

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
    bias = sitk.Cast(bias, sitk.sitkFloat32)
    bias_np = sitk.GetArrayFromImage(bias)

    img_bc = sitk.Cast(img, sitk.sitkFloat32) * bias
    img_bc_np = sitk.GetArrayFromImage(img_bc)
    img_bc_np += image_min - 1
    return img_bc_np,bias_np

def save_tile(s3, raw_tile, out_path, out_bucket):
    start_time = time.time()
    fp = BytesIO()
    raw_tile_bc = np.around(raw_tile)
    raw_tile_bc = np.clip(raw_tile_bc,0,np.iinfo(np.uint16).max)
    tf.imwrite(fp, data=raw_tile_bc.astype('uint16'), compress=1)
    # reset pointer to beginning  of file
    fp.seek(0)
    s3.Object(out_bucket, out_path).upload_fileobj(fp)
    print(f'SAVE - time: {time.time() - start_time} s path: {out_path}')

def adjust_brightness(raw_tile_bc):
    # correct_brightness if non-background tile
    curr_brightness = np.mean(raw_tile_bc)
    if np.std(raw_tile_bc) < 10 and curr_brightness < 100:
        pass
    else:
        brightness_correction_factor = BRIGHTNESS - np.mean(raw_tile_bc)
        raw_tile_bc += brightness_correction_factor
    return raw_tile_bc

def adjust_intensity(img,minval=1,maxval=1):
    img_r = (img - img.min()) / (img.max() - img.min())
    img_r += minval
    img_r *= maxval
    return img_r

def correct_tile(s3, raw_tile_bucket, raw_tile_path, out_path, out_bucket, bias=None):
    start_time = time.time()
    raw_tile_obj = s3.Object(raw_tile_bucket, raw_tile_path)
    raw_tile = np.asarray(Image.open(BytesIO(raw_tile_obj.get()["Body"].read())))
    if bias is not None:
        raw_tile_bc = raw_tile * bias
        print(f'MULTIPLY - time: {time.time() - start_time}, path: {raw_tile_path}')
        return raw_tile_bc
    else:
        _, bias = correct_bias_field(raw_tile, scale=0.25)
        # rescale bias to be between 1 and 2
        bias = adjust_intensity(bias)
        raw_tile_bc = bias * raw_tile
        print(f'CORRECT - time: {time.time() - start_time}, path: {raw_tile_path}')
        return raw_tile_bc,bias


def correct_tiles(s3, raw_tile_bucket, raw_tile_path, out_path, out_bucket, auto_channel, num_channels):
    channels = list(range(num_channels))
    idx_auto = channels.index(auto_channel)
    channels[idx_auto], channels[0] = channels[0], channels[idx_auto]
    bias = None
    for i in channels:
        if i == auto_channel:
            raw_tile_bc,bias = correct_tile(s3, raw_tile_bucket, raw_tile_path, out_path, out_bucket)
            save_tile(s3, raw_tile_bc, out_path, out_bucket)
        else:
            tile_path = raw_tile_path.replace(f'CHN0{auto_channel}',f'CHN0{i}')
            out_path2 = out_path.replace(f'CHN0{auto_channel}',f'CHN0{i}')
            raw_tile_bc = correct_tile(s3, raw_tile_bucket, tile_path, out_path2, out_bucket, bias=bias)
            save_tile(s3, raw_tile_bc, out_path2, out_bucket)



def lambda_handler(event, context):
    s3 = boto3.resource("s3")
    # read in bias tile
    attributes = event['Records'][0]['messageAttributes']
    print(attributes)
    for message in event['Records']:
        attributes = message['messageAttributes']
        correct_tiles(
            s3,
            attributes["RawTileBucket"]["stringValue"],
            attributes["RawTilePath"]["stringValue"],
            attributes["OutPath"]["stringValue"],
            attributes["OutBucket"]["stringValue"],
            int(attributes["AutoChannel"]["stringValue"]),
            int(attributes["NumChannels"]["stringValue"])
        )
