import glob
import time
import os
from io import BytesIO
import argparse
import boto3
from botocore.client import Config
import numpy as np
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
import math
import tifffile as tf
import SimpleITK as sitk
from util import tqdm_joblib, chunks, imgResample, upload_file_to_s3, S3Url, s3_object_exists


config = Config(connect_timeout=5, retries={'max_attempts': 5})

def correct_bias_field(img, mask=None, scale=1.0, niters=[50, 50, 50, 50]):
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
    stats = sitk.StatisticsImageFilter()
    stats.Execute(img)
    std = math.sqrt(stats.GetVariance())
    img_rescaled = sitk.Cast(img, sitk.sitkFloat32) + 0.1*std

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

    img_bc = sitk.Cast(img, sitk.sitkFloat32) * sitk.Cast(bias, sitk.sitkFloat32)
    return img_bc,bias


def sum_tiles(files):
    raw_tile = np.squeeze(tf.imread(files[0])).T
    running_sum = np.zeros(raw_tile.shape, dtype='float')

    for f in files:
        running_sum += np.squeeze(tf.imread(f)).T

    return running_sum


def correct_tile(raw_tile_path, outdir, bias=None):
    # overwrite existing tile
    out_path = raw_tile_path
    raw_tile = np.squeeze(tf.imread(raw_tile_path)).T

    if bias is None:
        tf.imwrite(out_path, data=raw_tile.astype('uint16'), compress=3, append=False)

    else:
        # rescale corrected tile to be uint16
        # for Terastitcher
        corrected_tile = np.around(raw_tile * bias)
        # clip values above uint16.max and below 0
        corrected_tile = np.clip(corrected_tile, 0, np.iinfo(np.uint16).max)
        # corrected_tile = (corrected_tile/(2**12 - 1)) * np.iinfo('uint16').max
        tf.imwrite(out_path, data=corrected_tile.astype('uint16'), compress=3, append=False)


def correct_tiles(tiles, outdir, bias):
    for tile in tiles:
        correct_tile(
            tile,
            outdir,
            bias
        )


def correct_raw_data(
    raw_data_path,
    channel,
    subsample_factor=5,
    log_s3_path=None
):

    total_n_jobs = cpu_count()
    # overwrite existing raw data with corrected data
    outdir = raw_data_path

    # get list of all tiles to correct for  given channel
    all_files = np.sort(glob.glob(f'{raw_data_path}/*/*.tiff'))
    total_files = len(all_files)

    bias_path = f'{outdir}/CHN0{channel}_bias.tiff'
    if os.path.exists(bias_path):
        bias = tf.imread(bias_path)

    else:
        # subsample tiles
        files_cb = all_files[::subsample_factor]
        num_files = len(files_cb)

        # compute running sums in parallel
        sums = Parallel(total_n_jobs, verbose=10)(delayed(sum_tiles)(f) for f in chunks(files_cb,math.ceil(num_files//(total_n_jobs))+1))
        sums = [i[:,:,None] for i in sums]
        sum_tile = np.squeeze(np.sum(np.concatenate(sums,axis=2),axis=2))/num_files
        sum_tile = sitk.GetImageFromArray(sum_tile)

        # get the bias correction tile using N4ITK
        bias = sitk.GetArrayFromImage(correct_bias_field(sum_tile,scale=1.0)[-1])

        # save bias tile to local directory
        tf.imsave(bias_path, bias)

    # save bias tile to S3
    if log_s3_path:
        s3 = boto3.resource('s3')
        img = Image.fromarray(bias)
        fp = BytesIO()
        img.save(fp, format='TIFF')
        # reset pointer to beginning  of file
        fp.seek(0)
        log_s3_url = S3Url(log_s3_path.strip('/'))
        bias_path = f'{log_s3_url.key}/CHN0{channel}_bias.tiff'
        s3.Object(log_s3_url.bucket, bias_path).upload_fileobj(fp)

    # correct all the files and save them
    files_per_proc = math.ceil(total_files/total_n_jobs)+1
    work = chunks(all_files, files_per_proc)
    with tqdm_joblib(tqdm(desc="Correcting tiles", total=total_n_jobs)) as progress_bar:
        Parallel(n_jobs=total_n_jobs, verbose=10)(
            delayed(correct_tiles)(
                files, 
                outdir,
                bias
            ) 
            for files in work
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', help='Path to raw data in COLM format.', type=str)
    parser.add_argument('--log_s3_path', help='S3 path where bias correction will live.', type=str)
    parser.add_argument('--channel', help='Channel number to process. accepted values are 0, 1, or 2', type=str)
    # parser.add_argument('--outdir', help='Path to output directory to store corrected tiles. VW0 directory will  be saved here. Default: ~/', default='/home/ubuntu/' ,type=str)
    parser.add_argument('--subsample_factor', help='Factor to subsample the tiles by to compute the bias. Default is subsample by 5 which means every 5th tile  will be used.', type=int, default=5)

    args = parser.parse_args()

    correct_raw_data(
        args.raw_data_path,
        args.channel,
        args.subsample_factor,
        args.log_s3_path
    )

