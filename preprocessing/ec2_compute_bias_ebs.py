import time
import os
from io import BytesIO
import argparse

import boto3
import numpy as np
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed
import joblib
import math
import pathlib

import tifffile as tf
import SimpleITK as sitk


total_n_jobs = joblib.cpu_count()

# taken from: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
# to track a multiprocessing job with a progress bar
class BatchCompletionCallBack(object):
    # Added code - start
    global total_n_jobs
    # Added code - end
    def __init__(self, dispatch_timestamp, batch_size, parallel):
        self.dispatch_timestamp = dispatch_timestamp
        self.batch_size = batch_size
        self.parallel = parallel

    def __call__(self, out):
        self.parallel.n_completed_tasks += self.batch_size
        this_batch_duration = time.time() - self.dispatch_timestamp

        self.parallel._backend.batch_completed(self.batch_size,
                                           this_batch_duration)
        self.parallel.print_progress()
        # Added code - start
        progress = self.parallel.n_completed_tasks / total_n_jobs
        time_remaining = (this_batch_duration / self.batch_size) * (total_n_jobs - self.parallel.n_completed_tasks)
        print( "\rProgress: [{0:50s}] {1:.1f}% est {2:1f}mins left".format('#' * int(progress * 50), progress*100, time_remaining/60) , end="", flush=True)

        if self.parallel.n_completed_tasks == total_n_jobs:
            print('\n')
        # Added code - end
        if self.parallel._original_iterator is not None:
            self.parallel.dispatch_next()

Parallel.BatchCompletionCallBack = BatchCompletionCallBack

def get_list_of_files_to_process(in_bucket_name, prefix, channel):
    session = boto3.Session()
    s3_client = session.client('s3')
    loc_prefixes = s3_client.list_objects_v2(Bucket=in_bucket_name,Prefix=prefix,Delimiter='CHN')['CommonPrefixes']
    loc_prefixes = [i['Prefix'] + f'0{channel}' for i in loc_prefixes]
    all_files = []
    for i in tqdm(loc_prefixes):
        all_files.extend([f['Key'] for f in get_all_s3_objects(s3_client,Bucket=in_bucket_name,Prefix=i)])
    return all_files

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

def chunks(l,n):
    for i in range(0, len(l),n):
        yield l[i:i + n]

def sum_tile(s3, running_sum, raw_tile_bucket, raw_tile_path):
    # start_time = time.time()
    raw_tile_obj = s3.Object(raw_tile_bucket, raw_tile_path)
    raw_tile = np.asarray(Image.open(BytesIO(raw_tile_obj.get()["Body"].read())))
    # print(f"PULL - time: {time.time() - start_time}, path: {raw_tile_path}")
    # start_time = time.time()
    # tf.imsave(out_path, data=(raw_tile * bias))
    running_sum += raw_tile
    # print(f"SUM - time: {time.time() - start_time} s path: {raw_tile_path}")

def sum_tiles(files, raw_tile_bucket):
    running_sum = np.zeros((1024,1024))
    session = boto3.Session()
    s3 = session.resource('s3')
    for f in files:
        sum_tile(
            s3,
            running_sum,
            raw_tile_bucket,
            f
        )
    return running_sum

def correct_tile(raw_tile_bucket, raw_tile_path, out_path, bias):
    s3 = boto3.Session().resource('s3')
    raw_tile_obj = s3.Object(raw_tile_bucket, raw_tile_path)
    raw_tile = np.asarray(Image.open(BytesIO(raw_tile_obj.get()["Body"].read())))
    # print(f'PULL - time: {time.time() - start_time}, path: {raw_tile_path}')
    # fp = BytesIO()
    # tf.imwrite(fp, data=(raw_tile * bias), compress=1)
    tf.imwrite(out_path,data=(raw_tile * bias), compress=1)
#    np.save(fp,tile)
    # reset pointer to beginning  of file
    # fp.seek(0)
    # s3.Object(out_bucket, out_path).upload_fileobj(fp)
    # print(f'SAVE - time: {time.time() - start_time} s path: {out_path}')

def get_out_path(in_path):
    head,fname = os.path.split(in_path)
    head_tmp = head.split('/')
    head = './' + '/'.join(head_tmp[-2:])
    idx = fname.find('.')
    fname_new = fname[:idx] + '_corrected.tiff'
    out_path = f'{head}/{fname_new}'
    os.makedirs(head, exist_ok=True)  # succeeds even if directory exists.
    return out_path

def get_all_s3_objects(s3, **base_kwargs):
    continuation_token = None
    while True:
        list_kwargs = dict(MaxKeys=1000, **base_kwargs)
        if continuation_token:
            list_kwargs['ContinuationToken'] = continuation_token
        response = s3.list_objects_v2(**list_kwargs)
        yield from response.get('Contents', [])
        if not response.get('IsTruncated'):  # At the end of the list?
            break
        continuation_token = response.get('NextContinuationToken')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_bucket_name', help='Name of input S3 bucket where raw tiles live.', type=str)
    parser.add_argument('--bias_bucket_name', help='Name of S3 bucket where bias correction tile lives.', type=str)
    parser.add_argument('--out_bucket_name', help='Name of S3 bucket to store the corrected tiles.', type=str)
    parser.add_argument('--in_path', help='Full path  to VW0 directory on S3 bucket.', type=str)
    parser.add_argument('--bias_path', help='Full path  to bias file for given channel.', type=str)
    parser.add_argument('--channel', help='Channel number to process. accepted values are 0, 1, or 2', type=str)
    parser.add_argument('--experiment_name', help='Name of experiment used to name newly created AWS resources for this job.', type=str)
    parser.add_argument('--subsample_factor', help='Factor to subsample the tiles by to compute the bias. Default is subsample by 10 which means every 10th tile  will be used.', type=int, default=10)

    args = parser.parse_args()

    s = time.time()
    # get list of all tiles to correct for  given channel
    all_files = get_list_of_files_to_process(args.in_bucket_name,args.in_path,args.channel)
    total_files = len(all_files)
    # subsample tiles
    # files_cb = all_files[::args.subsample_factor]
    # num_files  = len(files_cb)

    # sums = Parallel(total_n_jobs, verbose=10, prefer='threads')(delayed(sum_tiles)(f, args.in_bucket_name) for f in chunks(files_cb,num_files//(total_n_jobs)))
    # sums = [i[:,:,None] for i in sums]
    # sum_tile = np.squeeze(np.sum(np.concatenate(sums,axis=2),axis=2))/num_files
    # sum_tile = sitk.GetImageFromArray(sum_tile)

    # bias = sitk.GetArrayFromImage(correct_bias_field(sum_tile,scale=1.0)[-1])

    # # save bias tile to S3
    # s3 = boto3.resource('s3')
    # img = Image.fromarray(bias)
    # fp = BytesIO()
    # img.save(fp,format='TIFF')
    # # reset pointer to beginning  of file
    # fp.seek(0)
    # s3.Object(args.bias_bucket_name, args.bias_path).upload_fileobj(fp)
    # print(f"total time taken for creating bias tile: {time.time() - s} s")

    # get bias tile from S3
    s3 = boto3.resource('s3')
    bias_obj = s3.Object(
        args.bias_bucket_name, args.bias_path
    )
    bias = np.asarray(Image.open(BytesIO(bias_obj.get()["Body"].read())))

    # correct all the files and post them to S3
    # global pbar_correct_tiles
    # pbar_correct_tiles = tqdm(total=total_files,desc='correcting tiles...')
    out_paths = [get_out_path(i) for i in tqdm(all_files,desc='getting output paths')]
    Parallel(n_jobs=-1,prefer='threads')(delayed(correct_tile)(args.in_bucket_name, in_path, out_path, bias) for in_path,out_path in tqdm(zip(all_files,out_paths),total=total_files))
    # pbar_correct_tiles.close()
    print(f"total time taken for creating bias tile AND correcting all tiles: {time.time() - s} s")


    # OR 
    #  correct all the files and save them locally

    # now delete the files
    # for f in tqdm(all_files, desc='Deleting summed tiles...'):
    #     raw_tile_obj = s3.Object(args.in_bucket_name, f)
    #     raw_tile_obj.delete()

if __name__ == "__main__":
    main()
   
