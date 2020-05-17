import time
import os
from io import BytesIO
import argparse

import boto3
from botocore.client import Config
import numpy as np
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed
import joblib
import math
import pathlib

import tifffile as tf
import SimpleITK as sitk


from scipy.ndimage import uniform_filter


total_n_jobs = joblib.cpu_count()

config = Config(connect_timeout=5, retries={'max_attempts': 5})

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
    s3_client = session.client('s3', config=config)
    loc_prefixes = s3_client.list_objects_v2(Bucket=in_bucket_name,Prefix=prefix,Delimiter='CHN')['CommonPrefixes']
    loc_prefixes = [i['Prefix'] + f'0{channel}' for i in loc_prefixes]
    all_files = []
    for i in tqdm(loc_prefixes):
        all_files.extend([f['Key'] for f in get_all_s3_objects(s3_client,Bucket=in_bucket_name,Prefix=i)])
    return all_files

def chunks(l,n):
    for i in range(0, len(l),n):
        yield l[i:i + n]

def get_correction_tile(raw_tile,blur_radius_fraction):
    # get blur radius from fraction of total image shape
    blur_radius = int(raw_tile.shape[0]*blur_radius_fraction)
    # blurring with 5 iterations of box blur approximates
    # gaussian blur but is ~10x faster
    tile_blurred = uniform_filter(raw_tile,blur_radius)
    for i in range(4):
        tile_blurred = uniform_filter(tile_blurred,blur_radius)
    correction = 1/tile_blurred.astype('float')
    correction_tile *= (1.0/np.mean(correction_tile))
    return correction_tile


def correct_tile(s3, raw_tile_bucket, raw_tile_path, outdir, blur_radius_fraction):
    #s = time.time()
    out_path = get_out_path(raw_tile_path, outdir)
    raw_tile_obj = s3.Object(raw_tile_bucket, raw_tile_path)
    # try this unless you get endpoin None error
    # then wait 30 seconds and retry
    try:
        raw_tile = np.asarray(Image.open(BytesIO(raw_tile_obj.get()["Body"].read())))
    except e:
        print(f"Encountered Exception. Waiting 60 seconds to retry")
        time.sleep(30)
        s3 = boto3.resource('s3')
        raw_tile_obj = s3.Object(raw_tile_bucket, raw_tile_path)
        raw_tile = np.asarray(Image.open(BytesIO(raw_tile_obj.get()["Body"].read())))

    # rescale corrected tile to be uint16
    # for  Terastitcher
    correction = get_correction_tile(raw_tile,blur_radius_fraction)

    corrected_tile = correction * raw_tile
    # clip values above uint16.max and below 0
    tf.imwrite(out_path,data=corrected_tile.astype('uint16'), compress=3, append=False)
    #print(f"took {time.time() - s}")


def correct_tiles(tiles, raw_tile_bucket,outdir,blur_radius_fraction=0.125):
    # global pbar_correct_tiles
    session = boto3.Session()
    s3 = session.resource('s3')
    
    for tile in tiles:
        correct_tile(
            s3,
            raw_tile_bucket,
            tile,
            outdir,
            blur_radius_fraction
        )

def get_out_path(in_path, outdir):
    head,fname = os.path.split(in_path)
    head_tmp = head.split('/')
    head = f'{outdir}' + '/'.join(head_tmp[-2:])
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
    parser.add_argument('--in_bucket_path', help='Full path to S3 bucket where raw tiles live. Should be of the form s3://<bucket-name>/<path-to-VW0-folder>/', type=str)
    parser.add_argument('--bias_bucket_name', help='Name of S3 bucket where bias correction will live.', type=str)
    parser.add_argument('--channel', help='Channel number to process. accepted values are 0, 1, or 2', type=str)
    parser.add_argument('--experiment_name', help='Name of experiment used to name newly created AWS resources for this job.', type=str)
    parser.add_argument('--outdir', help='Path to output directory to store corrected tiles. VW0 directory will  be saved here. Default: ~/', default='/home/ubuntu/' ,type=str)
    parser.add_argument('--subsample_factor', help='Factor to subsample the tiles by to compute the bias. Default is subsample by 10 which means every 10th tile  will be used.', type=int, default=10)

    args = parser.parse_args()
    args.in_bucket_name = args.in_bucket_path.split('s3://')[-1].split('/')[0]
    args.in_path = '/'.join(args.in_bucket_path.split('s3://')[-1].split('/')[1:])

    s = time.time()
    # get list of all tiles to correct for  given channel
    all_files = get_list_of_files_to_process(args.in_bucket_name,args.in_path,args.channel)
    total_files = len(all_files)

    # correct all the files and save them

    print(f"correcting tiles using {joblib.cpu_count()} cpus")
    Parallel(n_jobs=-1,verbose=10)(delayed(correct_tiles)(files, args.in_bucket_name, args.outdir) for files in chunks(all_files,math.ceil(total_files/float(joblib.cpu_count()))+1))
    print(f"total time taken for creating bias tile AND correcting all tiles: {time.time() - s} s")

if __name__ == "__main__":
    main()
   
