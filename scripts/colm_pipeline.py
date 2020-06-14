from correct_raw_data import correct_raw_data
from create_precomputed_volume import create_precomputed_volume
from generate_stitching_commands import generate_stitching_commands
from correct_stitched_data import correct_stitched_data
#from . import correct_raw_data, create_precomputed_volume, generate_stitching_commands
from util import S3Url, upload_file_to_s3, download_file_from_s3, download_terastitcher_files, tqdm_joblib
import boto3
import subprocess
import shlex
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
import os
from joblib import Parallel, delayed


def colm_pipeline(
    input_s3_path,
    output_s3_path,
    channel_of_interest,
    autofluorescence_channel,
    raw_data_path,
    stitched_data_path,
    log_s3_path=None
):
    """
    input_s3_path: S3 path to raw COLM data. Should be of the form s3://<bucket>/<experiment>
    output_s3_path: S3 path to store precomputed volume. Precomputed volumes for each channel will be stored under this path. Should be of the form s3://<bucket>/<path_to_precomputed>
    channel_of_interest: Channel number to operate on. Should be a single integer.
    autofluorescence_channel: Autofluorescence channel number. Should be a single integer.
    raw_data_path: Local path where corrected raw data will be stored.
    stitched_data_path: Local path where stitched slices will be stored.
    log_s3_path: S3 path at which pipeline intermediates can be stored including bias correction tile and xml files from Terastitcher.

    """
    # get the metadata file paths specific for COLM
    input_s3_url = S3Url(input_s3_path.strip('/'))
    output_s3_url = S3Url(output_s3_path.strip('/'))

    # pull raw data from S3, bias correct, and save to local directory
    # save bias correction tile to log_s3_path
    vw0_path = f'{input_s3_url.url}/VW0/'
    correct_raw_data(
        vw0_path,
        channel_of_interest,
        autofluorescence_channel,
        raw_data_path,
        log_s3_path=log_s3_path
    )
    
    # # generate commands to stitch data using Terastitcher
    stitch_only = False if channel_of_interest == 0 else True
    if stitch_only and not log_s3_path:
        raise("If using previous stitching results, must specify log_s3_path")
    elif stitch_only:
        pass
        # download terastitcher files if they arent already on local storage
        # download_terastitcher_files(log_s3_path, raw_data_path)
        
    metadata, commands = generate_stitching_commands(
        stitched_data_path,
        raw_data_path,
        input_s3_url.bucket,
        input_s3_url.key,
        stitch_only
    )

    # run the Terastitcher commands
    for i in commands:
        print(i)
        subprocess.run(
            shlex.split(i)
        )
    
    # upload xml results to log_s3_path if not None
    # and if not stitch_only
    if log_s3_path and not stitch_only:
        log_s3_url = S3Url(log_s3_path.strip('/'))
        files_to_save = glob(f'{raw_data_path}/*.xml')
        for i in tqdm(files_to_save,desc='saving xml files to S3'):
            out_path = i.split('/')[-1]
            upload_file_to_s3(i, log_s3_url.bucket, f'{log_s3_url.key}/{out_path}')


    # downsample and upload stitched data to S3
    create_precomputed_volume(
        stitched_data_path,
        np.array(metadata['voxel_size']),
        output_s3_path
    )

    # correct whole brain bias
    # in order to not replicate data (higher S3 cost)
    # overwrite original precomputed volume with corrected data
    correct_stitched_data(
        output_s3_path,
        output_s3_path
    )

    # REGISTRATION
    # only after stitching channel 1
    if channel_of_interest == 1:
        matlab_registration_command = f'matlab -nodisplay -nosplash -nodesktop ~/CloudReg/registration/registration_script_mouse_GN.m'
        subprocess.run(
            shlex.split(matlab_registration_command)
        )

    if log_s3_path:
        pass





if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run COLM pipeline including bias correction, stitching, upoad to S3')
    parser.add_argument('input_s3_path', help='S3 path to input colm data. Should be of the form s3://<bucket>/<experiment>', type=str)
    parser.add_argument('output_s3_path', help='S3 path to store precomputed volume. Precomputed volumes for each channel will be stored under this path. Should be of the form s3://<bucket>/<path_to_precomputed>. The data will be saved at s3://<bucket>/<path_to_precomputed>/CHN0<channel>',  type=str)
    # parser.add_argument('channel_of_interest', help='Channel of interest in experiment',  type=int)
    parser.add_argument('num_channels', help='Number of channels in experiment',  type=int)
    parser.add_argument('autofluorescence_channel', help='Autofluorescence channel number.',  type=int)
    parser.add_argument('--raw_data_path', help='Local path where corrected raw data will be stored.',  type=str, default='/home/ubuntu/ssd1')
    parser.add_argument('--stitched_data_path', help='Local path where stitched slices will be stored.',  type=str, default='/home/ubuntu/ssd2')
    parser.add_argument('--log_s3_path', help='S3 path at which pipeline intermediates can be stored including bias correctin tile.',  type=str, default=None)

    args = parser.parse_args()

    # for all channels in experiment
    for i in range(args.num_channels):
        colm_pipeline(
            args.input_s3_path,
            args.output_s3_path,
            i,
            args.autofluorescence_channel,
            args.raw_data_path,
            args.stitched_data_path,
            args.log_s3_path
        )
        # delete all tiff files in raw_data_path
        files_to_remove = glob.glob(f'{args.raw_data_path}/*/*.tiff')
        with tqdm_joblib(tqdm(desc=f"Delete files from CHN0{i}", total=len(files_to_remove))) as progress_bar:
            Parallel(-1)(delayed(os.remove)(
                    f
                ) for f in files_to_remove
            )