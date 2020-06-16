from correct_raw_data import correct_raw_data
from create_precomputed_volume import create_precomputed_volume
from generate_stitching_commands import generate_stitching_commands
from correct_stitched_data import correct_stitched_data
from download_data import download_data
#from . import correct_raw_data, create_precomputed_volume, generate_stitching_commands
from util import S3Url, upload_file_to_s3, download_file_from_s3, download_terastitcher_files, tqdm_joblib, aws_cli
import boto3
import subprocess
import shlex
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
import os
from joblib import Parallel, delayed
import shutil


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
    # correct_raw_data(
    #     vw0_path,
    #     channel_of_interest,
    #     autofluorescence_channel,
    #     raw_data_path,
    #     log_s3_path=log_s3_path
    # )
    
    # # generate commands to stitch data using Terastitcher
    stitch_only = False if channel_of_interest == 0 else True
    if stitch_only and not log_s3_path:
        raise("If using previous stitching results, must specify log_s3_path")
    elif stitch_only:
        pass
    #     # download terastitcher files if they arent already on local storage
    #     # download_terastitcher_files(log_s3_path, raw_data_path)
        
    metadata, commands = generate_stitching_commands(
        stitched_data_path,
        raw_data_path,
        input_s3_url.bucket,
        input_s3_url.key,
        stitch_only
    )

    # run the Terastitcher commands
    # for i in commands:
    #     print(i)
    #     subprocess.run(
    #         shlex.split(i)
    #     )
    
    # # upload xml results to log_s3_path if not None
    # # and if not stitch_only
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
    # only after stitching autofluorescence channel
    if channel_of_interest == autofluorescence_channel:
        base_path = f'{raw_data_path}'
        registration_prefix = f'{base_path}/registration/'
        target_name = f'{base_path}/autofluorescence_data.tif'

        # download downsampled autofluorescence channel
        voxel_size = download_data(output_s3_path, target_name)

        # run registration
        matlab_registration_command = f'''
            matlab -nodisplay -nosplash -nodesktop -r \"base_path={base_path};target_name={target_name};prefix={registration_prefix};dxJ0={voxel_size};run(~/CloudReg/registration/registration_script_mouse_GN.m\")
        '''
        subprocess.run(
            shlex.split(matlab_registration_command)
        )

        if log_s3_path:
            # sync registration results to log_s3_path
            aws_cli(['s3', 'sync', registration_prefix, log_s3_path])



        

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
        if i == 0: continue
        output_s3_path = args.output_s3_path.strip('/')
        colm_pipeline(
            args.input_s3_path,
            f"{output_s3_path}/CHN0{i}",
            i,
            args.autofluorescence_channel,
            args.raw_data_path,
            args.stitched_data_path,
            args.log_s3_path
        )
        if i == 0:
            # delete all tiff files in raw_data_path
            directories_to_remove = glob(f'{args.raw_data_path}/LOC*')
            directories_to_remove.append(glob(f'{args.stitched_data_path}/RES*'))
            with tqdm_joblib(tqdm(desc=f"Delete files from CHN0{i}", total=len(directories_to_remove))) as progress_bar:
                Parallel(-1)(delayed(shutil.rmtree)(
                        f
                    ) for f in directories_to_remove
                )
            # make sure to delete mdata.bin from terastitcher
            os.remove(f'{raw_data_path}/mdata.bin')