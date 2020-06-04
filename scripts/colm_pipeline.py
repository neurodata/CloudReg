from correct_raw_data import correct_raw_data
from create_precomputed_volume import create_precomputed_volume
from generate_stitching_commands import generate_stitching_commands
from correct_stitched_data import correct_stitched_data
#from . import correct_raw_data, create_precomputed_volume, generate_stitching_commands
from util import S3Url, upload_file_to_s3
import subprocess
import shlex
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse


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
    input_s3_path: 
    output_s3_path: 
    channel_of_interest:
    autofluorescence_channel:
    raw_data_path: 
    stitched_data_path: 
    log_s3_path:

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
    
    # generate commands to stitch data using Terastitcher
    stitch_only = False if channel_of_interest == 0 else True
    metadata, commands = generate_stitching_commands(
        stitched_data_path,
        raw_data_path,
        input_s3_url.bucket,
        input_s3_url.key,
        stitch_only
    )

    # run the Terastitcher commands
    for i in commands:
        subprocess.run(
            shlex.split(i),
        )
    
    # upload xml results to log_s3_path if not None
    if log_s3_path:
        log_s3_url = S3Url(log_s3_path.strip('/'))
        files_to_save = glob.glob(f'{raw_data_path}/*.xml')
        for i in tqdm(files_to_save,desc='saving xml files to S3'):
            out_path = i.split('/')[-1]
            upload_file_to_s3(i, log_s3_url.bucket, f'{log_s3_url.key}/{out_path}')


    # downsample and upload stitched data to S3
    create_precomputed_volume(
        stitched_data_path,
        np.array(metadata['voxel_size']),
        output_s3_path
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run COLM pipeline including bias correction, stitching, upoad to S3')
    parser.add_argument('input_s3_path', help='S3 path to input colm data. Should be of the form s3://<bucket>/<experiment>/VW0', type=str)
    parser.add_argument('output_s3_path', help='S3 path to store precomputed volume. Precomputed volumes for each channel will be stored under this path. Should be of the form s3://<bucket>/<path_to_precomputed>',  type=str)
    parser.add_argument('channel_of_interest', help='Channel number to operate on. Should be a single integer',  type=int)
    parser.add_argument('autofluorescence_channel', help='Autofluorescence channel number.',  type=int)
    parser.add_argument('--raw_data_path', help='',  type=str, default='/home/ubuntu/ssd1/VW0')
    parser.add_argument('--stitched_data_path', help='',  type=str, default='/home/ubuntu/ssd2/')
    parser.add_argument('--log_s3_path', help='S3 path at which pipeline intermediates can be stored including bias correctin tile.',  type=str, default=None)

    args = parser.parse_args()

    colm_pipeline(
        args.input_s3_path,
        args.output_s3_path,
        args.channel_of_interest,
        args.autofluorescence_channel,
        args.raw_data_path,
        args.stitched_data_path,
        args.log_s3_path
    )