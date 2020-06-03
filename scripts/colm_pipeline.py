from . import correct_raw_data, create_precomputed_volume, generate_stitching_commands
from util import S3Url, upload_file_to_s3
import subprocess
import shlex
import numpy as np
from glob import glob


def colm_pipeline(
    input_s3_path,
    output_s3_path,
    channel_of_interest,
    autofluorescence_channel,
    experiment_name,
    raw_data_path,
    stitched_data_path,
    log_s3_path=None
):
    """
    input_s3_path: 
    output_s3_path: 
    channel_of_interest:
    autofluorescence_channel:
    experiment_name:
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
            upload_file_to_s3(i, log_s3_url.bucket, f'{log_s3_url.key}/{i.split('/')[-1]}')


    # downsample and upload stitched data to S3
    create_precomputed_volume(
        stitched_data_path,
        np.array(metadata['voxel_size']),
        output_s3_path
    )
