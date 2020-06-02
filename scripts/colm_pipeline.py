from . import correct_bias_whole_brain, correct_raw_data, create_precomputed_volume, generate_stitching_commands


def colm_pipeline():
    # pull raw data from S3, bias correct, and save to local directory
    correct_raw_data()
    
    # stitch raw data to create slices
    generate_stitching_commands()

    # downsample and upload stitched data to S3
    create_precomputed_volume()
