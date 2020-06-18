import shlex
from download_data import download_data

    # REGISTRATION
    # only after stitching autofluorescence channel
    if channel_of_interest == autofluorescence_channel:
        base_path = f'{raw_data_path}'
        registration_prefix = f'{base_path}/registration/'
        target_name = f'{base_path}/autofluorescence_data.tif'

        # download downsampled autofluorescence channel
        voxel_size = download_data(output_s3_path, target_name)

        # initialize affine transformation for data
        

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