import shlex
from download_data import download_data
from util import S3Url
from cloudvolume import CloudVolume
from scipy.spatial.transform import Rotation
import numpy as np
from util import get_reorientations, aws_cli 
from visualization import ara_average_data_link
import argparse
import subprocess
import os

atlas_orientation = "PIR"


def get_affine_matrix(
    translation,
    rotation,
    from_orientation,
    to_orientation,
    fixed_scale,
    s3_path,
    center=False
):
    """
    translation: x,y,z  translations respectively in microns
    center: if not None, should be an S3 path to compute the origin from
    """
    # since neuroglancer uses corner 0 coordinates we need to center the volume at it's center
    vol = CloudVolume(s3_path)
    # volume size in um
    vol_size = np.multiply(vol.scales[0]['size'], vol.scales[0]['resolution']) / 1e3
    # make affine matrix in homogenous coordinates
    affine = np.zeros((4,4))
    affine[-1,-1] = 1
    order, flips = get_reorientations(from_orientation, to_orientation)
    # reorder vol_size to match reorientation
    vol_size = vol_size[order]
    dim = affine.shape[0]
    # swap atlas axes to match target
    affine[range(len(order)),order] = 1
    # flip across appropriate dimensions
    affine[:3,:3] = np.diag(flips) @ affine[:3,:3]

    if center:
        # for each flip add the size of image in that dimension
        affine[:3,-1] += np.array([vol_size[i]  if flips[i] == -1 else 0 for i in range(len(flips))])
        # make image centered at the middle of the image
        # volume is now centered
        affine[:3,-1] -= vol_size/2

    # get rotation matrix
    if np.array(rotation).any():
        rotation_matrix = np.eye(4)
        rotation_matrix[:3,:3] = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
        # compose rotation with affine
        affine = rotation_matrix @ affine
    # add translation components
    # note: for neuroglancer affine, we scale the translations by voxel size
    # because neuroglancer expects translation in voxels
    affine[:3,-1] += translation
    
    # scale by fixed_scale
    affine = np.diag([fixed_scale]*dim) @ affine

    return affine


def register(
    input_s3_path,
    output_s3_path,
    log_s3_path,
    orientation,
    fixed_scale,
    translation,
    rotation
):

    # registration
    # get channel name
    print(input_s3_path)
    s3_url = S3Url(input_s3_path)
    channel = s3_url.key.split('/')[-1]
    exp = s3_url.key.split('/')[-2]

    # only after stitching autofluorescence channel
    base_path = os.path.expanduser('~/')
    registration_prefix = f'{base_path}/{exp}_{channel}_registration/'
    target_name = f'{base_path}/autofluorescence_data.tif'

    # download downsampled autofluorescence channel
    print("downloading data for registration...")
    voxel_size = download_data(input_s3_path, target_name)

    # initialize affine transformation for data
    atlas_res = 100
    atlas_s3_path = ara_average_data_link(atlas_res)
    initial_affine = get_affine_matrix(translation, rotation, atlas_orientation, orientation, fixed_scale, atlas_s3_path)


    # run registration
    affine_string = [', '.join(map(str,i)) for i in initial_affine]
    affine_string = '; '.join(affine_string)
    matlab_registration_command = f'''
        matlab -nodisplay -nosplash -nodesktop -r \"base_path={base_path};target_name={target_name};prefix={registration_prefix};dxJ0={voxel_size};fixed_scale={fixed_scale};initial_affine=[{affine_string}];run(~/CloudReg/registration/registration_script_mouse_GN.m\")
    '''
    subprocess.run(
        shlex.split(matlab_registration_command)
    )

    if log_s3_path:
        # sync registration results to log_s3_path
        aws_cli(['s3', 'sync', registration_prefix, log_s3_path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run COLM pipeline on remote EC2 instance with given input parameters')
    # data args
    parser.add_argument('-input_s3_path', help='S3 path to precomputed volume used to register the data', type=str)
    parser.add_argument('-log_s3_path', help='S3 path at which registration outputs are stored.',  type=str)
    parser.add_argument('--output_s3_path', help='S3 path to store atlas transformed to target as precomputed volume. Should be of the form s3://<bucket>/<path_to_precomputed>. Default is same as input s3_path with atlas_to_target as channel name',  type=str)

    # affine initialization args
    parser.add_argument('-orientation', help='3-letter orientation of data. i.e. LPS',  type=str)
    parser.add_argument('--scale', help='Fixed scale of data, uniform in all dimensions. Default is 1.',  type=float, default=1.0)
    parser.add_argument('--translation', help='Initial translation in x,y,z respectively in microns.',  nargs='+', type=float, default=[0,0,0])
    parser.add_argument('--rotation', help='Initial rotation in x,y,z respectively in degrees.',  nargs='+', type=float, default=[0,0,0])

    args = parser.parse_args()

    register(
        args.input_s3_path,
        args.output_s3_path,
        args.log_s3_path,
        args.orientation,
        args.scale,
        args.translation,
        args.rotation
    )
