# local imports
from .util import aws_cli
from .visualization import create_viz_link_from_json, get_neuroglancer_json

import pathlib
import subprocess
import shlex
import requests as r
import numpy as np
import h5py
from cloudvolume import CloudVolume
from collections import defaultdict
import uuid
import argparse
from scipy.io import loadmat
from skimage.io import imsave
import json
import random

def transform_data(
    target_layer_source,
    atlas_viz_link,
    path_to_affine,
    path_to_velocity,
    # voxel size of velocity field
    velocity_voxel_size,
):
    # identify layer to downlooad
    for mip in range(5):
        target_vol = CloudVolume(target_layer_source, mip=mip)
        if (target_vol.resolution > 10000).any():
            source_voxel_size = list(np.array(target_vol.resolution) / 1000)
            break

    file_dir = pathlib.Path(path_to_affine).parent.parent
    path_to_source = file_dir / f"target_mip{mip}.tif"

    print(f"Downloading layer of mip {mip}...")
    img = np.squeeze(np.array(target_vol[:,:,:]))
    print(f"Saving image of shape {img.shape} to {path_to_source}...")
    imsave(path_to_source, img)

    atlas_vol = CloudVolume(atlas_viz_link)

    transformed_file = file_dir / f"transformed_mip{mip}.tif"

    # run matlab command to get transformed layer
    if path_to_affine != "" and path_to_velocity != "":
        # velocity field voxel size
        v_size = ", ".join(str(i) for i in velocity_voxel_size)
        # get current file path and set path to transform_points
        base_path = pathlib.Path(__file__).parent.parent.absolute() / 'registration'
        # base_path = os.path.expanduser("~/CloudReg/registration")
        transformed_points_path = "./transformed_points.mat"

        matlab_path = 'matlab'
        matlab_command = f"""
            {matlab_path} -nodisplay -nosplash -nodesktop -r \"addpath(\'{base_path}\');path_to_source=\'{path_to_source}\';source_voxel_size=[{source_voxel_size}];path_to_affine=\'{path_to_affine}\';path_to_velocity=\'{path_to_velocity}\';velocity_voxel_size=[{velocity_voxel_size}];destination_voxel_size=[10,10,10];destination_shape=[1320,800,1140];transformation_direction=\'atlas\';path_to_output=\'{transformed_file}\';interpolation_method=\'nearest\';transform_data(path_to_source,source_voxel_size,path_to_affine,path_to_velocity,velocity_voxel_size,destination_voxel_size,destination_shape,transformation_direction,path_to_output,interpolation_method);exit;\"
        """
        subprocess.run(shlex.split(matlab_command),)

        print(f"Transformed image saved at: {transformed_file}")

    else:
        raise Exception


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Transform image in Neuroglancer from one space to another given a transformation."
    )
    parser.add_argument(
        "--target_layer_source", help="Source URL to target data layer to be transformed", type=str
    )
    parser.add_argument(
        "--atlas_viz_link", help="Neuroglancer viz link to atlas (optionally with fiducials labelled if transforming to input data space). Default is link to ARA.", 
        type=str,
        default="https://ara.viz.neurodata.io/?json_url=https://json.neurodata.io/v1?NGStateID=ifm4oFKOl10eiA"
    )
    parser.add_argument(
        "--affine_path",
        help="S3 path or local path to matlab transformation files. These will be downloaded to compute the fiducial accuracy",
        type=str,
        default="",
    )
    parser.add_argument(
        "--velocity_path",
        help="S3 path ot local matlab transformation files. These will be downloaded to compute the fiducial accuracy",
        type=str,
        default="",
    )
    parser.add_argument(
        "--velocity_voxel_size",
        help="Voxel size of velocity field in microns",
        nargs="+",
        type=float,
        default=[100.0] * 3,
    )

    args = parser.parse_args()

    if args.affine_path.startswith("s3://") or args.affine_path.startswith("http"):
        # download affine mat to local storage
        aws_cli(shlex.split(f"s3 cp {args.affine_path} ./A.mat"))
        args.affine_path = "./A.mat"
    if args.velocity_path.startswith("s3://") or args.velocity_path.startswith("http"):
        # download velocity mat to local storage
        aws_cli(shlex.split(f"s3 cp {args.velocity_path} ./v.mat"))
        args.velocity_path = "./v.mat"


    transform_data(
        args.target_layer_source,
        args.atlas_viz_link,
        args.affine_path,
        args.velocity_path,
        args.velocity_voxel_size,
    )
