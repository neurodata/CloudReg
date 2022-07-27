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
from skimage.io import imsave, imread
import json
import random

def transform_data(
    target_layer_source,
    transformed_layer_source,
    path_to_affine,
    path_to_velocity,
    # voxel size of velocity field
    velocity_voxel_size,
):
    # identify layer to downlooad
    for mip in range(5):
        target_vol = CloudVolume(target_layer_source, mip=mip)
        if (target_vol.resolution > 5000).any():
            source_voxel_size = list(np.array(target_vol.resolution) / 1000)
            break
    transformed_vol = CloudVolume(transformed_layer_source)

    file_dir = pathlib.Path(path_to_affine).parent.parent
    path_to_source = file_dir / f"target_mip{mip}.tif"

    print(f"Downloading layer of mip {mip}...")
    #source_voxel_size = [source_voxel_size[2], source_voxel_size[1], source_voxel_size[0]]
    img = np.squeeze(np.array(target_vol[:,:,:])).T
    print(f"Saving image of shape {img.shape} with res {source_voxel_size} to {path_to_source}...")
    imsave(path_to_source, img, plugin="tifffile")

    transformed_file = file_dir / f"transformed_mip{mip}"

    # run matlab command to get transformed layer
    if path_to_affine != "" and path_to_velocity != "":
        # velocity field voxel size
        v_size = ", ".join(str(i) for i in velocity_voxel_size)
        # get current file path and set path to transform_points
        base_path = pathlib.Path(__file__).parent.parent.absolute() / 'registration'
        print(base_path)
        # base_path = os.path.expanduser("~/CloudReg/registration")

        matlab_path = 'matlab'
        matlab_command = f"""
            {matlab_path} -nodisplay -nosplash -nodesktop -r \"addpath(\'{base_path}\');path_to_source=\'{path_to_source}\';source_voxel_size=[{source_voxel_size}];path_to_affine=\'{path_to_affine}\';path_to_velocity=\'{path_to_velocity}\';velocity_voxel_size=[{velocity_voxel_size}];destination_voxel_size=[10,10,10];destination_shape=[1320,800,1140];transformation_direction=\'atlas\';path_to_output=\'{transformed_file}\';interpolation_method=\'linear\';transform_data(path_to_source,source_voxel_size,path_to_affine,path_to_velocity,velocity_voxel_size,destination_voxel_size,destination_shape,transformation_direction,path_to_output,interpolation_method);exit;\"
        """
        subprocess.run(shlex.split(matlab_command),)
        print(f"Transformed image saved at: {transformed_file}")

        img_trans = imread(str(transformed_file) + ".img")
        img_trans = np.swapaxes(img_trans, 0, 2)
        img_trans = img_trans.astype(transformed_vol.data_type)
        print(f"Uploading transformed image of shape {img_trans.shape} to {transformed_layer_source}...")
        transformed_vol[:,:,:] = img_trans[:,:,:]
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
        "--transformed_layer_source", help="URL for destination of transformed data.", 
        type=str
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
        args.transformed_layer_source,
        args.affine_path,
        args.velocity_path,
        args.velocity_voxel_size,
    )
