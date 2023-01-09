import argparse
import numpy as np
from .create_precomputed_volume import create_precomputed_volume
import os

if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(
        description="Convert local volume into precomputed volume on S3."
    )
    parser.add_argument(
        "--input_parent_dir",
        help="Path to parent of directories containing stitched tiles named sequentially."
    )
    parser.add_argument(
        "--s3_output_parent_dir",
        help="Path to parent of directories containing stitched tiles named sequentially."
    )
    parser.add_argument(
        "--local_output_parent_dir",
        help="Path to parent of directories containing stitched tiles named sequentially."
    )
    parser.add_argument(
        "--s3_input_paths",
        help="Path to directory containing stitched tiles named sequentially.", nargs='+', default=[]
    )
    parser.add_argument(
        "--voxel_size",
        help="Voxel size in microns of image in 3D in X, Y, Z order.",
        nargs="+",
        type=float,
    )
    parser.add_argument(
        "--s3_output_paths",
        help="Path to location on s3 where precomputed volume should be stored. Example: s3://<bucket>/<experiment>/<channel>", nargs='+', default=[]
    )
    parser.add_argument(
        "--local_input_paths",
        help="Path to directory containing stitched tiles named sequentially.", nargs='+', default=[]
    )
    parser.add_argument(
        "--local_output_paths",
        help="Path to local location where precomputed volume should be stored. Example: file:///<bucket>/<experiment>/<channel>", nargs='+', default=[]
    )
    parser.add_argument(
        "--num_procs",
        help="Number of processes to use in parallel. It is possible we may exceed the request rate so you may want to reduce the number of cores.",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--compress",
        help="Whether to use a compressed format for the precomputed volume.",
        default=False,
        type=str2bool,
    )
    parser.add_argument(
        "--resample_iso",
        help="Whether to immediately write another version of the volume that has isotropic chunks to be able to use several views on neuroglancer.",
        default=False,
        type=str2bool,
    )
    args = parser.parse_args()

    if args.input_parent_dir == None:
        s3_input_paths = args.s3_input_paths
        s3_output_paths = args.s3_output_paths
        local_input_paths = args.local_input_paths
        local_output_paths = args.local_output_paths
    else:
        channels_in = ["Ex_561_Em_600_stitched/", "Ex_488_Em_525_stitched/", "Ex_647_Em_680_stitched/"]
        channels_out = ["Ch_561/", "Ch_488/", "Ch_647/"]
        s3_input_paths = [os.path.join(args.input_parent_dir, channel) for channel in channels_in]
        local_input_paths = [os.path.join(args.input_parent_dir, channel) for channel in channels_in[:1]]

        s3_output_paths = [os.path.join(args.s3_output_parent_dir, channel) for channel in channels_out]
        local_output_paths = [os.path.join(args.local_output_parent_dir, channel) for channel in channels_out[:1]]

        print("Generated input/output paths automatically:")
        for i,j in zip(s3_input_paths, s3_output_paths):
            print(f"Writing {i} to {j}")
        for i,j in zip(local_input_paths, local_output_paths):
            print(f"Writing {i} to {j}")
        cont = input("Are these paths correct? (y/n)")
        if cont != "y":
            raise ValueError("User chose not to proceed with writing volumes (did not enter y)")

    for input_path, precomputed_path in zip(s3_input_paths, s3_output_paths):
        print(f"**************Writing {input_path} to {precomputed_path} on s3*****************")
        create_precomputed_volume(
            input_path,
            np.array(args.voxel_size),
            precomputed_path,
            num_procs=args.num_procs,
            compress=args.compress,
            resample_iso=args.resample_iso,
        )


    for input_path, precomputed_path in zip(local_input_paths, local_output_paths):
        print(f"**************Writing {input_path} to {precomputed_path} locally*****************")
        create_precomputed_volume(
            input_path,
            np.array(args.voxel_size),
            precomputed_path,
            num_procs=args.num_procs,
            compress=args.compress,
            resample_iso=False,
        )
