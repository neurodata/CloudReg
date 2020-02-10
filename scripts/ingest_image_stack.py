import time
import tifffile as tf
import numpy as np
import argparse
import os
import SimpleITK as sitk
import math
from cloudvolume import CloudVolume
import tinybrain

def get_vol_at_mip(precomputed_path, mip, parallel=False):
    return CloudVolume(precomputed_path,mip=mip,parallel=parallel)

def create_cloud_volume(precomputed_path,img_size,voxel_size, num_hierarchy_levels=5):
    info = CloudVolume.create_new_info(
        num_channels    = 1,
        layer_type      = 'image',
        data_type       = 'uint16', # Channel images might be 'uint8'
        encoding        = 'raw', # raw, jpeg, compressed_segmentation, fpzip, kempressed
        resolution      = voxel_size, # Voxel scaling, units are in nanometers
        voxel_offset    = [0, 0, 0], # x,y,z offset in voxels from the origin
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size      = [ 1024, 1024, 1 ], # units are voxels
        volume_size     = img_size, # e.g. a cubic millimeter dataset
    )
    vol = CloudVolume(precomputed_path,info=info)
    # add mip 1
    [vol.add_scale((2**i,2**i,1)) for i in range(num_hierarchy_levels)]
    vol.commit_info()
    return vol

def process(z, img):
    start = time.time()
    global layer_path, num_mips
    vols = [get_vol_at_mip(layer_path,i,parallel=False) for i in range(num_mips)]
    if img.dtype in (np.uint8, np.uint16, np.float32, np.float64):
        img_pyramid = tinybrain.accelerated.average_pooling_2x2(img, num_mips=num_mips)
    else:
        img_pyramid = tinybrain.accelerated.mode_pooling_2x2(img, num_mips=num_mips)
    vols[0][:,:, z] = img
    for i in range(num_mips-1):
        vols[i+1][:,:,z] = img_pyramid[i]

    print(f'Processing {z} took {time.time() - start}')

def main():

    parser = argparse.ArgumentParser(description='Ingest a tif stack into S3.')
    parser.add_argument('-s3_path', help='S3 path to store image as precomputed volume. ', type=str)
    parser.add_argument('-img_stack', help='Path to image stack to be uploaded', type=str)
    parser.add_argument('-fmt', help='extension of file. can be tif or img', type=str)
    parser.add_argument('-voxel_size', help='Voxel size of image. 3 numbers in nanometers', nargs=3, type=float)
    parser.add_argument('--dtype', help='Datatype of image', type=str, default='uint16')
    parser.add_argument('--new_channel', help='True or False depending on if this is an existing channel or new channel needs to be created', default=False)
    
    args = parser.parse_args()

    if (args.fmt == 'tif'): img = tf.imread(os.path.expanduser(args.img_stack))
    else: 
        tmp = sitk.ReadImage(os.path.expanduser(args.img_stack))
#        tmp = sitk.RescaleIntensity(tmp, outputMinimum=0, outputMaximum=10000)
        img = sitk.GetArrayFromImage(tmp)
    img = np.asarray(img, dtype=args.dtype)
    
    vol = create_cloud_volume(args.s3_path, img.shape, args.voxel_size)

    mem = virtual_memory() 
    num_procs = min(math.floor(mem.total/(img.shape[0]*img.shape[1]*8)),joblib.cpu_count())
    print(f"num processes: {num_procs}")
    print(f"layer path: {vol.layer_cloudpath}")
    global layer_path, num_mips
    num_mips = 7
    layer_path = vol.layer_cloudpath

    files = list(((i,img[:,:,i]) for i in range(img.shape[-1])))

    with ProcessPoolExecutor(max_workers=num_procs) as executor:
        executor.map(process, files)


if __name__ == '__main__':
    main()
