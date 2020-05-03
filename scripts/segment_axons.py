from argparse import ArgumentParser
from cloudvolume import CloudVolume
import tifffile as tf
import numpy as np
from skimage import filters, morphology, measure, exposure, util, segmentation, transform, feature
from tqdm import tqdm, tqdm_notebook, trange, tnrange

from scipy import ndimage as ndi

from glob import glob

import SimpleITK as sitk

import pandas as pd

from skimage.morphology import selem,binary_opening
from joblib import Parallel,delayed

from tqdm import tnrange

import tinybrain

def get_vol_at_mip(precomputed_path, mip, parallel=False,progress=False):
    return CloudVolume(precomputed_path,mip=mip,parallel=parallel,progress=progress)

def compute_gradient_slice(data_s3_path,output_s3_path,z_slice,num_mips=6,sigma=2,dtype='float32'):
    # create vols
    data_vol = CloudVolume(data_s3_path,parallel=False,progress=False)
    binarized_vols = [get_vol_at_mip(output_s3_path,i) for i in range(num_mips)]
    data_size = data_vol.scales[0]['size'][::-1]
    data_slice = np.squeeze(data_vol[:,:,z_slice]).T
    data_native_sitk = sitk.GetImageFromArray(data_slice)
    data_native_sitk_b = sitk.SmoothingRecursiveGaussian(data_native_sitk,sigma)
    data_native_sitk_grad = sitk.GradientMagnitude(data_native_sitk_b)
    data_native_sitk_grad_b = sitk.SmoothingRecursiveGaussian(data_native_sitk_grad,3)
    binarized_slice = sitk.GetArrayViewFromImage(data_native_sitk_grad_b)
    # upload to S3
    img_pyramid = tinybrain.accelerated.average_pooling_2x2(binarized_slice.T[:,:,None].astype(dtype), num_mips)
    binarized_vols[0][:,:,z_slice] = binarized_slice.T[:,:,None].astype(dtype)
    for i in range(num_mips-1):
        binarized_vols[i+1][:,:,z_slice] = img_pyramid[i]
    print(f"{z_slice} z slice done")

def remove_small_ccs(data_binarized, min_cc_size):
    data_labeled = sitk.ConnectedComponent(data_binarized)
    relabel_filter = sitk.RelabelComponentImageFilter()
    data_relabeled = relabel_filter.Execute(data_labeled)
    labels_to_remove = [i+1 for i,j in enumerate(relabel_filter.GetSizeOfObjectsInPixels()) if j < min_cc_size]
    if len(labels_to_remove) > 0:
        data_relabeled = sitk.BinaryThreshold(data_relabeled,lowerThreshold=1,upperThreshold=labels_to_remove[0])
    return data_relabeled

def threshold_slice(data_slice, threshold, min_cc_size=150, closing_radius=3):
    data_sitk = sitk.GetImageFromArray(data_slice)
    # binarize slice
    data_native_binarized = sitk.BinaryThreshold(data_sitk,lowerThreshold=threshold)
    data_labeled = remove_small_ccs(data_native_binarized, min_cc_size)
    print("before closing")
    data_relabeled2 = sitk.BinaryClosingByReconstruction(data_labeled,closing_radius)
    print("after closing")
    binarized_slice = sitk.GetArrayViewFromImage(data_relabeled2)
    return binarized_slice

def binarize_slice(data_s3_path,binarized_s3_path,z_slice,data_threshold,mask_s3_path=None,mask_threshold=None,num_mips=6,dtype='uint8'):
    min_cc_size = 150
    closing_radius = 3
    # create vols
    data_vol = CloudVolume(data_s3_path,parallel=False,progress=False)
    binarized_vols = [get_vol_at_mip(binarized_s3_path,i) for i in range(num_mips)]
    data_size = data_vol.scales[0]['size'][::-1]
    data_slice = np.squeeze(data_vol[:,:,z_slice]).T
#     binarized_slice = data_slice
#     binarized_slice = threshold_slice(data_slice,data_threshold)
    data_sitk = sitk.GetImageFromArray(data_slice)
    # binarize slice
    data_native_binarized = sitk.BinaryThreshold(data_sitk,lowerThreshold=data_threshold)
    data_labeled = remove_small_ccs(data_native_binarized, min_cc_size)
    data_relabeled2 = sitk.BinaryClosingByReconstruction(data_labeled,closing_radius)
    binarized_slice = sitk.GetArrayViewFromImage(data_relabeled2)
    if mask_s3_path != None and mask_threshold != None:
        mask_vol = CloudVolume(mask_s3_path,parallel=False,progress=False)
        mask_slice = np.squeeze(mask_vol[:,:,z_slice]).T > mask_threshold
        binarized_slice = binarized_slice.astype('float') - mask_slice.astype('float')
        binarized_slice[ binarized_slice < 0 ] = 0
        data_sitk = sitk.GetImageFromArray(binarized_slice.astype('uint8'))
        binarized_slice_sitk = remove_small_ccs(data_sitk,min_cc_size=100)
        binarized_slice_sitk = sitk.BinaryClosingByReconstruction(binarized_slice_sitk,3)
        binarized_slice = sitk.GetArrayViewFromImage(binarized_slice_sitk)
    img_pyramid = tinybrain.accelerated.mode_pooling_2x2(binarized_slice.T[:,:,None].astype(dtype), num_mips)
    binarized_vols[0][:,:,z_slice] = binarized_slice.T[:,:,None].astype(dtype)
    for i in range(num_mips-1):
        binarized_vols[i+1][:,:,z_slice] = img_pyramid[i]
    print(f"{z_slice} z slice done")

def create_binarized_vol(vol_path_bin,vol_path_old,ltype='image',dtype='float32',res=0,parallel=False):
    vol = CloudVolume(vol_path_old)
    info = CloudVolume.create_new_info(
        num_channels    = 1,
        layer_type      = ltype,
        data_type       = dtype, # Channel images might be 'uint8'
        encoding        = 'raw', # raw, jpeg, compressed_segmentation, fpzip, kempressed
        resolution      = vol.scales[res]['resolution'], # Voxel scaling, units are in nanometers
        voxel_offset    = [0, 0, 0], # x,y,z offset in voxels from the origin
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size      = [ 1024, 1024, 1 ], # units are voxels
        volume_size     = vol.scales[res]['size'],
    )
    bin_vol = CloudVolume(vol_path_bin,info=info,parallel=parallel)
    [bin_vol.add_scale((2**i,2**i,1),chunk_size=[ 1024, 1024, 1 ]) 
         for i in range(6)]
    bin_vol.commit_info()
    return bin_vol

def main():
    parser = ArgumentParser('Binarize slice gievn two s3 paths and global threshold.')
    parser.add_argument('--num_processes', default=16, help='Number of parallel processes to use to process dataset.')
    args = parser.parse_args()
    # atlas_s3_path = 'https://d1o9rpg615hgq7.cloudfront.net/precomputed_volumes/2020-01-15/Gad2_812/atlas_to_target'
    data_s3_path = 'https://dlab-colm.neurodata.io/precomputed_volumes/2020-01-15/Gad2_812/CHN00_gradient'
    mask_s3_path = 'https://dlab-colm.neurodata.io/precomputed_volumes/2020-01-15/Gad2_812/CHN01_gradient'
    binarized_s3_path = 's3://colm-precomputed-volumes/precomputed_volumes/2020-01-15/Gad2_812/CHN00_binarized'

    bin_vol = create_binarized_vol(binarized_s3_path, data_s3_path, ltype='segmentation', dtype='uint8',res=0,parallel=False)

    _ = Parallel(args.num_processes)(delayed(binarize_slice)
                (data_s3_path,binarized_s3_path,i,1.96)
                    for i in tqdm_notebook(bin_vol.scales[0]['size'][-1]))