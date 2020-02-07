import math
from cloudvolume import CloudVolume
import numpy as np
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed
import joblib
from glob import glob
import argparse
import bs4
import time
import tifffile as tf
import PIL
from psutil import virtual_memory

from skimage import transform
import tinybrain

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from cloudvolume import CloudVolume
from cloudvolume.lib import mkdir, touch


PIL.Image.MAX_IMAGE_PIXELS = None

CHUNK_SIZE = 16

def chunks(l,n):
    for i in range(0,len(l),n):
        yield l[i:i+n]


def create_cloud_volume(precomputed_path,img_size,voxel_size,num_hierarchy_levels=6,parallel=True):
    chunk_size = [ 1024, 1024, 1 ]
    info = CloudVolume.create_new_info(
        num_channels    = 1,
        layer_type      = 'image',
        data_type       = 'uint16', # Channel images might be 'uint8'
        encoding        = 'raw', # raw, jpeg, compressed_segmentation, fpzip, kempressed
        resolution      = voxel_size, # Voxel scaling, units are in nanometers
        voxel_offset    = [0, 0, 0], # x,y,z offset in voxels from the origin
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size      = chunk_size, # units are voxels
        volume_size     = img_size, # e.g. a cubic millimeter dataset
    )
    vol = CloudVolume(precomputed_path,info=info,parallel=parallel)
    # add mip 1
    [vol.add_scale((2**i,2**i,1),chunk_size=chunk_size) for i in range(num_hierarchy_levels)]

    vol.commit_info()
    return vol

def get_vol_at_mip(precomputed_path, mip, parallel=True):
    return CloudVolume(precomputed_path,mip=mip,parallel=parallel)


def load_image(path_to_file,transpose=True):
    image = np.squeeze(np.asarray(Image.open(path_to_file)))
    if transpose:
        return image.T
    return image


def load_image_to_array(path_to_file,out_array,z_idx,transpose=True):
    # image = tf.imread(path_to_file)
    image = np.squeeze(np.asarray(Image.open(path_to_file)))
    if transpose:
        out_array[:,:,z_idx] = image.T
    else:
        out_array[:,:,z_idx] = image


def downsample_image(image,out_arrays,num_mips,z_idx,factor=(2,2,1)):
    img_pyramid = tinybrain.downsample_with_averaging(image, factor=factor, num_mips=num_mips)
    #print(f"num mips based on img_pyramid: {len(img_pyramid)}")
    for i in range(1,num_mips):
        out_arrays[i][:,:,z_idx] = img_pyramid[i-1]


def upload_image_to_volume(vol,files):
    size = vol.info['scales'][0]['size']
    num_mips = len(vol.info['scales'])
    vols = [get_vol_at_mip(vol.layer_cloudpath,i,parallel=True) for i in range(num_mips)]
    for i,f in tqdm(enumerate(chunks(files,CHUNK_SIZE)),total=int(len(files)/CHUNK_SIZE)+1):
        tmp_chunk = np.zeros((size[0],size[1],len(f)),dtype='uint16',order='F')
        #s = time.time()
        Parallel(n_jobs=len(f),require='sharedmem')(delayed(load_image_to_array)(img,tmp_chunk,idx) for idx,img in enumerate(f))
        #with tf.TiffSequence(f) as imseq:
        #    imseq.asarray(ioworkers=len(f), out=tmp_chunk.T)
        #print(f"took {time.time() - s} seconds to load {CHUNK_SIZE} slices into memory")
        start = i*CHUNK_SIZE
        end = start + len(f)
        # time upload at  res 0
        #s = time.time()
        vol[:,:,start:end]=tmp_chunk
        #print(f"took {time.time() - s} seconds to upload {CHUNK_SIZE} slices at res 0")
        for j in range(num_mips-1):
            #s = time.time()
            #print(f'flags: {img_pyramid[j].flags}')
            vols[j+1][:,:,start:end] = img_pyramid[j]
            #print(f"took {time.time() - s} seconds to upload {CHUNK_SIZE} slices at res {j+1}")


def get_image_dims(files):
    # get X,Y size of image by loading first slice
    img = load_image(files[0])
    # get Z size by number of files in directory
    z_size = len(files)
    x_size,y_size = img.shape
    return [x_size,y_size,z_size]

def get_voxel_dims(path_to_xml):
    strainer = bs4.SoupStrainer(name='voxel_dims')
    with open(path_to_xml,'r') as f:
        x = f.read()
    soup = bs4.BeautifulSoup(x,features='html.parser',parse_only=strainer)
    x_size = float(soup.voxel_dims.attrs['v'])*1000
    y_size = float(soup.voxel_dims.attrs['h'])*1000
    z_size = float(soup.voxel_dims.attrs['d'])*1000
    return [x_size,y_size,z_size]

def parallel_assign_image(array,idx,image):
    array[:,:,idx] = image

def process(z,file_path):
#    img_name = 'brain_%06d.tif' % z
    start = time.time()
   # print(f"starting {z}")
#    global vol
    global layer_path, progress_dir, num_mips
    vols = [get_vol_at_mip(layer_path,i,parallel=False) for i in range(num_mips)]
    image = Image.open(file_path)
    width, height = image.size
    array = np.array(image).T
    array = array.reshape((width,height,1))
#    print(f"arrayshape: {array.shape}\nF_CONTIGUOUS:{array.flags['F_CONTIGUOUS']}")
    img_pyramid = tinybrain.accelerated.average_pooling_2x2(array, num_mips)
    vols[0][:,:, z] = array
    for i in range(num_mips-1):
        vols[i+1][:,:,z] = img_pyramid[i]
    image.close()
    touch(os.path.join(progress_dir, str(z)))
    print(f'Processing {z} took {time.time() - start}')

def main():
    parser = argparse.ArgumentParser(description='Convert local volume into precomputed volume on S3.')
    parser.add_argument('input_path',help='Path to directory containing stitched tiles named sequentially.')
    parser.add_argument('input_xml',help='Path to xml_import.xml file to get metadata')
    parser.add_argument('precomputed_path',help='Path to location on s3 where precomputed volume should be stored. Example: s3://<bucket>/<experiment>/<channel>')
    parser.add_argument('--extension',help='Extension of stitched files. default is tif', default='tif',type=str)
    args = parser.parse_args()


    start_time = time.time()
    files_slices = list(enumerate(np.sort(glob(f'{args.input_path}/*.{args.extension}')).tolist()))
    zs = [i[0] for i in files_slices]
    files = np.array([i[1] for i in files_slices])
    print(f'input path: {args.input_path}')
    img_size = get_image_dims(files)
    voxel_size = get_voxel_dims(args.input_xml)
    print(f'image size is: {img_size}')
    print(f'voxel size is: {voxel_size}')
    global vol,num_mips,progress_dir
    num_mips = 8
    vol = create_cloud_volume(args.precomputed_path,img_size,voxel_size,parallel=False,num_hierarchy_levels=num_mips)
    progress_dir = mkdir('progress/') # unlike os.mkdir doesn't crash on prexisting 
    done_files = set([ int(z) for z in os.listdir(progress_dir) ])
    all_files = set(range(vol.bounds.minpt.z, vol.bounds.maxpt.z))
    
    to_upload = [ int(z) for z in list(all_files.difference(done_files)) ]
    to_upload.sort()
    remaining_files = files[to_upload]
#    vols = [CloudVolume(vol.layer_cloudpath,parallel=False) for i in range(len(remaining_files))]
    mem = virtual_memory() 
    num_procs = min(math.floor(mem.total/(img_size[0]*img_size[1]*8)),joblib.cpu_count())
    print(f"num processes: {num_procs}")
    print(f"layer path: {vol.layer_cloudpath}")
    global layer_path
    layer_path = vol.layer_cloudpath

    with ProcessPoolExecutor(max_workers=num_procs) as executor:
        executor.map(process, to_upload, remaining_files)

    print(f"took {time.time() - start_time} seconds to upload to S3")

if __name__ == "__main__":
    main()
