from cloudvolume import CloudVolume
from argparse import ArgumentParser
import tifffile as tf
import numpy as np


def get_mip_at_res(vol,resolution):
    tmp_mip = 0
    tmp_res = 0
    for i,scale in enumerate(vol.scales):
        if (scale['resolution'] < resolution).all():
            tmp_mip = i
            tmp_res = scale['resolution']
    print(f"saving resolution (in nm): {tmp_res}")
    return tmp_mip


def download_data(s3_path, outfile, desired_resolution=15000):
    vol = CloudVolume(s3_path)
    mip_needed = get_mip_at_res(vol,np.array([desired_resolution]*3))
    vol = CloudVolume(args.s3_path,mip=mip_needed,parallel=True)

    # img is F order
    img = vol[:,:,:]
    # save out as C order
    tf.imsave(args.outfile,img.T,compress=3)




if __name__ == "__main__":
    parser = ArgumentParser(description='Download volume from S3 for subsequent registration.')
    parser.add_argument('s3_path',help='S3 path to precomputed volume layer in the form s3://<bucket-name>/<path-to-precomputed-volume>')
    parser.add_argument('outfile',help='name of output file saved as tif stack.')
    parser.add_argument('desired_resolution',help='Desired minimum resolution for downloaded image.', nargs='+')
    args = parser.parse_args()

    download_data(
        args.s3_path,
        args.outfile,
    )