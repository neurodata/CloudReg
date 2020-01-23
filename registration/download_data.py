from cloudvolume import CloudVolume
from argparse import ArgumentParser
import tifffile as tf
import numpy as np


def get_mip_at_res(vol,resolution):
    tmp_mip = 0
    for i,scale in enumerate(vol.scales):
        print(scale['resolution'])
        if (i['resolution'] < resolution).all():
            tmp_mip = i
    return tmp_mip


def main():
    parser = ArgumentParser(description='Download volume from S3 for subsequent registration.')
    parser.add_argument('s3_path',help='S3 path to precomputed volume layer in the form s3://<bucket-name>/<path-to-precomputed-volume>')
    parser.add_argument('outfile',help='name of output file saved as tif stack.')
#    parser.add_argument('input_xml',help='Path to xml_import.xml file to get metadata')
#    parser.add_argument('precomputed_path',help='Path to location on s3 where precomputed volume should be stored. Example: s3://<bucket>/<experiment>/<channel>')
#    parser.add_argument('--extension',help='Extension of stitched files. default is tif', default='tif',type=str)
    args = parser.parse_args()

    registration_res = 50000.0 # nanometers
    vol = CloudVolume(args.s3_path)
    mip_needed = get_mip_at_res(vol,np.array([registration_res]*3))
    vol = CloudVolume(args.s3_path,mip=mip_needed,parallel=True)

    # img is F order
    img = vol[:,:,:]
    # save out as C order
    tf.imsave(args.outfile,img,compress=3)




if __name__ == "__main__":
    main()
