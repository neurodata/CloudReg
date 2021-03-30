from .util import imgResample
from cloudvolume import CloudVolume
from argparse import ArgumentParser
import numpy as np
import SimpleITK as sitk
import tifffile as tf


def get_mip_at_res(vol, resolution):
    """Find the mip that is at least a given resolution

    Args:
        vol (cloudvolume.CloudVoluem): CloudVolume object for desired precomputed volume
        resolution (int): Desired resolution in nanometers

    Returns:
        tuple: mip and resolution at that mip
    """
    tmp_mip = 0
    tmp_res = 0
    for i, scale in enumerate(vol.scales):
        if (scale["resolution"] <= resolution).all():
            tmp_mip = i
            tmp_res = scale["resolution"]
        elif i == 0:
            tmp_res = scale["resolution"]
            return tmp_mip, tmp_res

    return tmp_mip, tmp_res


def download_data(s3_path, outfile, desired_resolution, resample_isotropic=False, return_size=False):
    """Download whole precomputed volume from S3 at desired resolution and optionally resample data to be isotropic

    Args:
        s3_path (str): S3 path to precomputed volume
        outfile (str): Path to output file
        desired_resolution (int): Lowest resolution (in nanometers) at which to download data if desired_resolution isnt available.
        resample_isotropic (bool, optional): If true, resample data to be isotropic at desired_resolution.

    Returns:
        resolution: Resoluton of downloaded data in microns
    """
    vol = CloudVolume(s3_path)
    mip_needed, resolution = get_mip_at_res(vol, np.array([desired_resolution] * 3))
    vol = CloudVolume(s3_path, mip=mip_needed, parallel=True)

    # download img and convert to C order
    img = np.squeeze(vol[:, :, :]).T
    # save out as correct file type
    img_s = sitk.GetImageFromArray(img)
    # set spacing in microns
    resolution = np.divide(resolution, 1000.0).tolist()
    img_s.SetSpacing(resolution)
    if resample_isotropic:
        img_s = imgResample(img_s, np.divide([desired_resolution]*3,1000.))
        resolution = np.divide([desired_resolution]*3,1000.)
    # if output is tiff, use tiffile
    if 'tif' in outfile.split('.')[-1]:
        tf.imwrite(outfile, sitk.GetArrayFromImage(img_s))
    # else use SimpleITK
    else:
        sitk.WriteImage(img_s, outfile)
    # tf.imsave(outfile, img.T, compress=3)

    # return resolution in um
    if return_size:
        return (resolution, vol.scales[mip_needed]['size'])
    return resolution


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Download volume from S3 for subsequent registration."
    )
    parser.add_argument(
        "s3_path",
        help="S3 path to precomputed volume layer in the form s3://<bucket-name>/<path-to-precomputed-volume>",
    )
    parser.add_argument("outfile", help="name of output file with associated file extension. eg. /path/to/image.tif")
    parser.add_argument(
        "desired_resolution",
        help="Desired minimum resolution for downloaded image in nanometers. Resolution assumed to be same in all 3 dimensions.",
        type=int
    )
    parser.add_argument("--isotropic", help="Resample data to isotropic at desired_resolution.", action='store_true')
    args = parser.parse_args()

    download_data(
        args.s3_path, 
        args.outfile, 
        args.desired_resolution,
        resample_isotropic=args.isotropic
    )
