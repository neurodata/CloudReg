import pickle
import argparse
from joblib import Parallel, delayed
import numpy as np
from cloudvolume import CloudVolume
from collections import defaultdict,Counter
from skimage import transform
from tqdm import tqdm, trange

def get_region_stats(atlas_s3_path, data_s3_path, z_slice):
    # create vols
    atlas_vol = CloudVolume(atlas_s3_path, parallel=False, progress=False)
    data_vol = CloudVolume(data_s3_path, parallel=False, progress=False)
    data_size = data_vol.scales[0]['size'][::-1]
    # use vols
    fluorescence_sum = defaultdict(lambda: 0)
    region_volume = defaultdict(lambda: 0)
    atlas_slice = np.squeeze(atlas_vol[:, :, z_slice]).T
    atlas_slice_upsampled = transform.resize(atlas_slice, data_size[1:], order=0, preserve_range=True)
    unique_vals = np.unique(atlas_slice_upsampled)
    data_slice = np.squeeze(data_vol[:, :, z_slice]).T
    for j in unique_vals:
        if j == 0:
            continue
        idx = atlas_slice_upsampled == j
        fluorescence_sum[j] += np.sum(data_slice[idx])
        region_volume[j] += np.count_nonzero(idx)
    #print(f"{z_slice} z slice done")
#     with open('fluorescence_quantification_vglut3_539', 'wb') as fp:
#         pickle.dump([fluorescence_sum,region_volume], fp)
    return fluorescence_sum, region_volume

def combine_results(results):
    total_fluorescence = Counter()
    total_volume = Counter()
    for i in results:
        total_fluorescence += Counter(i[0])
        total_volume += Counter(i[1])
    return total_fluorescence,total_volume

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_s3_path',help='full s3 path to data of interest as precomputed volume. must be of the form `s3://bucket-name/path/to/channel`')
    parser.add_argument('atlas_s3_path',help='full s3 path to transfomed atlas. must have the same number of slices as native resolution data.')
    parser.add_argument('out_path',help='path to save output results')
    parser.add_argument('--num_procs',help='number of processes to use',default=35, type=int)
    args = parser.parse_args()
    data_vol = CloudVolume(args.data_s3_path)
    results = Parallel(args.num_procs)(delayed(get_region_stats)(args.atlas_s3_path, args.data_s3_path,i) 
                       for i in trange(data_vol.scales[0]['size'][-1]))
    total_fluorescence, total_volume = combine_results(results)
    fluorescence_density = defaultdict(float)
    for i,j in total_fluorescence.items():
        fluorescence_density[i] = float(j)/float(total_volume[i])
    experiment_name = '_'.join(args.data_s3_path.split('/')[-2:])
    with open(f'{args.out_path}/{experiment_name}_fluorescence_quantification', 'wb') as fp:
        pickle.dump([total_fluorescence,total_volume], fp)

if __name__ == "__main__":
    main()
