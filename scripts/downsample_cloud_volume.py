# downsample using igneous
#import gevent.monkey 
#gevent.monkey.patch_all()
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
import argparse
import joblib

def downsample(precomputed_path, num_mips, starting_mip):
    
    num_cpus = joblib.cpu_count()
    with LocalTaskQueue(parallel=num_cpus-1) as tq:
        tasks = tc.create_downsampling_tasks(
            precomputed_path,
            mip=starting_mip,  # Start downsampling from this mip level (writes to next level up)
            fill_missing=True,  # Ignore missing chunks and fill them with black
            axis="z",
            num_mips=num_mips,  # number of downsamples to produce. Downloaded shape is chunk_size * 2^num_mip
            chunk_size=None,  # manually set chunk size of next scales, overrides preserve_chunk_size
            preserve_chunk_size=True,  # use existing chunk size, don't halve to get more downsamples
            sparse=False,  # for sparse segmentation, allow inflation of pixels against background
            bounds=None,  # mip 0 bounding box to downsample
            encoding=None,  # e.g. 'raw', 'compressed_segmentation', etc
        )
        tq.insert_all(tasks)
    print("Done!")

def main():
    parser = argparse.ArgumentParser('Downsample a precomputed volume using Igneous')
    parser.add_argument('--precomputed_path', help='Path to precomputed volume to downsample.')
    parser.add_argument('--starting_mip', help='mip to start out when downsampling. Default is 0',default=0,type=int)
    parser.add_argument('--num_mips', help='Number of resolutions to produce. Each resolution anisotropically downsamples by 2^mip in X and Y.',default=3,type=int)

    args = parser.parse_args()

    downsample(args.precomputed_path,args.num_mips,args.starting_mip)

if __name__ == "__main__":
    main()
