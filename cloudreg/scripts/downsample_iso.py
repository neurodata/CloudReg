from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
from cloudvolume import CloudVolume
import argparse

def downsample_isotropically(input_path, output_path, compress=False):
  tq = LocalTaskQueue(parallel=8)

  vol = CloudVolume(input_path)
  print(f"Original chunk size: {vol.chunk_size}, shape: {vol.shape}")

  tasks = tc.create_transfer_tasks(
    input_path, output_path, 
    chunk_size=[128,128,128], compress=compress,
    skip_downsamples = True
  )

  tq.insert(tasks)
  tq.execute()

  vol = CloudVolume(output_path, mip=0)
  print(f"Output chunk size: {vol.chunk_size}, shape: {vol.shape}")

  tasks = tc.create_downsampling_tasks(
      output_path, # e.g. 'gs://bucket/dataset/layer'
      compress=compress, # None, 'gzip', and 'br' (brotli) are options
      factor=(2,2,2), # common options are (2,2,1) and (2,2,2)
    )

  tq.insert(tasks)
  tq.execute()
  print("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert precomputed layer to isotropic downsampled structure. Built to follow create_precomputed_volume"
    )
    parser.add_argument(
        "input_path",
        help="Path to directory containing layer at highest resolution. e.g. file://<path>",
    )
    parser.add_argument(
        "output_path",
        help="Path to directory for output volume.",
    )
    args = parser.parse_args()

    downsample_isotropically(
        args.input_path, args.output_path
    )
