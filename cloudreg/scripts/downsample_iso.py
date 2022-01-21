from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
from cloudvolume import CloudVolume

tq = LocalTaskQueue(parallel=8)

src_layer_path = "file:///mnt/data/Neuroglancer_Data/2021_10_06/8557/Ch_647_Iso"
dest_layer_path = "file:///mnt/data/Neuroglancer_Data/2021_10_06/8557/Ch_647_Iso2"

vol = CloudVolume(src_layer_path)
print(f"chunk size: {vol.chunk_size}, shape: {vol.shape}")

vol = CloudVolume(dest_layer_path, mip=0)
print(f"chunk size: {vol.chunk_size}, shape: {vol.shape}")


# tasks = tc.create_transfer_tasks(
#   src_layer_path, dest_layer_path, 
#   chunk_size=[128,128,128], compress=False,
#   skip_downsamples = True
# )

# tq.insert(tasks)
# tq.execute()

tasks = tc.create_downsampling_tasks(
    dest_layer_path, # e.g. 'gs://bucket/dataset/layer'
    compress=False, # None, 'gzip', and 'br' (brotli) are options
    factor=(2,2,2), # common options are (2,2,1) and (2,2,2)
  )

tq.insert(tasks)
tq.execute()
print("Done!")