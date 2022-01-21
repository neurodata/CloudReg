from taskqueue import LocalTaskQueue
import igneous.task_creation as tc

tq = LocalTaskQueue(parallel=8)

layer_path = "file:///mnt/data/Neuroglancer_Data/2021_10_06/8557/Ch_647_Iso"

tasks = tc.create_downsampling_tasks(
    layer_path, # e.g. 'gs://bucket/dataset/layer'
    mip=0, # Start downsampling from this mip level (writes to next level up)
    fill_missing=True, # Ignore missing chunks and fill them with black
    axis='z', 
    num_mips=5, # number of downsamples to produce. Downloaded shape is chunk_size * 2^num_mip
    chunk_size=[512,512, 512], # manually set chunk size of next scales, overrides preserve_chunk_size
    preserve_chunk_size=False, # use existing chunk size, don't halve to get more downsamples
    sparse=False, # for sparse segmentation, allow inflation of pixels against background
    bounds=None, # mip 0 bounding box to downsample 
    encoding=None, # e.g. 'raw', 'compressed_segmentation', etc
    delete_black_uploads=False, # issue a delete instead of uploading files containing all background
    background_color=0, # Designates the background color
    compress=False, # None, 'gzip', and 'br' (brotli) are options
    factor=(2,2,2), # common options are (2,2,1) and (2,2,2)
  )

tq.insert(tasks)
tq.execute()
print("Done!")