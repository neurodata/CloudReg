from taskqueue import LocalTaskQueue
import igneous.task_creation as tc

tq = LocalTaskQueue(parallel=8)

src_layer_path = "file:///mnt/data/Neuroglancer_Data/2021_10_06/8557/Ch_647_Iso"
dest_layer_path = "file:///mnt/data/Neuroglancer_Data/2021_10_06/8557/Ch_647_Iso2"

tasks = tc.create_transfer_tasks(
  src_layer_path, dest_layer_path, 
  chunk_size=[128,128,128], compress=False,
  factor=(2,2,2)
)

tq.insert(tasks)
tq.execute()
print("Done!")