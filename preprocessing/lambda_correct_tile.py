import os
import time
from io import BytesIO

import boto3
import numpy as np
from PIL import Image
import tifffile as tf

def correct_tile(s3,raw_tile_bucket, raw_tile_path, out_path, bias, out_bucket):
    start_time = time.time()
    raw_tile_obj = s3.Object(raw_tile_bucket, raw_tile_path)
    raw_tile = np.asarray(Image.open(BytesIO(raw_tile_obj.get()["Body"].read())))
    print(f'PULL - time: {time.time() - start_time}, path: {raw_tile_path}')
    start_time = time.time()
    # tf.imsave(out_path, data=(raw_tile * bias))
    # img = Image.fromarray(raw_tile * bias)
    # fp = BytesIO()
    # img.save(fp,format='TIFF',compression='tiff_lzw')
    fp = BytesIO()
    tf.imwrite(fp, data=(raw_tile * bias), compress=1)
    # reset pointer to beginning  of file
    fp.seek(0)
    s3.Object(out_bucket, out_path).upload_fileobj(fp)
    print(f'SAVE - time: {time.time() - start_time} s path: {out_path}')

def lambda_handler(event, context):
    s3 = boto3.resource("s3")
    # read in bias tile
    attributes = event['Records'][0]['messageAttributes']
    print(attributes)
    bias_obj = s3.Object(
        attributes["BiasBucket"]["stringValue"], attributes["BiasPath"]["stringValue"]
    )
    bias = np.asarray(Image.open(BytesIO(bias_obj.get()["Body"].read())))
    for message in event['Records']:
        attributes = message['messageAttributes']
        correct_tile(
            s3,
            attributes["RawTileBucket"]["stringValue"],
            attributes["RawTilePath"]["stringValue"],
            attributes["OutPath"]["stringValue"],
            bias,
            attributes["OutBucket"]["stringValue"]
        )
