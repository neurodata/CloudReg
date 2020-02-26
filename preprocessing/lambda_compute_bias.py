import time
from io import BytesIO

import boto3
from boto3.dynamodb.conditions import Key, Attr
import numpy as np
from PIL import Image
from hashlib import sha1


def sum_tile(s3, running_sum, raw_tile_bucket, raw_tile_path, delete_file=False):
    start_time = time.time()
    raw_tile_obj = s3.Object(raw_tile_bucket, raw_tile_path)
    raw_tile = np.asarray(Image.open(BytesIO(raw_tile_obj.get()["Body"].read())))
    print(f"PULL - time: {time.time() - start_time}, path: {raw_tile_path}")
    start_time = time.time()
    # tf.imsave(out_path, data=(raw_tile * bias))
    running_sum += raw_tile
    print(f"SUM - time: {time.time() - start_time} s path: {raw_tile_path}")
    if delete_file:
        raw_tile_obj.delete()

def post_tile(s3,queue,tile,out_bucket,bias_path,total_tiles,tile_counter,send_message=True):
    idx_p = bias_path.find('.')
    id_val = sha1(tile).hexdigest()
    out_path = bias_path[:idx_p] + f'_{id_val}.tiff'
    img = Image.fromarray(tile)
    fp = BytesIO()
#    np.save(fp,tile)
    img.save(fp,format='TIFF')
    # reset pointer to beginning  of file
    fp.seek(0)
    s3.Object(out_bucket, out_path).upload_fileobj(fp)
    if send_message:
        message = {
            'Id': f'{id_val}',
            'MessageBody': 'Summed tile',
            'DelaySeconds': 60 if tile_counter>10 else 0,
            'MessageAttributes' : {
                'RawTileBucket': {
                    'StringValue': out_bucket,
                    'DataType': 'String'
                },
                'RawTilePath': {
                    'StringValue': out_path,
                    'DataType': 'String'
                },
                'BiasPath': {
                    'StringValue': bias_path,
                    'DataType': 'String'
                },
                'BiasBucket': {
                    'StringValue': out_bucket,
                    'DataType': 'String'
                },
                'DynamoDBName': {
                    'StringValue': table_name,
                    'DataType': 'String'
                },
                'TotalTiles': {
                    'StringValue': f'{total_tiles}',
                    'DataType': 'Number'
                },
                'TileCounter': {
                    'StringValue': f'{tile_counter}',
                    'DataType': 'Number'
                }
            }
        }
        queue.send_messages(Entries=[message])


def lambda_handler(event, context):
    s3 = boto3.resource("s3")
    dynamodb = boto3.resource("dynamodb")
    sqs = boto3.resource('sqs')
    messages = [i for i in event["Records"]]
    attributes = [i["messageAttributes"]  for i in messages]
    if len(messages) == 1 and int(attributes[0]["TileCounter"]["stringValue"]) > 1:
        print(messages[0])
        return 
    global table_name
    table_name = attributes[0]["DynamoDBName"]["stringValue"]
    table = dynamodb.Table(table_name)

    # check if the tile paths exist on DynamoDB
    messages_to_process = []
    for i,j in zip(attributes,messages):
        response = table.query(
            KeyConditionExpression=Key('tile_name').eq(i["RawTilePath"]["stringValue"])
        )
        if response['Count'] == 0: messages_to_process.append(i)
        else: continue

    if len(messages_to_process) == 0: return


    running_sum = np.zeros((1024,1024))
    tile_counter = 0
    total_tiles = int(attributes[0]["TotalTiles"]["stringValue"])
    out_bucket = attributes[0]["BiasBucket"]["stringValue"]
    out_path = attributes[0]["BiasPath"]["stringValue"]
    queue = sqs.get_queue_by_name(QueueName=table_name)
    for message in messages_to_process:
        tile_count = int(message["TileCounter"]["stringValue"])
        sum_tile(
            s3,
            running_sum,
            message["RawTileBucket"]["stringValue"],
            message["RawTilePath"]["stringValue"],
            delete_file=(tile_count>1)
        )
        table.put_item(
            Item={
                'tile_name': message["RawTilePath"]["stringValue"]
            }
        )
        tile_counter += int(message["TileCounter"]["stringValue"])

    if tile_counter >= total_tiles:
        print(f"FINAL TILE -- tile_counter: {tile_counter}")
        img = Image.fromarray(running_sum)
        fp = BytesIO()
        img.save(fp,format='TIFF')
        # reset pointer to beginning  of file
        fp.seek(0)
        s3.Object(out_bucket, out_path).upload_fileobj(fp)
    elif tile_counter == 1:
        post_tile(s3,queue,running_sum, out_bucket, out_path,total_tiles,tile_counter,send_message=False)
    else:
        post_tile(s3,queue,running_sum, out_bucket, out_path,total_tiles,tile_counter)

   
