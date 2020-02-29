# post_tiles.py

import boto3
from tqdm import tqdm
import os
import json
import argparse
from joblib import Parallel, delayed
import joblib
import multiprocessing
import time
from pathlib import Path

def chunks(l,n):
    for i in range(0, len(l),n):
        yield l[i:i + n]

def get_all_s3_objects(s3, **base_kwargs):
    continuation_token = None
    while True:
        list_kwargs = dict(MaxKeys=1000, **base_kwargs)
        if continuation_token:
            list_kwargs['ContinuationToken'] = continuation_token
        response = s3.list_objects_v2(**list_kwargs)
        yield from response.get('Contents', [])
        if not response.get('IsTruncated'):  # At the end of the list?
            break
        continuation_token = response.get('NextContinuationToken')

def get_list_of_files_to_process(in_bucket_name, prefix, channel):
    session = boto3.Session(profile_name='batch-role')
    s3_client = session.client('s3')
    loc_prefixes = s3_client.list_objects_v2(Bucket=in_bucket_name,Prefix=prefix,Delimiter='CHN')['CommonPrefixes']
    loc_prefixes = [i['Prefix'] + f'0{channel}' for i in loc_prefixes]
    all_files = []
    for i in tqdm(loc_prefixes):
        all_files.extend([f['Key'] for f in get_all_s3_objects(s3_client,Bucket=in_bucket_name,Prefix=i)])
    return all_files

def create_message(in_bucket_name,in_file,id,out_bucket,num_channels,auto_channel):
    head,fname = os.path.split(in_file)
    idx = fname.find('.')
    fname_new = fname[:idx] + '_corrected.tiff'
    # make sure output is in tmp prefix of bucket so it is deleted within 2 days
    # otherwise storage costs might blow up
    out_path = f'tmp/{head}/{fname_new}'
    message = {
        'Id': f'{id}',
        'MessageBody': 'Raw tile',
        'MessageAttributes' : {
            'RawTileBucket': {
                'StringValue': in_bucket_name,
                'DataType': 'String'
            },
            'RawTilePath': {
                'StringValue': in_file,
                'DataType': 'String'
            },
            'OutBucket': {
                'StringValue': out_bucket,
                'DataType': 'String'
            },
            'OutPath': {
                'StringValue': out_path,
                'DataType': 'String'
            },
            'NumChannels': {
                'StringValue': num_channels,
                'DataType': 'String'
            },
            'AutoChannel': {
                'StringValue': auto_channel,
                'DataType': 'String'
            }

        }
    }
    return message

def send_messages(queue,messages):
    print("sending message batch")
    queue.send_messages(Entries=messages)

def send_message_batch(messages):
    session = boto3.session.Session()
    sqs = session.resource('sqs')
    queue = sqs.get_queue_by_name(QueueName=queue_name)
    for i in chunks(messages,10):
        queue.send_messages(Entries=i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_bucket_name', help='Name of input S3 bucket where raw tiles live.', type=str)
    parser.add_argument('--out_bucket_name', help='Name of input S3 bucket where raw tiles live.', type=str)
    parser.add_argument('--in_path', help='Full path  to VW0 directory on S3 bucket.', type=str)
    parser.add_argument('--auto_channel', help='Autofluorescence channel.', type=str)
    parser.add_argument('--num_channels', help='total number of channels acquired.', type=str)
    parser.add_argument('--experiment_name', help='Name of experiment used to name newly created AWS resources for this job.', type=str)

    args = parser.parse_args()

    # file name to save messages to or read from
    fname_messages = f'{args.experiment_name}_{args.auto_channel}.json'

    if not os.path.exists(fname_messages):
        # get list of all tiles to correct for  given channel
        all_files = get_list_of_files_to_process(args.in_bucket_name,args.in_path,args.auto_channel)
        print(f'num files: {len(all_files)}')
        # create one message for  each tile to be  corrected
        ch_messages = [create_message(args.in_bucket_name,f,i,args.out_bucket_name,args.num_channels,args.auto_channel) for i,f in tqdm(enumerate(all_files))]
        # save messages out to json file so dont need to recompute
        json.dump(ch_messages, open(fname_messages,'w'))
    else:
        ch_messages = json.load(open(fname_messages,'r'))
    
    # now post messages to new SQS queue with appropriate name
    sqs = boto3.resource('sqs')
    dead_letter_queue_arn = 'arn:aws:sqs:us-west-2:082194479755:colm_raw_tiles_dead'
    redrive_policy = {
        'deadLetterTargetArn': dead_letter_queue_arn,
        'maxReceiveCount': '5'
    }
    global queue_name
    queue_name = f'{args.experiment_name}_CHN0{args.auto_channel}'
    response_sqs = sqs.create_queue(QueueName=queue_name, Attributes={'DelaySeconds': '0', 'VisibilityTimeout': '2000', 'RedrivePolicy': json.dumps(redrive_policy)})
    acct_id = response_sqs.url.split('/')[-2]
    arn_queue = f'arn:aws:sqs:us-west-2:{acct_id}:{queue_name}'

    # attach our lambda function to this queue
    lambda_client = boto3.client('lambda')
    lambda_client.add_permission(
        FunctionName='colm-tile-correction-dev-hello',
        StatementId=f'{args.experiment_name}_CHN0{args.auto_channel}_{int(time.time())}_FunctionPermission',
        Action='lambda:InvokeFunction',
        Principal='sqs.amazonaws.com',
        SourceArn=arn_queue,
    )
    try:
        lambda_client.create_event_source_mapping(
            EventSourceArn=arn_queue,
            FunctionName='colm-tile-correction-dev-hello',
            BatchSize=5
        )
    except:
        pass


    print(ch_messages[0])
    num_cpus = joblib.cpu_count()
    Parallel(n_jobs=num_cpus)(delayed(send_message_batch)(i) for i in tqdm(chunks(ch_messages,num_cpus), desc='uploading messages to queue...'))

if __name__ == "__main__":
    main()
