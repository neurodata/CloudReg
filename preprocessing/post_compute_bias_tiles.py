import boto3
from tqdm import tqdm
import os
import json
import argparse
from joblib import Parallel, delayed
import joblib
import multiprocessing
import time

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

def create_message(in_bucket_name,in_file,bias_bucket,bias_path,id,out_bucket,total_tiles):
    global queue_name
    head,fname = os.path.split(in_file)
    idx = fname.find('.')
    fname_new = fname[:idx] + '_corrected.tif'
    out_path = f'{head}/{fname_new}'
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
            'BiasPath': {
                'StringValue': bias_path,
                'DataType': 'String'
            },
            'BiasBucket': {
                'StringValue': bias_bucket,
                'DataType': 'String'
            },
            'DynamoDBName': {
                'StringValue': queue_name,
                'DataType': 'String'
            },
            'TotalTiles': {
                'StringValue': f'{total_tiles}',
                'DataType': 'Number'
            },
            'TileCounter': {
                'StringValue': '1',
                'DataType': 'Number'
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

def create_cloudwatch_alarm():
    # Create CloudWatch client
    cloudwatch = boto3.client('cloudwatch')
    
    # Create alarm
    cloudwatch.put_metric_alarm(
        AlarmName=f'Bias_computed_{queue_name}',
        ComparisonOperator='LessThanThreshold',
        EvaluationPeriods=1,
        MetricName='ApproximateNumberOfMessagesVisible',
        Namespace='AWS/SQS',
        Period=60,
        Statistic='Average',
        Threshold=1.0,
        AlarmDescription='Alarm when compute bias queue is empty.',
        AlarmActions=[
            'arn:aws:sns:us-west-2:082194479755:COLM_bias_computation'
        ],
        Dimensions=[
            {
              'Name': 'QueueName',
              'Value': queue_name
            },
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_bucket_name', help='Name of input S3 bucket where raw tiles live.', type=str)
    parser.add_argument('--out_bucket_name', help='Name of input S3 bucket where raw tiles live.', type=str)
    parser.add_argument('--bias_bucket_name', help='Name of S3 bucket where bias correction tile lives.', type=str)
    parser.add_argument('--in_path', help='Full path  to VW0 directory on S3 bucket.', type=str)
    parser.add_argument('--bias_path', help='Full path  to bias file for given channel.', type=str)
    parser.add_argument('--channel', help='Channel number to process. accepted values are 0, 1, or 2', type=str)
    parser.add_argument('--experiment_name', help='Name of experiment used to name newly created AWS resources for this job.', type=str)
    parser.add_argument('--subsample_factor', help='Factor to subsample the tiles by to compute the bias. Default is subsample by 10 which means every 10th tile  will be used.', type=int, default=10)

    args = parser.parse_args()

    # file name to save messages to or read from
    fname_messages = f'./{args.experiment_name}_{args.channel}_compute_bias.json'
    global queue_name
    queue_name = f'{args.experiment_name}_CHN0{args.channel}_compute_bias'

    # now post messages to new SQS queue with appropriate name
    sqs = boto3.resource('sqs')
    dead_letter_queue_arn = 'arn:aws:sqs:us-west-2:082194479755:colm_raw_tiles_dead'
    redrive_policy = {
        'deadLetterTargetArn': dead_letter_queue_arn,
        'maxReceiveCount': '5'
    }
    response_sqs = sqs.create_queue(QueueName=queue_name, Attributes={'DelaySeconds': '0', 'VisibilityTimeout': '2000', 'RedrivePolicy': json.dumps(redrive_policy)})
    acct_id = response_sqs.url.split('/')[-2]
    arn_queue = f'arn:aws:sqs:us-west-2:{acct_id}:{queue_name}'

    # attach our lambda function to this queue
    lambda_client = boto3.client('lambda')
    lambda_client.add_permission(
        FunctionName='compute_colm_bias',
        StatementId=f'{args.experiment_name}_CHN0{args.channel}_compute_bias_{int(time.time())}_FunctionPermission',
        Action='lambda:InvokeFunction',
        Principal='sqs.amazonaws.com',
        SourceArn=arn_queue
    )
    try:
        lambda_client.create_event_source_mapping(
            EventSourceArn=arn_queue,
            FunctionName='compute_colm_bias'
        )
    except:
        pass

    # create a DynamoDB to keep  track of tiles that have been processed
    dynamodb = boto3.resource('dynamodb')
    try:
        table = dynamodb.create_table(
            TableName=queue_name,
            BillingMode='PAY_PER_REQUEST',
            KeySchema=[
                {
                    'AttributeName': 'tile_name',
                    'KeyType': 'HASH'
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'tile_name',
                    'AttributeType': 'S'
                }
            ]
        )
        # Wait until the table exists.
        table.meta.client.get_waiter('table_exists').wait(TableName=queue_name)
    except:
        pass

    

    if not os.path.exists(fname_messages):
        # get list of all tiles to correct for  given channel
        all_files = get_list_of_files_to_process(args.in_bucket_name,args.in_path,args.channel)
        # subsample tiles
        all_files = all_files[::args.subsample_factor]
        print(f'num files: {len(all_files)}')
        # create one message for  each tile to be  corrected
        ch_messages = [create_message(args.in_bucket_name,f,args.bias_bucket_name,args.bias_path,i,args.out_bucket_name,len(all_files)) for i,f in tqdm(enumerate(all_files))]
        # save messages out to json file so dont need to recompute
        json.dump(ch_messages, open(fname_messages,'w'))
    else:
        ch_messages = json.load(open(fname_messages,'r'))
    


    num_cpus = joblib.cpu_count()
    Parallel(n_jobs=num_cpus)(delayed(send_message_batch)(i) for i in tqdm(chunks(ch_messages,num_cpus), desc='uploading messages to queue...'))

if __name__ == "__main__":
    main()
