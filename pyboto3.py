import boto3

s3 = boto3.client("s3", endpoint_url="http://minio-svc:9000",
                  aws_access_key_id='minio',
                  aws_secret_access_key='minio123')
RESPONSE = s3.list_buckets()
for bucket in RESPONSE['Buckets']:
    print(f'{bucket["Name"]}')