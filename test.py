import logging
from boto3 import client

# Configure logging
logging.basicConfig(level=logging.INFO)

# S3 configuration
S3_BUCKET_NAME = "cdn"  # Your existing bucket
S3_KEY = "test/my-file.txt"  # The path/name in S3
FILE_PATH = "testfile.txt"  # Your local file

# Create S3 client with Hetzner's configuration
s3 = client(
    "s3",
    region_name="us-east-1",
    endpoint_url="https://f608bb774d91b3129540d98cc2871709.r2.cloudflarestorage.com",
    aws_access_key_id="64677b60acaba6176223b18054b79c14",
    aws_secret_access_key="d5eaa720283799632fc79f673db8f82aa1f0c7cabc2d259276c0375e39182534",
)

def upload_file(file_path, bucket, key):
    """Upload a file to S3"""
    try:
        # Read file content
        with open(file_path, 'rb') as file:
            file_content = file.read()

        # Upload file using put_object
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=file_content
        )
        
        logging.info(f"Successfully uploaded {file_path} to {bucket}/{key}")
        
    except Exception as e:
        logging.error(f"Upload failed: {str(e)}")
        raise

# Upload the file
upload_file(FILE_PATH, S3_BUCKET_NAME, S3_KEY)