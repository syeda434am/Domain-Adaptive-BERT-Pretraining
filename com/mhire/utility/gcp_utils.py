# Function to download from gcp bucket
# Function to upload in gcp bucket

import os
import logging
from google.cloud import storage

# Configure logging
logging.basicConfig(
    filename='./logs/gcp_utils.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GCPUtils:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()

    def download_file(self, gcs_file_path, local_file_path):
        """
        Downloads a file from GCS to a local path.
        """
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(gcs_file_path)
            blob.download_to_filename(local_file_path)
            logging.info(f'Successfully downloaded {gcs_file_path} to {local_file_path}')
        except Exception as e:
            logging.error(f'Failed to download {gcs_file_path}: {str(e)}')
            raise

    def upload_folder(self, local_folder_path, remote_folder_path):
        """
        Uploads a local folder to GCS.
        """
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            for root, _, files in os.walk(local_folder_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    remote_file_path = os.path.join(remote_folder_path, os.path.relpath(local_file_path, local_folder_path))
                    blob = bucket.blob(remote_file_path)
                    blob.upload_from_filename(local_file_path)
                    logging.info(f'Uploaded {local_file_path} to {remote_file_path}')
        except Exception as e:
            logging.error(f'Failed to upload folder {local_folder_path}: {str(e)}')
            raise
