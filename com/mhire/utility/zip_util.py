# To zip/unzip model files
import os
import zipfile
import logging

# Configure logging
logging.basicConfig(
    filename='./logs/zip_utils.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ZipUtils:
    """
    Utility class for zipping and unzipping files and directories.
    """

    @staticmethod
    def zip_folder(folder_path, zip_file_path):
        """
        Zips the contents of a folder.

        Args:
            folder_path (str): Path to the folder to zip.
            zip_file_path (str): Path where the zip file will be saved.

        Raises:
            Exception: If there is an issue zipping the folder.
        """
        try:
            logging.info(f"Starting to zip folder: {folder_path} into {zip_file_path}")
            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, folder_path)  # Maintain folder structure
                        zipf.write(file_path, arcname)
                        logging.debug(f"Zipped file: {file_path} as {arcname}")
            logging.info(f"Folder successfully zipped to {zip_file_path}")
        except Exception as e:
            logging.error(f"Failed to zip folder {folder_path}: {str(e)}")
            raise

    @staticmethod
    def unzip_file(zip_file_path, extract_to_folder):
        """
        Unzips a zip file into a specified folder.

        Args:
            zip_file_path (str): Path to the zip file.
            extract_to_folder (str): Directory where the contents will be extracted.

        Raises:
            Exception: If there is an issue unzipping the file.
        """
        try:
            logging.info(f"Starting to unzip file: {zip_file_path} into {extract_to_folder}")
            with zipfile.ZipFile(zip_file_path, 'r') as zipf:
                zipf.extractall(extract_to_folder)
                logging.info(f"Successfully extracted files to {extract_to_folder}")
        except Exception as e:
            logging.error(f"Failed to unzip file {zip_file_path}: {str(e)}")
            raise
