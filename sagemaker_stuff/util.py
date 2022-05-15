
from pathlib import Path
import os
import glob
import boto3
import shutil

from aws_utils.s3_sagemaker_utils import S3SageMakerUtils

from . import config


def copy_file_to_tar_dir(filepath):
    shutil.copyfile(
        filepath,
        os.path.join(config.DIR_OUTPUT, os.path.basename(filepath))
    )


def download_file_from_s3(s3_bucket, s3_path, destination, print_status=False):
    if print_status:
        print(f"Downloading File {s3_path} from S3 bucket {s3_bucket}")
    try:
        aws_session = S3SageMakerUtils()
        aws_session.download_data(destination, s3_bucket,
                                  s3_path)
        if print_status:
            print(
                f"Downloading File {s3_path} from S3 bucket {s3_bucket} --> Success")
    except:
        if print_status:
            print(
                f"Downloading File {s3_path} from S3 bucket {s3_bucket} --> Failure")


def upload_dir(localDir, awsInitDir, bucketName, tag, prefix='/'):
    """
    from current working directory, upload a 'localDir' with all its subcontents (files and subdirectories...)
    to a aws bucket
    Parameters
    ----------
    localDir :   localDirectory to be uploaded, with respect to current working directory
    awsInitDir : prefix 'directory' in aws
    bucketName : bucket in aws
    tag :        tag to select files, like *png
                 NOTE: if you use tag it must be given like --tag '*txt', in some quotation marks... for argparse
    prefix :     to remove initial '/' from file names

    Returns
    -------
    None
    """
    sesh = boto3.session.Session()
    s3 = sesh.resource('s3')
    cwd = str(Path.cwd())
    p = Path(os.path.join(Path.cwd(), localDir))
    mydirs = list(p.glob('**'))
    for mydir in mydirs:
        fileNames = glob.glob(os.path.join(mydir, tag))
        fileNames = [f for f in fileNames if not Path(f).is_dir()]
        rows = len(fileNames)
        for i, fileName in enumerate(fileNames):
            fileName = str(fileName).replace(cwd, '')
            # only modify the text if it starts with the prefix
            if fileName.startswith(prefix):
                # remove one instance of prefix
                fileName = fileName.replace(prefix, "", 1)
            print(f"fileName {fileName}")

            awsPath = os.path.join(awsInitDir, str(fileName))
            s3.meta.client.upload_file(fileName, bucketName, awsPath)
