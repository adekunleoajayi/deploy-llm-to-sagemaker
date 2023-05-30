import boto3
import ast

def folder_exists_and_not_empty(bucket:str, path:str) -> bool:
    '''
    Folder should exists. 
    Folder should not be empty.
    '''
    s3 = boto3.client('s3')
    if not path.endswith('/'):
        path = path+'/' 
    resp = s3.list_objects(Bucket=bucket, Prefix=path, Delimiter='/',MaxKeys=1)
    return 'Contents' in resp


def evaluate_str(value):
    """
    argparse helper function for evaluating strings
    """
    return ast.literal_eval(value)