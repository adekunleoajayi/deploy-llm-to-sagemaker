import ast

import boto3


def check_if_model_tar_exist(bucket: str, path: str) -> bool:
    s3 = boto3.client("s3")
    resp = s3.list_objects(Bucket=bucket, Prefix=path, Delimiter="/", MaxKeys=1)
    return "Contents" in resp


def evaluate_str(value):
    """
    argparse helper function for evaluating strings
    """
    return ast.literal_eval(value)
