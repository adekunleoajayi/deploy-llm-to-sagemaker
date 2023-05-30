import logging
import os

import boto3
import sagemaker

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(process)d -- [%(threadName)s] %(name)s : %(message)s",
)
logging.getLogger().setLevel("INFO")

# Set HF cache location and ensure that hf-transfer is used for faster download
os.environ["HF_HOME"] = "data"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

RoleName = "sagemaker_execution_rule"
IAM = boto3.client("iam")
ROLE = IAM.get_role(RoleName=RoleName)["Role"]["Arn"]
SESS = sagemaker.Session()

ORG_NAME = "OrgName"
MODEL_DIR = "hf-models"
MODEL_ID = "databricks/dolly-v2-12b"

# - HuggingFace Env Specification
TRANSFORMERS_VERSION = "4.26"
PYTORCH_VERSION = "1.13"
PY_VERSION = "py39"

logger.info(f"sagemaker role arn: {ROLE}")
logger.info(f"sagemaker bucket: {SESS.default_bucket()}")
logger.info(f"sagemaker session region: {SESS.boto_region_name}")
