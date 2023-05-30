import argparse
import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.s3 import S3Uploader

import config
import helper

logger = logging.getLogger(__name__)


def download_model_snapshot_from_huggingface(model_id, force_model_download):
    # - download Model
    model_tar_dir = Path(f'{config.MODEL_DIR}/{model_id.split("/")[-1]}')

    if force_model_download and model_tar_dir.exists():
        shutil.rmtree(str(model_tar_dir))
        model_tar_dir.mkdir(exist_ok=True)

        # Download model from Hugging Face into model_dir
        logger.info("Downloading model fom HuggingFace")
        snapshot_download(model_id, local_dir=str(model_tar_dir), local_dir_use_symlinks=False)

    elif model_tar_dir.exists():
        logger.info(f"{model_tar_dir} is already downloaded")

    else:
        os.makedirs(str(model_tar_dir), exist_ok=True)

        # Download model from Hugging Face into model_dir
        logger.info("Downloading model fom HuggingFace")
        snapshot_download(model_id, local_dir=str(model_tar_dir), local_dir_use_symlinks=False)

    return model_tar_dir


def compress_model_folder(model_tar_dir):
    # - compress model folder
    parent_dir = os.getcwd()

    # change to model dir
    os.chdir(str(model_tar_dir))

    logger.info("Compressing model folder")
    # use pigz for faster and parallel compression
    os.system("tar -cf model.tar.gz --use-compress-program=pigz *")

    # change back to parent dir
    os.chdir(parent_dir)


def move_compress_model_folder_to_s3(model_tar_dir):
    # - upload model folder to S3
    logger.info("Uploading compressed model folder to S3")
    s3_model_uri = S3Uploader.upload(
        local_path=str(model_tar_dir.joinpath("model.tar.gz")),
        desired_s3_uri=f"s3://{config.SESS.default_bucket()}/{model_tar_dir}",
    )
    logging.info(f"Model uploaded to: {s3_model_uri}")
    return s3_model_uri


def deploy_model_to_sagemaker(
    s3_model_uri,
    model_server_workers=1,
    initial_instance_count=1,
    instance_type="ml.g5.4xlarge",
):
    # - deploy model
    huggingface_model = HuggingFaceModel(
        model_data=s3_model_uri,
        role=config.ROLE,
        transformers_version=config.TRANSFORMERS_VERSION,
        pytorch_version=config.PYTORCH_VERSION,
        py_version=config.PY_VERSION,
        model_server_workers=model_server_workers,
    )

    logger.info("Deploying model to Sagemaker")
    huggingface_model.deploy(
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        endpoint_name=f"{config.ORG_NAME}-{config.MODEL_ID.split('/')[-1]}-endpoint",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # -
    parser.add_argument("--model-server-workers", type=int, help="model_server_workers", default=1)

    # -
    parser.add_argument(
        "--initial-instance-count", type=int, help="initial_instance_count", default=1
    )
    parser.add_argument("--instance-type", type=str, help="instance_type", default="ml.g5.4xlarge")

    parser.add_argument(
        "--force-model-download",
        type=helper.evaluate_str,
        help="force model download",
        default=False,
    )

    parser.add_argument(
        "--force-compress",
        type=helper.evaluate_str,
        help="force compress",
        default=False,
    )

    parser.add_argument(
        "--force-move-model-to-s3",
        type=helper.evaluate_str,
        help="force_move_model_to_s3",
        default=False,
    )

    args = parser.parse_args()

    logger.info("Arguments parsed")
    logger.info(f"{args}")

    # - download model from hugging face
    model_tar_dir = download_model_snapshot_from_huggingface(
        config.MODEL_ID, args.force_model_download
    )

    # - copy custom code to model folder
    logger.info(f"Copy code folder into {model_tar_dir} folder")
    shutil.copytree("code", f"{model_tar_dir}/code", dirs_exist_ok=True)

    # - compress mode folder
    if args.force_compress or not os.path.exists(f"{model_tar_dir}/model.tar.gz"):
        compress_model_folder(model_tar_dir)
    else:
        logger.info(f"{model_tar_dir}/model.tar.gz a;ready exist")

    # - upload compress folder to S3
    if args.force_move_model_to_s3 or not helper.folder_exists_and_not_empty(
        config.SESS.default_bucket(), str(model_tar_dir)
    ):
        s3_model_uri = move_compress_model_folder_to_s3(model_tar_dir)
    else:
        logger.info(f"{model_tar_dir} exist on {config.SESS.default_bucket()}")
        s3_model_uri = f"s3://{config.SESS.default_bucket()}/{model_tar_dir}/model.tar.gz"

    # - deply model to sagemaker
    deploy_model_to_sagemaker(
        s3_model_uri,
        model_server_workers=args.model_server_workers,
        initial_instance_count=args.initial_instance_count,
        instance_type=args.instance_type,
    )
