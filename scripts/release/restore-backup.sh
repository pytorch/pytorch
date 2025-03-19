#!/usr/bin/env bash

set -eou pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${DIR}/promote/common_utils.sh"

if [[ -z "${RESTORE_FROM:-}" ]]; then
    echo "ERROR: RESTORE_FROM environment variable must be specified"
    echo "       example: RESTORE_FROM=v1.6.0-rc3 ${0}"
    exit 1
fi

DRY_RUN=${DRY_RUN:-enabled}

PYTORCH_S3_BACKUP_BUCKET=${PYTORCH_S3_BACKUP_BUCKET:-s3://pytorch-backup/${RESTORE_FROM}}
PYTORCH_S3_TEST_BUCKET=${PYTORCH_S3_TEST_BUCKET:-s3://pytorch/}
PYTORCH_S3_FROM=${PYTORCH_S3_FROM:-${PYTORCH_S3_BACKUP_BUCKET}}
PYTORCH_S3_TO=${PYTORCH_S3_TO:-s3://pytorch/}

restore_wheels() {
    aws_promote torch whl
}

restore_libtorch() {
    aws_promote libtorch-* libtorch
}


restore_wheels
restore_libtorch
