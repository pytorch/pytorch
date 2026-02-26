#!/usr/bin/env bash

# Upload a binary to a bucket, supports dry-run mode

set -euo pipefail

# Optional inputs. By default upload to s3://ossci-linux
TARGET_OS=${TARGET_OS:-linux}
UPLOAD_BUCKET=${UPLOAD_BUCKET:-s3://ossci-${TARGET_OS}}
UPLOAD_SUBFOLDER=${UPLOAD_SUBFOLDER:-}

# Download to ${{ runner.temp }}/artifacts to match the default
PKG_DIR=${PKG_DIR:-/tmp/workspace/artifacts}

# Optional package include.
# By default looks for and uploads *.tar.bz2 files only
PKG_INCLUDE=${PKG_INCLUDE:-'*.tar.bz2'}

# Dry-run logs the upload command without actually executing it
# Dry-run is enabled by default, it has to be disabled to upload
DRY_RUN=${DRY_RUN:-enabled}
# Don't actually do work unless explicit
AWS_S3_CP="aws s3 cp --dryrun"
if [[ "${DRY_RUN}" = "disabled" ]]; then
  AWS_S3_CP="aws s3 cp"
fi

# Install dependencies (should be a no-op if previously installed)
pip install -q awscli

# Handle subfolders, if provided
s3_root_dir="${UPLOAD_BUCKET}"
if [[ -z ${UPLOAD_SUBFOLDER:-} ]]; then
    s3_upload_dir="${s3_root_dir}/"
else
    s3_upload_dir="${s3_root_dir}/${UPLOAD_SUBFOLDER}/"
fi

# Upload all packages that match PKG_INCLUDE within PKG_DIR and subdirs
set -x
${AWS_S3_CP} --no-progress --acl public-read --exclude="*" --include="${PKG_INCLUDE}" --recursive "${PKG_DIR}" "${s3_upload_dir}"
