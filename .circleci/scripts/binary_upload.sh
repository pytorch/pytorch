#!/usr/bin/env bash

set -euo pipefail

PACKAGE_TYPE=${PACKAGE_TYPE:-wheel}

PKG_DIR=${PKG_DIR:-/tmp/workspace/final_pkgs}

# Designates whether to submit as a release candidate or a nightly build
# Value should be `test` when uploading release candidates
# currently set within `designate_upload_channel`
UPLOAD_CHANNEL=${UPLOAD_CHANNEL:-nightly}
# Designates what subfolder to put packages into
UPLOAD_SUBFOLDER=${UPLOAD_SUBFOLDER:-}
UPLOAD_BUCKET="s3://pytorch"
BACKUP_BUCKET="s3://pytorch-backup"
BUILD_NAME=${BUILD_NAME:-}

DRY_RUN=${DRY_RUN:-enabled}
# Don't actually do work unless explicit
AWS_S3_CP="aws s3 cp --dryrun"
if [[ "${DRY_RUN}" = "disabled" ]]; then
  AWS_S3_CP="aws s3 cp"
fi

# this is special build with all dependencies packaged
if [[ ${BUILD_NAME} == *-full* ]]; then
  UPLOAD_SUBFOLDER="${UPLOAD_SUBFOLDER}_full"
fi


do_backup() {
  local backup_dir
  backup_dir=$1
  (
    pushd /tmp/workspace
    set -x
    ${AWS_S3_CP} --recursive . "${BACKUP_BUCKET}/${CIRCLE_TAG}/${backup_dir}/"
  )
}

s3_upload() {
  local extension
  local pkg_type
  extension="$1"
  pkg_type="$2"
  s3_root_dir="${UPLOAD_BUCKET}/${pkg_type}/${UPLOAD_CHANNEL}"
  if [[ -z ${UPLOAD_SUBFOLDER:-} ]]; then
    s3_upload_dir="${s3_root_dir}/"
  else
    s3_upload_dir="${s3_root_dir}/${UPLOAD_SUBFOLDER}/"
  fi
  (
    cache_control_flag=""
    if [[ "${UPLOAD_CHANNEL}" = "test" ]]; then
      cache_control_flag="--cache-control='no-cache,no-store,must-revalidate'"
    fi
    for pkg in ${PKG_DIR}/*.${extension}; do
      (
        set -x
        shm_id=$(sha256sum "${pkg}" | awk '{print $1}')
        ${AWS_S3_CP} --no-progress --acl public-read "${pkg}" "${s3_upload_dir}" \
          --metadata "checksum-sha256=${shm_id}" ${cache_control_flag}
      )
    done
  )
}

# Install dependencies (should be a no-op if previously installed)
pip install -q awscli uv

case "${PACKAGE_TYPE}" in
  libtorch)
    s3_upload "zip" "libtorch"
    BACKUP_DIR="libtorch/${UPLOAD_CHANNEL}/${UPLOAD_SUBFOLDER}"
    ;;
  # wheel can either refer to wheel/manywheel
  *wheel)
    s3_upload "whl" "whl"
    BACKUP_DIR="whl/${UPLOAD_CHANNEL}/${UPLOAD_SUBFOLDER}"
    ;;
  *)
    echo "ERROR: unknown package type: ${PACKAGE_TYPE}"
    exit 1
    ;;
esac

# CIRCLE_TAG is defined by upstream circleci,
# this can be changed to recognize tagged versions
if [[ -n "${CIRCLE_TAG:-}" ]]; then
  do_backup "${BACKUP_DIR}"
fi
