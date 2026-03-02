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
    for pkg in ${PKG_DIR}/*.${extension}; do
      (
        set -x
        shm_id=$(sha256sum "${pkg}" | awk '{print $1}')
        ${AWS_S3_CP} --no-progress --acl public-read "${pkg}" "${s3_upload_dir}" \
          --metadata "checksum-sha256=${shm_id}"
      )
    done
  )
}

R2_UPLOAD=${R2_UPLOAD:-}
R2_BUCKET="s3://pytorch-downloads"
R2_ACCOUNT_ID=${R2_ACCOUNT_ID:-}
R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID:-}
R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY:-}

r2_upload() {
  if [[ -z "${R2_ACCOUNT_ID}" || -z "${R2_ACCESS_KEY_ID}" || -z "${R2_SECRET_ACCESS_KEY}" ]]; then
    echo "WARNING: R2 credentials not configured, skipping R2 upload"
    return
  fi
  local extension
  local pkg_type
  extension="$1"
  pkg_type="$2"
  r2_root_dir="${R2_BUCKET}/${pkg_type}/${UPLOAD_CHANNEL}"
  if [[ -z ${UPLOAD_SUBFOLDER:-} ]]; then
    r2_upload_dir="${r2_root_dir}/"
  else
    r2_upload_dir="${r2_root_dir}/${UPLOAD_SUBFOLDER}/"
  fi
  (
    for pkg in ${PKG_DIR}/*.${extension}; do
      (
        set -x
        shm_id=$(sha256sum "${pkg}" | awk '{print $1}')
        AWS_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}" \
        AWS_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}" \
        AWS_SESSION_TOKEN="" \
        AWS_DEFAULT_REGION="auto" \
        ${AWS_S3_CP} --no-progress "${pkg}" "${r2_upload_dir}" \
          --metadata "checksum-sha256=${shm_id}" \
          --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
      )
    done
  )
}

# Install dependencies (should be a no-op if previously installed)
pip install -q awscli uv

case "${PACKAGE_TYPE}" in
  libtorch)
    s3_upload "zip" "libtorch"
    if [[ "${R2_UPLOAD}" == "true" ]]; then
      r2_upload "zip" "libtorch"
    fi
    BACKUP_DIR="libtorch/${UPLOAD_CHANNEL}/${UPLOAD_SUBFOLDER}"
    ;;
  # wheel can either refer to wheel/manywheel
  *wheel)
    s3_upload "whl" "whl"
    if [[ "${R2_UPLOAD}" == "true" ]]; then
      r2_upload "whl" "whl"
    fi
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
