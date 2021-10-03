#!/usr/bin/env bash

set -euo pipefail

PACKAGE_TYPE=${PACKAGE_TYPE:-conda}

PKG_DIR=${PKG_DIR:-/tmp/workspace/final_pkgs}

# Designates whether to submit as a release candidate or a nightly build
# Value should be `test` when uploading release candidates
# currently set within `designate_upload_channel`
UPLOAD_CHANNEL=${UPLOAD_CHANNEL:-nightly}
# Designates what subfolder to put packages into
UPLOAD_SUBFOLDER=${UPLOAD_SUBFOLDER:-cpu}
UPLOAD_BUCKET="s3://pytorch"
BACKUP_BUCKET="s3://pytorch-backup"

DRY_RUN=${DRY_RUN:-enabled}
# Don't actually do work unless explicit
ANACONDA="true anaconda"
AWS_S3_CP="aws s3 cp --dryrun"
if [[ "${DRY_RUN}" = "disabled" ]]; then
  ANACONDA="anaconda"
  AWS_S3_CP="aws s3 cp"
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

conda_upload() {
  (
    set -x
    ${ANACONDA} \
      upload  \
      ${PKG_DIR}/*.tar.bz2 \
      -u "pytorch-${UPLOAD_CHANNEL}" \
      --label main \
      --no-progress \
      --force
  )
}

s3_upload() {
  local extension
  local pkg_type
  extension="$1"
  pkg_type="$2"
  s3_dir="${UPLOAD_BUCKET}/${pkg_type}/${UPLOAD_CHANNEL}/${UPLOAD_SUBFOLDER}/"
  (
    for pkg in ${PKG_DIR}/*.${extension}; do
      (
        set -x
        ${AWS_S3_CP} --no-progress --acl public-read "${pkg}" "${s3_dir}"
      )
    done
  )
}

case "${PACKAGE_TYPE}" in
  conda)
    conda_upload
    # Fetch  platform (eg. win-64, linux-64, etc.) from index file
    # Because there's no actual conda command to read this
    subdir=$(\
      tar -xOf ${PKG_DIR}/*.bz2 info/index.json \
        | grep subdir  \
        | cut -d ':' -f2 \
        | sed -e 's/[[:space:]]//' -e 's/"//g' -e 's/,//' \
    )
    BACKUP_DIR="conda/${subdir}"
    ;;
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
