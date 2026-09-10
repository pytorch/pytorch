#!/usr/bin/env bash

SCRIPT_DIR=$(dirname "$0")
APP_ID=$1
INSTALL_ID=$2
APP_PRIVATE_KEY=$3
DST_FILE="$4"

ACCESS_TOKEN="$(APP_ID="$(<"${APP_ID}")" INSTALL_ID="$(<"${INSTALL_ID}")" APP_PRIVATE_KEY="$(<"${APP_PRIVATE_KEY}")" "${SCRIPT_DIR}/app_token.sh")"
echo "${ACCESS_TOKEN}" > "${DST_FILE}"
