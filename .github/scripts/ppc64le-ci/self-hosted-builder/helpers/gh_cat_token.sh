#!/usr/bin/env bash

TOKEN_FILE=$1
TOKEN_PIPE=$2

echo "Starting gh_cat_token.sh with TOKEN_FILE=${TOKEN_FILE}, TOKEN_PIPE=${TOKEN_PIPE}"

# Validate inputs
if [[ ! -r "${TOKEN_FILE}" ]]; then
    echo "Error: Token file '${TOKEN_FILE}' does not exist or is not readable."
    exit 1
fi

rm "${TOKEN_PIPE}" 2>/dev/null ||:
mkfifo "${TOKEN_PIPE}"
cat "${TOKEN_FILE}" > "${TOKEN_PIPE}" &