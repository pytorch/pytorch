#!/usr/bin/env bash

TOKEN_FILE=$1
OUTPUT_FILE=$2

echo "Starting gh_cat_token.sh with TOKEN_FILE=${TOKEN_FILE}, OUTPUT_FILE=${OUTPUT_FILE}"

# Validate inputs
if [[ ! -r "${TOKEN_FILE}" ]]; then
    echo "Error: Token file '${TOKEN_FILE}' does not exist or is not readable."
    exit 1
fi

# Write the token to the output file
cat "${TOKEN_FILE}" > "${OUTPUT_FILE}"
echo "Token written to ${OUTPUT_FILE}"
