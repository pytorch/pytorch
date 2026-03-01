#!/usr/bin/env bash
#
# Request an ACCESS_TOKEN to be used by a GitHub APP
# Environment variable that need to be set up:
# * APP_ID, the GitHub's app ID
# * INSTALL_ID, the Github's app's installation ID
# * APP_PRIVATE_KEY, the content of GitHub app's private key in PEM format.
#
# https://github.com/orgs/community/discussions/24743#discussioncomment-3245300
#

set -o pipefail

set -e  # Exit on error

# Generate JWT
header='{"alg":"RS256","typ":"JWT"}'
payload="{\"iat\":$(date +%s),\"exp\":$(( $(date +%s) + 600 )),\"iss\":${APP_ID}}"

header_base64=$(echo -n "$header" | openssl base64 | tr -d '=' | tr '/+' '_-' | tr -d '\n')
payload_base64=$(echo -n "$payload" | openssl base64 | tr -d '=' | tr '/+' '_-' | tr -d '\n')

signature=$(echo -n "${header_base64}.${payload_base64}" | \
  openssl dgst -sha256 -sign "${APP_PRIVATE_KEY}" | \
  openssl base64 | tr -d '=' | tr '/+' '_-' | tr -d '\n')

generated_jwt="${header_base64}.${payload_base64}.${signature}"

API_VERSION=v3
API_HEADER="Accept: application/vnd.github+json"

auth_header="Authorization: Bearer ${generated_jwt}"

app_installations_response=$(curl -sX POST \
        -H "${auth_header}" \
        -H "${API_HEADER}" \
        --url "https://api.github.com/app/installations/${INSTALL_ID}/access_tokens" \
    )

echo "$app_installations_response" | jq --raw-output '.token'
