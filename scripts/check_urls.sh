#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

status=0
green='\e[1;32m'; red='\e[1;31m'; cyan='\e[1;36m'; yellow='\e[1;33m'; reset='\e[0m'
user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
last_filepath=

while IFS=: read -r filepath url; do
  if [ "$filepath" != "$last_filepath" ]; then
    printf '\n%s:\n' "$filepath"
    last_filepath=$filepath
  fi
  code=$(curl -gsLm30 --retry 3 --retry-delay 3 --retry-connrefused -o /dev/null -w "%{http_code}" -I "$url") || code=000
  if [ "$code" -lt 200 ] || [ "$code" -ge 400 ]; then
    code=$(curl -gsLm30 --retry 3 --retry-delay 3 --retry-connrefused -o /dev/null -w "%{http_code}" -r 0-0 -A "$user_agent" "$url") || code=000
  fi
  if [ "$code" -lt 200 ] || [ "$code" -ge 400 ]; then
    request_id=$(curl -sS -H 'Accept: application/json' "https://check-host.net/check-http?host=$url&max_nodes=1&node=us3.node.check-host.net" \
      | jq -r .request_id)
    for _ in {1..3}; do
      code=$(curl -sS -H 'Accept: application/json' "https://check-host.net/check-result/$request_id" \
        | jq -r -e '.[][0][3]') || code=000
      [[ "$code" =~ ^[0-9]+$ ]] || code=000
      sleep 3
    done
  fi
  if [ "$code" -lt 200 ] || [ "$code" -ge 400 ]; then
    printf "${red}%s${reset} ${yellow}%s${reset}\n" "$code" "$url" >&2
    status=1
  else
    printf "${green}%s${reset} ${cyan}%s${reset}\n" "$code" "$url"
  fi
done < <(
  git --no-pager grep --no-color -I -P -o \
    '(?<!git\+)(?<!\$\{)https?://(?![^\s<>\")]*[\{\}\$])[^[:space:]<>\")\[\]\(]+' \
    -- '*' \
    ':(exclude).*' \
    ':(exclude)**/.*' \
    ':(exclude)**/*.lock' \
    ':(exclude)**/*.svg' \
    ':(exclude)**/*.xml' \
    ':(exclude)**/*.gradle*' \
    ':(exclude)**/*gradle*' \
    ':(exclude)**/third-party/**' \
  | sed -E 's/[^/[:alnum:]]+$//' \
  | grep -Ev '://(0\.0\.0\.0|127\.0\.0\.1|localhost)([:/])' \
  | grep -Ev 'fwdproxy:8080' \
  || true
)

exit $status
