#!/bin/bash

set -euo pipefail

trap 'kill 0' SIGINT

status=0
green='\e[1;32m'; red='\e[1;31m'; cyan='\e[1;36m'; yellow='\e[1;33m'; reset='\e[0m'
user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
max_jobs=10
pids=()

running_jobs() {
  jobs -rp | wc -l
}

while IFS=: read -r filepath url; do
  (
    code=$(curl -k -gsLm30 --retry 3 --retry-delay 3 --retry-connrefused -o /dev/null -w "%{http_code}" -I "$url") || code=000
    if [ "$code" -lt 200 ] || [ "$code" -ge 400 ]; then
      sleep 1
      code=$(curl -k -gsLm30 --retry 3 --retry-delay 3 --retry-connrefused -o /dev/null -w "%{http_code}" -r 0-0 -A "$user_agent" "$url") || code=000
    fi
    if [ "$code" -lt 200 ] || [ "$code" -ge 400 ]; then
      sleep 1
      request_id=$(curl -sS -G -H 'Accept: application/json' \
        --data-urlencode "host=$url" \
        --data-urlencode "max_nodes=1" \
        --data-urlencode "node=us3.node.check-host.net" \
        https://check-host.net/check-http \
        | jq -r .request_id) || request_id=""
      if [ -n "$request_id" ]; then
        sleep 5
        for _ in {1..5}; do
          new_code=$(curl -sS -H 'Accept: application/json' \
            "https://check-host.net/check-result/$request_id" \
            | jq -r -e '.[][0][3]') || new_code=000
          [[ "$new_code" =~ ^[0-9]+$ ]] || new_code=000
          if [ "$new_code" -ge 200 ] && [ "$new_code" -lt 400 ]; then
            code=$new_code
            break
          fi
          sleep 5
        done
      fi
    fi
    # Treat Cloudflare JS-challenge and rate-limit as success.
    if [[ "$code" == "403" || "$code" == "429" || "$code" == "503" ]]; then
      printf "${yellow}WARN %s${reset} ${cyan}%s${reset} %s\n" "$code" "$url" "$filepath"
      exit 0
    fi
    if [ "$code" -lt 200 ] || [ "$code" -ge 400 ]; then
      printf "${red}FAIL %s${reset} ${yellow}%s${reset} %s\n" "$code" "$url" "$filepath" >&2
      exit 1
    else
      printf "${green} OK  %s${reset} ${cyan}%s${reset} %s\n" "$code" "$url" "$filepath"
      exit 0
    fi
  ) &
  pids+=($!)
  while [ "$(running_jobs)" -ge "$max_jobs" ]; do
    sleep 1
  done
 done < <(
  pattern='(?!.*@lint-ignore)(?<!git\+)(?<!\$\{)https?://(?![^/]*@)(?![^\s<>\")]*[<>\{\}\$])[^[:space:]<>")\[\]\\|]+'
  excludes=(
    ':(exclude,glob)**/.*'
    ':(exclude,glob)**/*.lock'
    ':(exclude,glob)**/*.svg'
    ':(exclude,glob)**/*.xml'
    ':(exclude,glob)**/*.gradle*'
    ':(exclude,glob)**/*gradle*'
    ':(exclude,glob)**/third-party/**'
    ':(exclude,glob)**/third_party/**'
  )
  if [ $# -gt 0 ]; then
    paths=("$@")
  else
    paths=('*')
  fi
  git --no-pager grep --no-color -I -P -o "$pattern" -- "${paths[@]}" "${excludes[@]}" \
  | sed -E 's/[^/[:alnum:]]+$//' \
  | grep -Ev '://(0\.0\.0\.0|127\.0\.0\.1|localhost)([:/])' \
  | grep -Ev '://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' \
  | grep -Ev 'fwdproxy:8080' \
  || true
)

for pid in "${pids[@]}"; do
  wait "$pid" 2>/dev/null || {
    case $? in
      1) status=1 ;;
      127) ;;  # ignore "not a child" noise
      *) exit $? ;;
    esac
  }
done

exit $status
