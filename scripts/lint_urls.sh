#!/bin/bash

set -euo pipefail

status=0
green='\e[1;32m'; red='\e[1;31m'; cyan='\e[1;36m'; yellow='\e[1;33m'; reset='\e[0m'
user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
max_jobs=10
pids=()

set +e
debug_url='https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201'
echo "=== DEBUG: $debug_url ==="

echo -e "\n-- HEAD (no UA) --"
curl -v -I "$debug_url"
echo "→ HEAD exit $?"

echo -e "\n-- GET byte-0 (with UA) --"
curl -v -r 0-0 -A "$user_agent" -I "$debug_url"
echo "→ byte-0 exit $?"

echo -e "\n-- check-host.net submit --"
submit=$(curl -sS -G \
  -H 'Accept: application/json' \
  --data-urlencode "host=$debug_url" \
  --data-urlencode "max_nodes=1" \
  --data-urlencode "node=us3.node.check-host.net" \
  https://check-host.net/check-http)
echo "submit response: $submit"

request_id=$(printf '%s\n' "$submit" | jq -r .request_id) || request_id=""
echo "→ parsed request_id = '$request_id'"

echo -e "\n-- check-host.net polling results --"
sleep 3
for i in {1..3}; do
  result=$(curl -sS -H 'Accept: application/json' \
    "https://check-host.net/check-result/$request_id")
  code=$(printf '%s\n' "$result" | jq -r '.[ ][0][3] // empty')
  echo "Attempt $i → raw: $result → code: '$code'"
  if [[ "$code" =~ ^[0-9]+$ ]]; then
    echo "→ final code: $code"
    break
  fi
  sleep 3
done

set -euo pipefail
exit 1

running_jobs() {
  jobs -rp | wc -l
}

while IFS=: read -r filepath url; do
  fpath="$filepath"
  (
    code=$(curl -gsLm30 --retry 3 --retry-delay 3 --retry-connrefused -o /dev/null -w "%{http_code}" -I "$url") || code=000
    if [ "$code" -lt 200 ] || [ "$code" -ge 400 ]; then
      code=$(curl -gsLm30 --retry 3 --retry-delay 3 --retry-connrefused -o /dev/null -w "%{http_code}" -r 0-0 -A "$user_agent" "$url") || code=000
    fi
    if [ "$code" -lt 200 ] || [ "$code" -ge 400 ]; then
      request_id=$(curl -sS -G -H 'Accept: application/json' \
        --data-urlencode "host=$url" \
        --data-urlencode "max_nodes=1" \
        --data-urlencode "node=us3.node.check-host.net" \
        https://check-host.net/check-http \
        | jq -r .request_id)
      for _ in {1..3}; do
        new_code=$(curl -sS -H 'Accept: application/json' \
          "https://check-host.net/check-result/${request_id}" \
          | jq -r -e '.[][0][3]') || new_code=000
        [[ "$new_code" =~ ^[0-9]+$ ]] || new_code=000
        if [ "$new_code" -ge 200 ] && [ "$new_code" -lt 400 ]; then
          code=$new_code
          break
        fi
        sleep 3
      done
    fi
    if [ "$code" -lt 200 ] || [ "$code" -ge 400 ]; then
      printf "${red}%s${reset} ${yellow}%s${reset} %s\n" "$code" "$url" "$fpath" >&2
      exit 1
    else
      printf "${green}%s${reset} ${cyan}%s${reset} %s\n" "$code" "$url" "$fpath"
      exit 0
    fi
  ) &
  pids+=($!)
  while [ "$(running_jobs)" -ge "$max_jobs" ]; do
    sleep 1
  done
done < <(
  git --no-pager grep --no-color -I -P -o \
    '(?!.*@lint-ignore)(?<!git\+)(?<!\$\{)https?://(?![^\s<>\")]*[<>\{\}\$])[^[:space:]<>\")\[\]\(\\]+' \
    -- '*' \
    ':(exclude).*' \
    ':(exclude,glob)**/.*' \
    ':(exclude,glob)**/*.lock' \
    ':(exclude,glob)**/*.svg' \
    ':(exclude,glob)**/*.xml' \
    ':(exclude,glob)**/*.gradle*' \
    ':(exclude,glob)**/*gradle*' \
    ':(exclude,glob)**/third-party/**' \
    ':(exclude,glob)**/third_party/**' \
  | sed -E 's/[^/[:alnum:]]+$//' \
  | grep -Ev '://(0\.0\.0\.0|127\.0\.0\.1|localhost)([:/])' \
  | grep -Ev 'fwdproxy:8080' \
  || true
)

for pid in "${pids[@]}"; do
  if ! wait "$pid" 2>/dev/null; then
    status=1
  fi
done

exit $status
