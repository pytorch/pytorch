#!/bin/bash

set -euo pipefail

status=0
green='\e[1;32m'; red='\e[1;31m'; cyan='\e[1;36m'; yellow='\e[1;33m'; reset='\e[0m'
last_filepath=

while IFS=: read -r filepath link; do
  if [ "$filepath" != "$last_filepath" ]; then
    printf '\n%s:\n' "$filepath"
    last_filepath=$filepath
  fi
  if [ -e "$(dirname "$filepath")/${link%%#*}" ]; then
    printf " ${green}OK${reset}  ${cyan}%s${reset}\n" "$link"
  else
    printf "${red}FAIL${reset} ${yellow}%s${reset}\n" "$link" >&2
    status=1
  fi
done < <(
  pattern='(?!.*@lint-ignore)(?:\[[^]]+\]\([^[:space:]\)]+/[^[:space:]\)]+\)|href="[^"]*/[^"]*"|src="[^"]*/[^"]*")'
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
  | grep -Ev 'https?://' \
  | sed -E \
      -e 's#([^:]+):\[[^]]+\]\(([^)]+)\)#\1:\2#' \
      -e 's#([^:]+):href="([^"]+)"#\1:\2#' \
      -e 's#([^:]+):src="([^"]+)"#\1:\2#' \
      -e 's/[[:punct:]]*$//' \
  | grep -Ev '\{\{' \
  || true
)

exit $status
