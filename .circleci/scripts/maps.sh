#!/bin/bash
set -eux -o pipefail

# The map's key and value must be delimited by `:`
# for example
# declare -a installers=(
#    "10.1: cudnn-10.1-windows10-x64-v7.6.4.38",
#    "11.1: cudnn-11.1-windows-x64-v8.0.5.39",
#    "11.2: cudnn-11.2-windows-x64-v8.1.0.77",
#    "11.3: cudnn-11.3-windows-x64-v8.2.0.53",
#)

map_get_value() {
    key=$1
    echo $key
    shift
    maps=("$@")

    map_return_value=""

    for elem in "${maps[@]}"; do
    IFS=":" read -a pair <<< "$elem"
    if [[ "$key" == "${pair[0]}" ]]; then
        map_return_value=${pair[1]}
        break
    fi
    done
}

