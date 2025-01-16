#!/bin/bash

base_url='https://github.com/AlnisM/autoheuristic-datasets/raw/main/'
a100_data='mixedmm_a100_data.zip'
h100_data='mixedmm_h100_data.zip'
datasets=("${a100_data}" "${h100_data}")
for dataset in "${datasets[@]}"; do
    rm -f ${dataset}
    url="${base_url}${dataset}"
    wget ${url}
    unzip -o ${dataset}
    rm ${dataset}
done
