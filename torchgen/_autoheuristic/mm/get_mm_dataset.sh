#!/bin/bash

base_url='https://github.com/AlnisM/autoheuristic-datasets/raw/main/'
a100_data='a100_mm.zip'
h100_data='h100_mm.zip'
datasets=("${a100_data}" "${h100_data}")
for dataset in "${datasets[@]}"; do
    url="${base_url}${dataset}"
    wget ${url}
    unzip ${dataset}
    rm ${dataset}
done
