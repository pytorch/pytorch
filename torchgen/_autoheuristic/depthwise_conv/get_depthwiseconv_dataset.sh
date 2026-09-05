#!/bin/bash

base_url='https://github.com/gderossi/aten-autoheuristic-datasets/raw/main/'  # @lint-ignore
dataset='depthwiseconv_data.zip'
rm -f ${dataset}
url="${base_url}${dataset}"
wget ${url}
unzip -o ${dataset}
rm ${dataset}
