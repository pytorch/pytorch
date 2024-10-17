#!/bin/bash

a100_zip="pad_mm_a100_data.zip"
a100_data="https://github.com/AlnisM/autoheuristic-datasets/raw/main/${a100_zip}"
rm -f ${a100_zip}
wget ${a100_data}
unzip -o ${a100_zip}
rm ${a100_zip}
