#!/bin/bash

a100_data='https://github.com/AlnisM/autoheuristic-datasets/raw/main/mixedmm_a100_data.zip'
wget ${a100_data}
unzip mixedmm_a100_data.zip
rm mixedmm_a100_data.zip
