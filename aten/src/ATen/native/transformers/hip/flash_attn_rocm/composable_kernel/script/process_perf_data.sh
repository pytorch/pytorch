#!/bin/bash 
#
# in order to run this script you'd need the following python packages:

#pip3 install --upgrade pip
#pip3 install sqlalchemy pymysql pandas sshtunnel

# you would also need to set up some environment variables in order to 
# post your new test results to the database and compare them to the baseline
# please contact Illia.Silin@amd.com for more details

#process results
python3 process_perf_data.py perf_gemm.log
python3 process_perf_data.py perf_resnet50_N256.log
python3 process_perf_data.py perf_resnet50_N4.log
