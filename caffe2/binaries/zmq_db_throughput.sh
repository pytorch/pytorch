#!/bin/bash

INPUT_DB=gen/data/mnist/mnist-train-nchw-leveldb/ 
INPUT_DB_TYPE=leveldb

echo "Starting zmq feeder..."
gen/caffe2/binaries/zmq_feeder \
  --input_db $INPUT_DB \
  --input_db_type $INPUT_DB_TYPE &

FEEDER_PID=$!

gen/caffe2/binaries/db_throughput \
  --input_db "tcp://localhost:5555" \
  --input_db_type zmqdb \
  --report_interval 10000 \
  --repeat 10

kill $FEEDER_PID
