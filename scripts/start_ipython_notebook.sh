#!/usr/bin/env sh
# This script simply starts the ipython notebook and allows all network machines
# to access it.

# Use the following command for very verbose prints.
# GLOG_logtostderr=1 GLOG_v=1 PYTHONPATH=../../../build:$PYTHONPATH jupyter notebook "$@"

# Use the following command for a normal run.
PYTHONPATH=build:$PYTHONPATH jupyter notebook --notebook-dir=caffe2/python/tutorials "$@"
