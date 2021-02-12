pip uninstall torch -y
USE_DISTRIBUTED=1 USE_ROCM=1 USE_LMDB=1 USE_OPENCV=1 MAX_JOBS=4 python3.6 setup.py develop --user