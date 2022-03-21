export MAX_JOBS=16
pip uninstall torch -y

# USE_DISTRIBUTED=1 USE_ROCM=1 USE_LMDB=1 USE_OPENCV=1 MAX_JOBS=4 python3.6 setup.py develop --user

cp scripts/amd/build_develop.sh .jenkins/pytorch/
bash .jenkins/pytorch/build_develop.sh
