# faster build
export DEBUG=1
export USE_KINETO=0
export BUILD_CAFFE2=0
export USE_DISTRIBUTED=1
export BUILD_TEST=0
export USE_XNNPACK=0
export USE_FBGEMM=0
export USE_QNNPACK=0
export USE_MKLDNN=0
export USE_MIOPEN=0
export USE_NNPACK=0
export BUILD_CAFFE2_OPS=0
export USE_TENSORPIPE=1
export USE_CUDA=1

# cuda configs
export ATEN_STATIC_CUDA=0
export CUDA_HOME=/data/home/jessecai/cluster/cuda/11.6/toolkit
export CUDA_TOOLKIT_ROOT_DIR=/data/home/jessecai/cluster/cuda/11.6/toolkit
export CUDA_NVCC_EXECUTABLE=/data/home/jessecai/cluster/cuda/11.6/toolkit/bin/nvcc
