Start with a clean conda environment.
1.  conda create --prefix /bert_ort/wechi/conda/dexport python==3.9
2.  Install PyTorch's dependencies. PyTorch itself will be rebuilt from source later.
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
3.  git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install -e .
4.  pip install cmake
5.  Clone PyTorch branch: pytorch/pytorch at onnx-team/dynamo-exporter (github.com)
6.  Build PyTorch
    example command:
    VERBOSE=1 BUILD_LAZY_TS_BACKEND=1 TORCH_CUDA_ARCH_LIST="7.0;7.2" PATH=cuda-11.8/lib64:cuda-11.8/include:cuda-11.8/bin:$PATH CUDACXX=cuda-11.8/bin/nvcc BUILD_SHARED_LIBS=1 BUILD_CAFFE2=0 BUILD_CAFFE2_OPS=0 USE_GLOO=1 USE_NCCL=1 USE_NUMPY=1 USE_OBSERVERS=1 USE_OPENMP=1 USE_DISTRIBUTED=1 USE_MPI=1 BUILD_PYTHON=1 USE_MKLDNN=0 USE_CUDA=1 BUILD_TEST=1 USE_FBGEMM=1 USE_NNPACK=1 USE_QNNPACK=0 USE_XNNPACK=1 python setup.py install 2>&1 | tee build.log
7.  pip install networkx
8.  pip install onnxruntime
9.  pip install expecttest
10. Test if exporter works
    python /bert_ort/wechi/dexport/pytorch/test/onnx/test_fx_to_onnx_with_onnxruntime.py

Test tiny GPT
 1. python pytorch/torch/onnx/_internal/docs/gpt.py gpt.py