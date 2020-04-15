# How to run FakeLowP vs Glow tests
This was tested on Ubuntu 16.04 LTS but should work in general Linux system.

## Install and Build Glow
Follow https://github.com/pytorch/glow/blob/master/README.md to install the dependency of Glow. Then at glow root run
```
mkdir build && cd build
cmake -G Ninja -DGLOW_BUILD_ONNXIFI_DYNLIB=ON ..
ninja all
```
Note that here you probably want to add other flags like `-DGLOW_WITH_NNPI=1` to enable specific backend if you have the flow setup.
Once built successfully, you will get an dynamic library at `build/lib/Onnxifi/libonnxifi.so`. We will use it later.

## Install and Build PyTorch
Follow https://github.com/pytorch/pytorch/blob/master/README.md to install the dependency of PyTorch. It might be easy to 
setup a python virtualenv or conda. And please use Python > 3.5.2 because hypothesis library will expose a bug in Python which 
is fixed after 3.5.2. Something like 3.7 mighr be good enough. Here I give a virtualenv flow:
```
python3.7 -m venv venv3
source venv3/bin/active
cd pytorch
pip install -r requirements.txt
```
You probably need to install gflags-dev too with
```
sudo apt-get install libgflags-dev
```

Once you have all the dependency libs installed, build PyTorch with FakeLowP op support
```
USE_CUDA=0 USE_ROCM=0 USE_FAKELOWP=ON DEBUG=1 CMAKE_BUILD_TYPE=Debug USE_GFLAGS=1 USE_GLOG=1 USE_MKLDNN=0 python setup.py install
```
The key options here are `USE_FAKELOWP=ON` which enables building of FakeLowP operators and `USE_GFLAGS=1` which enables gflags as we 
use gflags in Glow to pass options. Other flags are mostl for fast build time and debug purpose. 

## Run the test
You can now run the tests with command like the following  when you are inside the virtual python env:
```
OSS_ONNXIFI_LIB=${PATH_TO_GLOW}/build/lib/Onnxifi/libonnxifi.so python pytorch/caffe2/contrib/fakelowp/test/test_sls_nnpi_fp16.py
```

