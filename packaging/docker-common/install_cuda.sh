#!/bin/bash

set -ex

function install_92 {
    echo "Installing CUDA 9.2 and CuDNN"
    # install CUDA 9.2 in the same container
    wget -q https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda_9.2.148_396.37_linux -O setup
    chmod +x setup
    ./setup --silent --no-opengl-libs --toolkit
    rm -f setup

    # patch 1
    wget -q https://developer.nvidia.com/compute/cuda/9.2/Prod2/patches/1/cuda_9.2.148.1_linux -O setup
    chmod +x setup
    ./setup -s --accept-eula
    rm -f setup

    # install CUDA 9.2 CuDNN
    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.6.3.30-1+cuda9.2_amd64.deb -O cudnn-dev.deb
    wget -q http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.6.3.30-1+cuda9.2_amd64.deb -O cudnn.deb
    ar -x cudnn-dev.deb && tar -xvf data.tar.xz
    ar -x cudnn.deb && tar -xvf data.tar.xz
    mkdir -p cuda/include && mkdir -p cuda/lib64
    cp -a usr/include/x86_64-linux-gnu/cudnn_v7.h cuda/include/cudnn.h
    cp -a usr/lib/x86_64-linux-gnu/libcudnn* cuda/lib64
    mv cuda/lib64/libcudnn_static_v7.a cuda/lib64/libcudnn_static.a
    ln -s libcudnn.so.7 cuda/lib64/libcudnn.so
    chmod +x cuda/lib64/*.so
    cp -a cuda/include/* /usr/local/cuda/include/
    cp -a cuda/lib64/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn
    ldconfig
}

function install_101 {
    echo "Installing CUDA 10.1 and CuDNN"
    rm -rf /usr/local/cuda-10.1 /usr/local/cuda
    # # install CUDA 10.1 in the same container
    wget -q http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
    chmod +x cuda_10.1.243_418.87.00_linux.run
    ./cuda_10.1.243_418.87.00_linux.run    --extract=/tmp/cuda
    rm -f cuda_10.1.243_418.87.00_linux.run
    mv /tmp/cuda/cuda-toolkit /usr/local/cuda-10.1
    rm -rf /tmp/cuda
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-10.1 /usr/local/cuda

    # install CUDA 10.1 CuDNN
    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.6.3.30-1+cuda10.1_amd64.deb -O cudnn-dev.deb
    wget -q http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.6.3.30-1+cuda10.1_amd64.deb -O cudnn.deb
    ar -x cudnn-dev.deb && tar -xvf data.tar.xz
    ar -x cudnn.deb && tar -xvf data.tar.xz
    mkdir -p cuda/include && mkdir -p cuda/lib64
    cp -a usr/include/x86_64-linux-gnu/cudnn_v7.h cuda/include/cudnn.h
    cp -a usr/lib/x86_64-linux-gnu/libcudnn* cuda/lib64
    mv cuda/lib64/libcudnn_static_v7.a cuda/lib64/libcudnn_static.a
    ln -s libcudnn.so.7 cuda/lib64/libcudnn.so
    chmod +x cuda/lib64/*.so
    cp -a cuda/include/* /usr/local/cuda/include/
    cp -a cuda/lib64/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn
    ldconfig

}

function install_102 {
    echo "Installing CUDA 10.2 and CuDNN"
    rm -rf /usr/local/cuda-10.2 /usr/local/cuda
    # # install CUDA 10.2 in the same container
    wget -q http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
    chmod +x cuda_10.2.89_440.33.01_linux.run
    ./cuda_10.2.89_440.33.01_linux.run    --extract=/tmp/cuda
    rm -f cuda_10.2.89_440.33.01_linux.run
    mv /tmp/cuda/cuda-toolkit /usr/local/cuda-10.2
    rm -rf /tmp/cuda
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-10.2 /usr/local/cuda

    # install CUDA 10.2 CuDNN
    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.6.5.32-1+cuda10.2_amd64.deb -O cudnn-dev.deb
    wget -q http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb -O cudnn.deb
    ar -x cudnn-dev.deb && tar -xvf data.tar.xz
    ar -x cudnn.deb && tar -xvf data.tar.xz
    mkdir -p cuda/include && mkdir -p cuda/lib64
    cp -a usr/include/x86_64-linux-gnu/cudnn_v7.h cuda/include/cudnn.h
    cp -a usr/lib/x86_64-linux-gnu/libcudnn* cuda/lib64
    mv cuda/lib64/libcudnn_static_v7.a cuda/lib64/libcudnn_static.a
    ln -s libcudnn.so.7 cuda/lib64/libcudnn.so
    chmod +x cuda/lib64/*.so
    cp -a cuda/include/* /usr/local/cuda/include/
    cp -a cuda/lib64/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn
    ldconfig

}

function install_110 {
    echo "Installing CUDA 11.0 and CuDNN"
    rm -rf /usr/local/cuda-11.0 /usr/local/cuda
    # # install CUDA 11.0 in the same container
    wget -q http://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda_11.0.3_450.51.06_linux.run
    chmod +x cuda_11.0.3_450.51.06_linux.run
    ./cuda_11.0.3_450.51.06_linux.run --toolkit --silent
    rm -f cuda_11.0.3_450.51.06_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-11.0 /usr/local/cuda

    # install CUDA 11.0 CuDNN
    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.5/cudnn-11.0-linux-x64-v8.0.5.39.tgz -O cudnn-8.0.tgz
    tar xf cudnn-8.0.tgz
    cp -a cuda/include/* /usr/local/cuda/include/
    cp -a cuda/lib64/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn
    ldconfig
}

function install_111 {
    echo "Installing CUDA 11.1 and CuDNN"
    rm -rf /usr/local/cuda-11.1 /usr/local/cuda
    # install CUDA 11.1 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
    chmod +x cuda_11.1.1_455.32.00_linux.run
    ./cuda_11.1.1_455.32.00_linux.run --toolkit --silent
    rm -f cuda_11.1.1_455.32.00_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-11.1 /usr/local/cuda

    # install CUDA 11.1 CuDNN 8.0.5
    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.5/cudnn-11.1-linux-x64-v8.0.5.39.tgz -O cudnn-8.0.tgz
    tar xf cudnn-8.0.tgz
    cp -a cuda/include/* /usr/local/cuda/include/
    cp -a cuda/lib64/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn
    ldconfig
}

function install_112 {
    echo "Installing CUDA 11.2 and CuDNN"
    rm -rf /usr/local/cuda-11.2 /usr/local/cuda
    # install CUDA 11.2 in the same container
    wget -q https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
    chmod +x cuda_11.2.0_460.27.04_linux.run
    ./cuda_11.2.0_460.27.04_linux.run --toolkit --silent
    rm -f cuda_11.2.0_460.27.04_linux.run
    rm -f /usr/local/cuda && ln -s /usr/local/cuda-11.2 /usr/local/cuda

    # TODO: install CUDA 11.2 CuDNN 8.0.5, currently it's the 11.1 version
    # cuDNN license: https://developer.nvidia.com/cudnn/license_agreement
    mkdir tmp_cudnn && cd tmp_cudnn
    wget -q https://developer.download.nvidia.com/compute/redist/cudnn/v8.0.5/cudnn-11.1-linux-x64-v8.0.5.39.tgz -O cudnn-8.0.tgz
    tar xf cudnn-8.0.tgz
    cp -a cuda/include/* /usr/local/cuda/include/
    cp -a cuda/lib64/* /usr/local/cuda/lib64/
    cd ..
    rm -rf tmp_cudnn
    ldconfig
}

function prune_92 {
    echo "Pruning CUDA 9.2 and CuDNN"
    #####################################################################################
    # CUDA 9.2 prune static libs
    #####################################################################################
    export NVPRUNE="/usr/local/cuda-9.2/bin/nvprune"
    export CUDA_LIB_DIR="/usr/local/cuda-9.2/lib64"

    export GENCODE="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70"
    export GENCODE_CUDNN="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70"

    if [[ -n "$OVERRIDE_GENCODE" ]]; then
        export GENCODE=$OVERRIDE_GENCODE
    fi

    # all CUDA libs except CuDNN and CuBLAS (cudnn and cublas need arch 3.7 included)
    ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "cusolver" \
	| xargs -I {} bash -c \
		"echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

    # prune CuDNN and CuBLAS
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcudnn_static.a -o $CUDA_LIB_DIR/libcudnn_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_device.a -o $CUDA_LIB_DIR/libcublas_device.a

    #####################################################################################
    # CUDA 9.2 prune visual tools
    #####################################################################################
    export CUDA_BASE="/usr/local/cuda-9.2/"
    rm -rf $CUDA_BASE/jre  $CUDA_BASE/libnsight $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins

}

function prune_101 {
    echo "Pruning CUDA 10.1 and CuDNN"
    #####################################################################################
    # CUDA 10.1 prune static libs
    #####################################################################################
    export NVPRUNE="/usr/local/cuda-10.1/bin/nvprune"
    export CUDA_LIB_DIR="/usr/local/cuda-10.1/lib64"

    export GENCODE="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75"
    export GENCODE_CUDNN="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75"

    if [[ -n "$OVERRIDE_GENCODE" ]]; then
        export GENCODE=$OVERRIDE_GENCODE
    fi

    # all CUDA libs except CuDNN and CuBLAS (cudnn and cublas need arch 3.7 included)
    ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
	| xargs -I {} bash -c \
		"echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

    # prune CuDNN and CuBLAS
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcudnn_static.a -o $CUDA_LIB_DIR/libcudnn_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

    #####################################################################################
    # CUDA 10.1 prune visual tools
    #####################################################################################
    export CUDA_BASE="/usr/local/cuda-10.1/"
    rm -rf $CUDA_BASE/libnsight $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2019.4.0 $CUDA_BASE/nsight-systems-2019.3.7.5
}

function prune_102 {
    echo "Pruning CUDA 10.2 and CuDNN"
    #####################################################################################
    # CUDA 10.2 prune static libs
    #####################################################################################
    export NVPRUNE="/usr/local/cuda-10.2/bin/nvprune"
    export CUDA_LIB_DIR="/usr/local/cuda-10.2/lib64"

    export GENCODE="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75"
    export GENCODE_CUDNN="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75"

    if [[ -n "$OVERRIDE_GENCODE" ]]; then
        export GENCODE=$OVERRIDE_GENCODE
    fi

    # all CUDA libs except CuDNN and CuBLAS (cudnn and cublas need arch 3.7 included)
    ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
	| xargs -I {} bash -c \
		"echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

    # prune CuDNN and CuBLAS
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcudnn_static.a -o $CUDA_LIB_DIR/libcudnn_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

    #####################################################################################
    # CUDA 10.2 prune visual tools
    #####################################################################################
    export CUDA_BASE="/usr/local/cuda-10.2/"
    rm -rf $CUDA_BASE/libnsight $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2019.5.0 $CUDA_BASE/nsight-systems-2019.5.2

}

function prune_110 {
    echo "Pruning CUDA 11.0 and CuDNN"
    #####################################################################################
    # CUDA 11.0 prune static libs
    #####################################################################################
    export NVPRUNE="/usr/local/cuda-11.0/bin/nvprune"
    export CUDA_LIB_DIR="/usr/local/cuda-11.0/lib64"

    export GENCODE="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80"
    export GENCODE_CUDNN="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80"

    if [[ -n "$OVERRIDE_GENCODE" ]]; then
        export GENCODE=$OVERRIDE_GENCODE
    fi

    # all CUDA libs except CuDNN and CuBLAS (cudnn and cublas need arch 3.7 included)
    ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
      | xargs -I {} bash -c \
		"echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

    # prune CuDNN and CuBLAS
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcudnn_static.a -o $CUDA_LIB_DIR/libcudnn_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

    #####################################################################################
    # CUDA 11.0 prune visual tools
    #####################################################################################
    export CUDA_BASE="/usr/local/cuda-11.0/"
    rm -rf $CUDA_BASE/libnsight $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2020.1.0 $CUDA_BASE/nsight-systems-2020.2.5
}

function prune_111 {
    echo "Pruning CUDA 11.1 and CuDNN"
    #####################################################################################
    # CUDA 11.1 prune static libs
    #####################################################################################
    export NVPRUNE="/usr/local/cuda-11.1/bin/nvprune"
    export CUDA_LIB_DIR="/usr/local/cuda-11.1/lib64"

    export GENCODE="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86"
    export GENCODE_CUDNN="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86"

    if [[ -n "$OVERRIDE_GENCODE" ]]; then
        export GENCODE=$OVERRIDE_GENCODE
    fi

    # all CUDA libs except CuDNN and CuBLAS (cudnn and cublas need arch 3.7 included)
    ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
      | xargs -I {} bash -c \
		"echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

    # prune CuBLAS (not CuDNN since that segfaults and is a known bug with CUDA11.1)
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

    #####################################################################################
    # CUDA 11.1 prune visual tools
    #####################################################################################
    export CUDA_BASE="/usr/local/cuda-11.1/"
    rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2020.2.1 $CUDA_BASE/nsight-systems-2020.3.4
}

function prune_112 {
    echo "Pruning CUDA 11.2 and CuDNN"
    #####################################################################################
    # CUDA 11.2 prune static libs
    #####################################################################################
    export NVPRUNE="/usr/local/cuda-11.2/bin/nvprune"
    export CUDA_LIB_DIR="/usr/local/cuda-11.2/lib64"

    export GENCODE="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86"
    export GENCODE_CUDNN="-gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86"

    if [[ -n "$OVERRIDE_GENCODE" ]]; then
        export GENCODE=$OVERRIDE_GENCODE
    fi

    # all CUDA libs except CuDNN and CuBLAS (cudnn and cublas need arch 3.7 included)
    ls $CUDA_LIB_DIR/ | grep "\.a" | grep -v "culibos" | grep -v "cudart" | grep -v "cudnn" | grep -v "cublas" | grep -v "metis"  \
      | xargs -I {} bash -c \
		"echo {} && $NVPRUNE $GENCODE $CUDA_LIB_DIR/{} -o $CUDA_LIB_DIR/{}"

    # prune CuDNN and CuBLAS
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcudnn_static.a -o $CUDA_LIB_DIR/libcudnn_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublas_static.a -o $CUDA_LIB_DIR/libcublas_static.a
    $NVPRUNE $GENCODE_CUDNN $CUDA_LIB_DIR/libcublasLt_static.a -o $CUDA_LIB_DIR/libcublasLt_static.a

    #####################################################################################
    # CUDA 11.2 prune visual tools
    #####################################################################################
    export CUDA_BASE="/usr/local/cuda-11.2/"
    rm -rf $CUDA_BASE/libnvvp $CUDA_BASE/nsightee_plugins $CUDA_BASE/nsight-compute-2020.3.0 $CUDA_BASE/nsight-systems-2020.4.3
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
	9.2) install_92; prune_92
		;;
	10.1) install_101; prune_101
		;;
	10.2) install_102; prune_102
		;;
	11.0) install_110; prune_110
		;;
    11.1) install_111; prune_111
		;;
    11.2) install_112; prune_112
		;;
	*) echo "bad argument $1"; exit 1
	   ;;
    esac
    shift
done
