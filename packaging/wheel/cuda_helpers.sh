export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
export NCCL_ROOT_DIR=/usr/local/cuda
export TH_BINARY_BUILD=1
export USE_STATIC_CUDNN=1
export USE_STATIC_NCCL=1
export ATEN_STATIC_CUDA=1
export USE_CUDA_STATIC_LINK=1
export INSTALL_TEST=0 # dont install test binaries into site-packages

export CUDA_VERSION=$(nvcc --version | tail -n1 | cut -f5 -d" " | cut -f1 -d",")

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0"
    case ${CUDA_VERSION} in
        11.*)
            TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;7.5;8.0"
            EXTRA_CAFFE2_CMAKE_FLAGS+=("-DATEN_NO_TEST=ON")
            ;;
        10.*)
            TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;7.5"
            EXTRA_CAFFE2_CMAKE_FLAGS+=("-DATEN_NO_TEST=ON")
            ;;
        9.*)
            TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
            ;;
        *)
            echo "ERROR: unknown cuda version $CUDA_VERSION"
            exit 1
            ;;
    esac
fi

LIBGOMP_PATH=$(find / -name libgomp.so.1 | head -1)
if ! find / -name libgomp.so.1 | head -1 >/dev/null 2>/dev/null; then
    echo "ERROR: libgomp.so.1 not found, exiting"
    exit 1
fi

export DEPS_LIST=(
    "/usr/local/cuda/lib64/libcudart.so.${CUDA_VERSION}"
    "/usr/local/cuda/lib64/libnvToolsExt.so.1"
    "/usr/local/cuda/lib64/libnvrtc.so.${CUDA_VERSION}"
    "/usr/local/cuda/lib64/libnvrtc-builtins.so"
    "$LIBGOMP_PATH"
)

export DEPS_SONAME=(
    "libcudart.so.${CUDA_VERSION}"
    "libnvToolsExt.so.1"
    "libnvrtc.so.${CUDA_VERSION}"
    "libnvrtc-builtins.so"
    "libgomp.so.1"
)
