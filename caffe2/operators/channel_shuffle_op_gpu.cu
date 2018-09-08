#include "caffe2/operators/channel_shuffle_op.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/fixed_divisor.h"

namespace caffe2 {

template <typename T>
__global__ void ChannelShuffleNCHWCUDAKernel(
    const int size,
    const FixedDivisor<int> C,
    const FixedDivisor<int> G,
    const FixedDivisor<int> HxW,
    const T* X,
    T* Y) {
  const int K = C.d() / G.d();
  CUDA_1D_KERNEL_LOOP(i, size) {
    int nc;
    int hw;
    HxW.DivMod(i, &nc, &hw);
    int n;
    int c;
    C.DivMod(nc, &n, &c);
    int g;
    int k;
    G.DivMod(c, &k, &g);
#if __CUDA_ARCH__ >= 350
    Y[i] = __ldg(X + (n * C.d() + g * K + k) * HxW.d() + hw);
#else
    Y[i] = X[(n * C.d() + g * K + k) * HxW.d() + hw];
#endif
  }
}

template <typename T, int kChannelSize>
__global__ void ChannelShuffleNHWCSharedCUDAKernel(
    const int outer_size,
    const int C,
    const FixedDivisor<int> G,
    const T* X,
    T* Y) {
  const int K = C / G.d();
  __shared__ T channel[kChannelSize + 1];
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
#if __CUDA_ARCH__ >= 350
      channel[j] = __ldg(X + i * C + j);
#else
      channel[j] = X[i * C + j];
#endif
    }
    __syncthreads();
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
      int g;
      int k;
      G.DivMod(j, &k, &g);
      Y[i * C + j] = channel[g * K + k];
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void ChannelShuffleNHWCCUDAKernel(
    const int size,
    const FixedDivisor<int> C,
    const FixedDivisor<int> G,
    const T* X,
    T* Y) {
  const int K = C.d() / G.d();
  CUDA_1D_KERNEL_LOOP(i, size) {
    int nhw;
    int c;
    C.DivMod(i, &nhw, &c);
    int g;
    int k;
    G.DivMod(c, &k, &g);
#if __CUDA_ARCH__ >= 350
    Y[i] = __ldg(X + nhw * C.d() + g * K + k);
#else
    Y[i] = X[nhw * C.d() + g * K + k];
#endif
  }
}

template <>
bool ChannelShuffleOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  const int size = X.size();
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int G = this->group_;
  CAFFE_ENFORCE_EQ(C % G, 0);
  const int HxW = size / (N * C);
  ChannelShuffleNCHWCUDAKernel<float>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          size,
          FixedDivisor<int>(C),
          FixedDivisor<int>(G),
          FixedDivisor<int>(HxW),
          X.data<float>(),
          Y->mutable_data<float>());
  return true;
}

template <>
bool ChannelShuffleOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  const int ndim = X.ndim();
  const int size = X.size();
  const int C = X.dim32(ndim - 1);
  const int G = this->group_;
  CAFFE_ENFORCE_EQ(C % G, 0);
  const int outer_size = size / C;
  const float* X_data = X.data<float>();
  float* Y_data = Y->mutable_data<float>();
  if (C <= 32) {
    ChannelShuffleNHWCSharedCUDAKernel<float, 32>
        <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            outer_size, C, FixedDivisor<int>(G), X_data, Y_data);
  } else if (C <= 128) {
    ChannelShuffleNHWCSharedCUDAKernel<float, 128>
        <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            outer_size, C, FixedDivisor<int>(G), X_data, Y_data);
  } else if (C <= 512) {
    ChannelShuffleNHWCSharedCUDAKernel<float, 512>
        <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            outer_size, C, FixedDivisor<int>(G), X_data, Y_data);
  } else {
    ChannelShuffleNHWCCUDAKernel<float>
        <<<CAFFE_GET_BLOCKS(size),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            size, FixedDivisor<int>(C), FixedDivisor<int>(G), X_data, Y_data);
  }
  return true;
}

template <>
bool ChannelShuffleGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  const auto& dY = Input(0);
  auto* dX = Output(0);
  dX->ResizeLike(dY);
  const int size = dY.size();
  const int N = dY.dim32(0);
  const int C = dY.dim32(1);
  const int G = this->group_;
  CAFFE_ENFORCE_EQ(C % G, 0);
  const int K = C / G;
  const int HxW = size / (N * C);
  ChannelShuffleNCHWCUDAKernel<float>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          size,
          FixedDivisor<int>(C),
          FixedDivisor<int>(K),
          FixedDivisor<int>(HxW),
          dY.data<float>(),
          dX->mutable_data<float>());
  return true;
}

template <>
bool ChannelShuffleGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  const auto& dY = Input(0);
  auto* dX = Output(0);
  dX->ResizeLike(dY);
  const int ndim = dY.ndim();
  const int size = dY.size();
  const int C = dY.dim32(ndim - 1);
  const int G = this->group_;
  CAFFE_ENFORCE_EQ(C % G, 0);
  const int outer_size = size / C;
  const int K = C / G;
  const float* dY_data = dY.data<float>();
  float* dX_data = dX->mutable_data<float>();
  if (C <= 32) {
    ChannelShuffleNHWCSharedCUDAKernel<float, 32>
        <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            outer_size, C, FixedDivisor<int>(K), dY_data, dX_data);
  } else if (C <= 128) {
    ChannelShuffleNHWCSharedCUDAKernel<float, 128>
        <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            outer_size, C, FixedDivisor<int>(K), dY_data, dX_data);
  } else if (C <= 512) {
    ChannelShuffleNHWCSharedCUDAKernel<float, 512>
        <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            outer_size, C, FixedDivisor<int>(K), dY_data, dX_data);
  } else {
    ChannelShuffleNHWCCUDAKernel<float>
        <<<CAFFE_GET_BLOCKS(size),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            size, FixedDivisor<int>(C), FixedDivisor<int>(K), dY_data, dX_data);
  }
  return true;
}

REGISTER_CUDA_OPERATOR(ChannelShuffle, ChannelShuffleOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ChannelShuffleGradient,
    ChannelShuffleGradientOp<float, CUDAContext>);

} // namespace caffe2
