#include "caffe2/operators/order_switch_ops.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/fixed_divisor.h"

namespace caffe2 {

template <typename T>
__global__ void NHWC2NCHWCUDAKernel(
    const int size,
#ifndef __HIPCC__
    const FixedDivisor<int> C,
    const FixedDivisor<int> HxW,
#else
    const int C,
    const int HxW,
#endif
    const T* X,
    T* Y) {
  CUDA_1D_KERNEL_LOOP(i, size) {
    int n;
    int c;
    int hxw;

    int c_d;
    int hxw_d;
#ifndef __HIPCC__
    HxW.DivMod(i, &c, &hxw);
    C.DivMod(c, &n, &c);

    c_d = C.d();
    hxw_d = HxW.d();
#else
    c = i / HxW;
    hxw = i % HxW;
    n = c / C;
    c = c % C;

    c_d = C;
    hxw_d = HxW;
#endif

#if __CUDA_ARCH__ >= 350
    Y[i] = __ldg(X + (n * hxw_d + hxw) * c_d + c);
#else
    Y[i] = X[(n * hxw_d + hxw) * c_d + c];
#endif
  }
}

template <typename T>
__global__ void NCHW2NHWCCUDAKernel(
    const int size,
#ifndef __HIPCC__
    const FixedDivisor<int> C,
    const FixedDivisor<int> HxW,
#else
    const int C,
    const int HxW,
#endif
    const T* X,
    T* Y) {
  CUDA_1D_KERNEL_LOOP(i, size) {
    int n;
    int c;
    int hxw;

    int c_d;
    int hxw_d;
#ifndef __HIPCC__
    C.DivMod(i, &hxw, &c);
    HxW.DivMod(hxw, &n, &hxw);

    c_d = C.d();
    hxw_d = HxW.d();
#else
    hxw = i / C;
    c = i % C;
    n = hxw / HxW;
    hxw = hxw % HxW;

    c_d = C;
    hxw_d = HxW;
#endif
#if __CUDA_ARCH__ >= 350
    Y[i] = __ldg(X + (n * c_d + c) * hxw_d + hxw);
#else
    Y[i] = X[(n * c_d + c) * hxw_d + hxw];
#endif
  }
}

template <>
bool NHWC2NCHWOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  const int ndim = X.ndim();
  CAFFE_ENFORCE_GE(ndim, 3);
  const int N = X.dim32(0);
  const int C = X.dim32(ndim - 1);
  vector<TIndex> Y_dims(ndim);
  Y_dims[0] = N;
  Y_dims[1] = C;
  int HxW = 1;
  for (int i = 2; i < ndim; ++i) {
    Y_dims[i] = X.dim32(i - 1);
    HxW *= Y_dims[i];
  }
  Y->Resize(Y_dims);
  const int size = X.size();
  NHWC2NCHWCUDAKernel<float>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          size,
#ifndef __HIPCC__
          FixedDivisor<int>(C),
          FixedDivisor<int>(HxW),
#else
          C,
          HxW,
#endif
          X.data<float>(),
          Y->template mutable_data<float>());
  return true;
}

template <>
bool NCHW2NHWCOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  const int ndim = X.ndim();
  CAFFE_ENFORCE_GE(X.ndim(), 3);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  vector<TIndex> Y_dims(ndim);
  Y_dims[0] = N;
  int HxW = 1;
  for (auto i = 1; i < ndim - 1; ++i) {
    Y_dims[i] = X.dim32(i + 1);
    HxW *= Y_dims[i];
  }
  Y_dims[ndim - 1] = C;
  Y->Resize(Y_dims);
  const int size = X.size();
  NCHW2NHWCCUDAKernel<float>
      <<<CAFFE_GET_BLOCKS(size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          size,
#ifndef __HIPCC__
          FixedDivisor<int>(C),
          FixedDivisor<int>(HxW),
#else
          C,
          HxW,
#endif
          X.data<float>(),
          Y->template mutable_data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(NHWC2NCHW, NHWC2NCHWOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(NCHW2NHWC, NCHW2NHWCOp<float, CUDAContext>);

} // namespace caffe2
