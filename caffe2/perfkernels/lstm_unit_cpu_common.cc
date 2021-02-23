#include "caffe2/perfkernels/lstm_unit_cpu_common.h"
#include "caffe2/perfkernels/common.h"
#include "caffe2/perfkernels/lstm_unit_cpu-impl.h"

namespace caffe2 {
namespace detail {

// Define templated implementation fo LSTM kernels on CPU
template <typename T>
void LstmUnitCpu(
    const int N,
    const int D,
    const int t,
    const T* H_prev,
    const T* C_prev,
    const T* X,
    const int32_t* seqLengths,
    const bool drop_states,
    T* C,
    T* H,
    const float forget_bias) {
  // Do CPU dispatching
  AVX2_FMA_DO(
      perfkernels::LstmUnitImpl,
      N,
      D,
      t,
      H_prev,
      C_prev,
      X,
      seqLengths,
      drop_states,
      C,
      H,
      forget_bias);
  perfkernels::LstmUnitImpl(
      N, D, t, H_prev, C_prev, X, seqLengths, drop_states, C, H, forget_bias);
}

template <typename T>
void LstmUnitGradientCpu(
    int N,
    int D,
    int t,
    const T* C_prev,
    const T* X,
    const int32_t* seqLengths,
    const T* C,
    const T* H,
    const T* C_diff,
    const T* H_diff,
    bool drop_states,
    T* H_prev_diff,
    T* C_prev_diff,
    T* X_diff,
    const float forget_bias) {
  // Do CPU dispatching
  AVX2_FMA_DO(
      perfkernels::LstmUnitGradientImpl,
      N,
      D,
      t,
      C_prev,
      X,
      seqLengths,
      C,
      H,
      C_diff,
      H_diff,
      drop_states,
      H_prev_diff,
      C_prev_diff,
      X_diff,
      forget_bias);
  perfkernels::LstmUnitGradientImpl(
      N,
      D,
      t,
      C_prev,
      X,
      seqLengths,
      C,
      H,
      C_diff,
      H_diff,
      drop_states,
      H_prev_diff,
      C_prev_diff,
      X_diff,
      forget_bias);
}

// Explicit initialize for float
template void LstmUnitCpu<float>(
    const int N,
    const int D,
    const int t,
    const float* H_prev,
    const float* C_prev,
    const float* X,
    const int32_t* seqLengths,
    const bool drop_states,
    float* C,
    float* H,
    const float forget_bias);

template void LstmUnitGradientCpu<float>(
    int N,
    int D,
    int t,
    const float* C_prev,
    const float* X,
    const int32_t* seqLengths,
    const float* C,
    const float* H,
    const float* C_diff,
    const float* H_diff,
    bool drop_states,
    float* H_prev_diff,
    float* C_prev_diff,
    float* X_diff,
    const float forget_bias);

} // namespace detail
} // namespace caffe2
