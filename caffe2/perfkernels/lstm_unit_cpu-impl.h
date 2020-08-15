#pragma once
#include <cmath>
#include "caffe2/utils/conversions.h"

#if (ENABLE_VECTORIZATION > 0) && !defined(_DEBUG) && !defined(DEBUG)
#if defined(__clang__) && (__clang_major__ > 7)
#define IS_SANITIZER                          \
  ((__has_feature(address_sanitizer) == 1) || \
   (__has_feature(memory_sanitizer) == 1) ||  \
   (__has_feature(thread_sanitizer) == 1) ||  \
   (__has_feature(undefined_sanitizer) == 1))

#if IS_SANITIZER == 0
#define VECTOR_LOOP _Pragma("clang loop vectorize(enable)")
#endif
#elif defined(_OPENMP) && (_OPENMP >= 201511)
// Support with OpenMP4.5 and above
#define VECTOR_LOOP _Pragma("omp for simd")
#endif
#endif

#ifndef VECTOR_LOOP
// Not supported
#define VECTOR_LOOP
#endif

namespace caffe2 {
namespace perfkernels {
namespace {
template <typename T>
inline T sigmoid(T x) {
  return 1 / (1 + std::exp(-x));
}

template <typename T>
inline T host_tanh(T x) {
  return 2 * sigmoid(2 * x) - 1;
}

template <typename T>
inline void LstmUnitImpl(
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
  const T forgetBias = convert::To<float, T>(forget_bias);
  for (int n = 0; n < N; ++n) {
    const bool valid = seqLengths == nullptr || t < seqLengths[n];
    if (!valid) {
      if (drop_states) {
        memset(H, 0, sizeof(T) * D);
        memset(C, 0, sizeof(T) * D);
      } else {
        memcpy(H, H_prev, sizeof(T) * D);
        memcpy(C, C_prev, sizeof(T) * D);
      }
    } else {
      const T* X_D = &X[D];
      const T* X_2D = &X[2 * D];
      const T* X_3D = &X[3 * D];
      VECTOR_LOOP for (int d = 0; d < D; ++d) {
        const T i = sigmoid(X[d]);
        const T f = sigmoid(X_D[d] + forgetBias);
        const T o = sigmoid(X_2D[d]);
        const T g = host_tanh(X_3D[d]);
        const T c_prev = C_prev[d];
        const T c = f * c_prev + i * g;
        C[d] = c;
        const T host_tanh_c = host_tanh(c);
        H[d] = o * host_tanh_c;
      }
    }
    H_prev += D;
    C_prev += D;
    X += 4 * D;
    C += D;
    H += D;
  }
}

template <typename T>
inline void LstmUnitGradientImpl(
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
  const T localForgetBias = convert::To<float, T>(forget_bias);
  for (int n = 0; n < N; ++n) {
    const bool valid = seqLengths == nullptr || t < seqLengths[n];

    if (!valid) {
      if (drop_states) {
        memset(C_prev_diff, 0, sizeof(T) * D);
        memset(H_prev_diff, 0, sizeof(T) * D);
      } else {
        memcpy(H_prev_diff, H_diff, sizeof(T) * D);
        memcpy(C_prev_diff, C_diff, sizeof(T) * D);
      }
      memset(X_diff, 0, 4 * sizeof(T) * D);
    } else {
      VECTOR_LOOP for (int d = 0; d < D; ++d) {
        T* c_prev_diff = C_prev_diff + d;
        T* h_prev_diff = H_prev_diff + d;
        T* i_diff = X_diff + d;
        T* f_diff = X_diff + 1 * D + d;
        T* o_diff = X_diff + 2 * D + d;
        T* g_diff = X_diff + 3 * D + d;

        const T i = sigmoid(X[d]);
        const T f = sigmoid(X[1 * D + d] + localForgetBias);
        const T o = sigmoid(X[2 * D + d]);
        const T g = host_tanh(X[3 * D + d]);
        const T c_prev = C_prev[d];
        const T c = C[d];
        const T host_tanh_c = host_tanh(c);
        const T c_term_diff =
            C_diff[d] + H_diff[d] * o * (1 - host_tanh_c * host_tanh_c);
        *c_prev_diff = c_term_diff * f;
        *h_prev_diff = 0; // not used in 'valid' case
        *i_diff = c_term_diff * g * i * (1 - i);
        *f_diff = c_term_diff * c_prev * f * (1 - f);
        *o_diff = H_diff[d] * host_tanh_c * o * (1 - o);
        *g_diff = c_term_diff * i * (1 - g * g);
      }
    }
    C_prev += D;
    X += 4 * D;
    C += D;
    H += D;
    C_diff += D;
    H_diff += D;
    X_diff += 4 * D;
    H_prev_diff += D;
    C_prev_diff += D;
  }
}

} // namespace
} // namespace perfkernels
} // namespace caffe2
