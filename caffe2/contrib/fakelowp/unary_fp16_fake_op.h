#pragma once

#include <vector>

#include <fbgemm/FbgemmConvert.h>
#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

C10_DECLARE_bool(caffe2_fbgemm_fake_fp16_clamp);

namespace caffe2 {
using namespace std;
template <class Context>
struct ReluFakeFp16Functor {
  template <typename T>
  bool operator()(const int N, const T* X, T* Y, Context* /* unused */) const {
    std::vector<float> X_fp16(N);
    fbgemm::RoundToFloat16(
        X, X_fp16.data(), N, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    EigenVectorMap<T>(Y, N) =
        ConstEigenVectorMap<float>(X_fp16.data(), N).cwiseMax(T(0));
    return true;
  }
};

template <class Context>
struct SqrFakeFp16Functor {
  template <typename T>
  bool operator()(const int N, const T* X, T* Y, Context* context) const {
    std::vector<float> X_fp16(N);
    fbgemm::RoundToFloat16(
        X, X_fp16.data(), N, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    math::Sqr(N, X_fp16.data(), Y, context);
    fbgemm::RoundToFloat16(Y, Y, N, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    return true;
  }
};

struct SigmoidFakeIdealFp16Functor {
  template <typename T>
  bool operator()(const int N, const T* X, T* Y, CPUContext* /* unused */)
      const {
    std::vector<float> X_fp16(N);
    fbgemm::RoundToFloat16(X, X_fp16.data(), N);
    EigenVectorArrayMap<T>(Y, N) =
        T(1) / (T(1) + (-ConstEigenVectorArrayMap<T>(X_fp16.data(), N)).exp());
    fbgemm::RoundToFloat16(Y, Y, N, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    return true;
  }
};

struct TanhFakeIdealFp16Functor {
  template <typename T>
  bool operator()(const int N, const T* X, T* Y, CPUContext* context) const {
    std::vector<float> X_fp16(N);
    fbgemm::RoundToFloat16(
        X, X_fp16.data(), N, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    math::Tanh<T, CPUContext>(N, X_fp16.data(), Y, context);
    fbgemm::RoundToFloat16(Y, Y, N, FLAGS_caffe2_fbgemm_fake_fp16_clamp);
    return true;
  }
};

} // namespace caffe2

namespace fake_fp16 {

at::Half CalcSigmoidByLUT(at::Half x);
at::Half CalcSwishByLUT(at::Half x);
at::Half CalcSwishByLUTCubic(at::Half x);
at::Half CalcTanhByLUT(at::Half input);

} // namespace fake_fp16
