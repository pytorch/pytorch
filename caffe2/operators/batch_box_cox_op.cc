#include "caffe2/operators/batch_box_cox_op.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

#ifdef CAFFE2_USE_MKL
#include <mkl.h>
#endif // CAFFE2_USE_MKL

namespace caffe2 {

#ifdef CAFFE2_USE_MKL
namespace {

// Helpers for copying parameters.
template <typename T>
void TileArrayIntoVector(const T* a, int D, int K, vector<T>* b) {
  b->resize(K * D);
  for (int k = 0; k < K; k++) {
    std::copy(a, a + D, b->begin() + k * D);
  }
}

void TileIndicesInPlace(vector<int>* v, int D, int K) {
  int n = v->size();
  v->resize(K * n);
  for (int k = 1; k < K; k++) {
    for (int j = 0; j < n; j++) {
      (*v)[k * n + j] = (*v)[j] + k * D;
    }
  }
}

// MKL VML function templates.
template <typename T>
void PackV(const int N, const T* a, const int* ia, T* y);
template <typename T>
void UnpackV(const int N, const T* a, T* y, const int* iy);
template <typename T>
void Pow(const int N, const T* a, const T* b, T* y);

#define DELEGATE_PACKV_FUNCTION(T, OriginalFunc)                \
  template <>                                                   \
  void PackV<T>(const int N, const T* a, const int* ia, T* y) { \
    OriginalFunc(N, a, ia, y);                                  \
  }
DELEGATE_PACKV_FUNCTION(float, vsPackV)
DELEGATE_PACKV_FUNCTION(double, vdPackV)
#undef DELEGATE_PACKV_FUNCTION

#define DELEGATE_UNPACKV_FUNCTION(T, OriginalFunc)                \
  template <>                                                     \
  void UnpackV<T>(const int N, const T* a, T* y, const int* iy) { \
    OriginalFunc(N, a, y, iy);                                    \
  }
DELEGATE_UNPACKV_FUNCTION(float, vsUnpackV)
DELEGATE_UNPACKV_FUNCTION(double, vdUnpackV)
#undef DELEGATE_UNPACKV_FUNCTION

#define DELEGATE_SIMPLE_BINARY_FUNCTION(T, Funcname, OriginalFunc) \
  template <>                                                      \
  void Funcname<T>(const int N, const T* a, const T* b, T* y) {    \
    OriginalFunc(N, a, b, y);                                      \
  }
DELEGATE_SIMPLE_BINARY_FUNCTION(float, Pow, vsPow)
DELEGATE_SIMPLE_BINARY_FUNCTION(double, Pow, vdPow)
#undef DELEGATE_SIMPLE_BINARY_FUNCTION

} // namespace
#endif // CAFFE2_USE_MKL

template <>
template <typename T>
bool BatchBoxCoxOp<CPUContext>::DoRunWithType() {
  auto& data = Input(DATA);
  auto& lambda1 = Input(LAMBDA1);
  auto& lambda2 = Input(LAMBDA2);
  CAFFE_ENFORCE_GE(data.dim(), 1);
  auto N = data.size(0);
  auto D = data.size_from_dim(1);

  auto* output = Output(0, Input(DATA).sizes(), at::dtype<T>());
  auto* output_ptr = output->template mutable_data<T>();

  if (data.numel() <= 0) {
    return true;
  }

  CAFFE_ENFORCE_EQ(lambda1.numel(), D);
  CAFFE_ENFORCE_EQ(lambda2.numel(), D);

  const auto* data_ptr = data.template data<T>();
  const auto* lambda1_ptr = lambda1.template data<T>();
  const auto* lambda2_ptr = lambda2.template data<T>();

  const T k_eps = static_cast<T>(1e-6);

#ifdef CAFFE2_USE_MKL
  if (min_block_size_ < 1) {
    BoxCoxNaive(N, D, data_ptr, lambda1_ptr, lambda2_ptr, k_eps, output_ptr);
  } else {
    // Find zero-valued columns, since they get special treatment.
    nonzeros_.clear();
    zeros_.clear();
    nonzeros_.reserve(D);
    zeros_.reserve(D);
    for (int64_t j = 0; j < D; j++) {
      if (lambda1_ptr[j] == 0) {
        zeros_.push_back(j);
      } else {
        nonzeros_.push_back(j);
      }
    }

    // Process K rows at a time for effective vectorization with small rows.
    const int K = std::min(N, (min_block_size_ + D - 1) / D);

    // Avoid copying data if all lambda1 values are zero, or if all are nonzero.
    // In each of the three cases here, when K > 1, first process batches of K
    // rows by replicating the input parameters K times. Then finish row-by-row.
    TypedCachedBuffers<T>& b = GetBuffers<T>();
    if (nonzeros_.size() == D) {
      int64_t i = 0;
      if (K > 1) {
        TileArrayIntoVector(lambda1_ptr, D, K, &b.lambda1_);
        TileArrayIntoVector(lambda2_ptr, D, K, &b.lambda2_);
        DCHECK_EQ(K * D, b.lambda1_.size());
        DCHECK_EQ(K * D, b.lambda2_.size());
        for (; i < N - K + 1; i += K, data_ptr += K * D, output_ptr += K * D) {
          BoxCoxNonzeroLambda(
              K * D,
              data_ptr,
              b.lambda1_.data(),
              b.lambda2_.data(),
              k_eps,
              output_ptr);
        }
      }
      for (; i < N; i++, data_ptr += D, output_ptr += D) {
        BoxCoxNonzeroLambda(
            D, data_ptr, lambda1_ptr, lambda2_ptr, k_eps, output_ptr);
      }
    } else if (zeros_.size() == D) {
      int64_t i = 0;
      if (K > 1) {
        TileArrayIntoVector(lambda2_ptr, D, K, &b.lambda2_z_);
        DCHECK_EQ(K * D, b.lambda2_z_.size());
        for (; i < N - K + 1; i += K, data_ptr += K * D, output_ptr += K * D) {
          BoxCoxZeroLambda(
              K * D, data_ptr, b.lambda2_z_.data(), k_eps, output_ptr);
        }
      }
      for (; i < N; i++, data_ptr += D, output_ptr += D) {
        BoxCoxZeroLambda(D, data_ptr, lambda2_ptr, k_eps, output_ptr);
      }
    } else { // General case of mixed zero and non-zero lambda1 values.
      int n = nonzeros_.size();
      if (K > 1) {
        TileIndicesInPlace(&nonzeros_, 0, K);
        TileIndicesInPlace(&zeros_, 0, K);
      }

      // Gather parameter values into contiguous memory.
      b.lambda1_.resize(nonzeros_.size());
      b.lambda2_.resize(nonzeros_.size());
      b.lambda2_z_.resize(zeros_.size());
      PackV(nonzeros_.size(), lambda1_ptr, nonzeros_.data(), b.lambda1_.data());
      PackV(nonzeros_.size(), lambda2_ptr, nonzeros_.data(), b.lambda2_.data());
      PackV(zeros_.size(), lambda2_ptr, zeros_.data(), b.lambda2_z_.data());

      int64_t i = 0;
      b.accumulator_.resize(std::max(nonzeros_.size(), zeros_.size()));
      if (K > 1) {
        // Truncate to original size, and re-tile with offsets this time.
        nonzeros_.resize(n);
        zeros_.resize(D - n);
        TileIndicesInPlace(&nonzeros_, D, K);
        TileIndicesInPlace(&zeros_, D, K);
        DCHECK_EQ(nonzeros_.size(), b.lambda1_.size());
        DCHECK_EQ(nonzeros_.size(), b.lambda2_.size());
        DCHECK_EQ(zeros_.size(), b.lambda2_z_.size());
        for (; i < N - K + 1; i += K, data_ptr += K * D, output_ptr += K * D) {
          BoxCoxMixedLambda(
              data_ptr,
              nonzeros_,
              zeros_,
              b.lambda1_.data(),
              b.lambda2_.data(),
              b.lambda2_z_.data(),
              k_eps,
              b.accumulator_.data(),
              output_ptr);
        }
        // Truncate to original size.
        nonzeros_.resize(n);
        zeros_.resize(D - n);
      }
      for (; i < N; i++, data_ptr += D, output_ptr += D) {
        BoxCoxMixedLambda(
            data_ptr,
            nonzeros_,
            zeros_,
            b.lambda1_.data(),
            b.lambda2_.data(),
            b.lambda2_z_.data(),
            k_eps,
            b.accumulator_.data(),
            output_ptr);
      }
    }
  }
#else // CAFFE2_USE_MKL
  BoxCoxNaive(N, D, data_ptr, lambda1_ptr, lambda2_ptr, k_eps, output_ptr);
#endif // CAFFE2_USE_MKL
  return true;
}

template <>
template <typename T>
void BatchBoxCoxOp<CPUContext>::BoxCoxNaive(
    int64_t N,
    int64_t D,
    const T* data_ptr,
    const T* lambda1_ptr,
    const T* lambda2_ptr,
    T k_eps,
    T* output_ptr) {
  for (int64_t i = 0; i < N; i++) {
    for (int64_t j = 0; j < D; j++, data_ptr++, output_ptr++) {
      T lambda1_v = lambda1_ptr[j];
      T lambda2_v = lambda2_ptr[j];
      T tmp = std::max(*data_ptr + lambda2_v, k_eps);
      if (lambda1_v == 0) {
        *output_ptr = std::log(tmp);
      } else {
        *output_ptr = (std::pow(tmp, lambda1_v) - 1) / lambda1_v;
      }
    }
  }
}

#ifdef CAFFE2_USE_MKL

template <>
template <typename T>
void BatchBoxCoxOp<CPUContext>::BoxCoxNonzeroLambda(
    int64_t D,
    const T* data_ptr,
    const T* lambda1,
    const T* lambda2,
    T k_eps,
    T* out) {
  caffe2::math::Add(D, data_ptr, lambda2, out, &context_);
  for (int64_t j = 0; j < D; j++) {
    out[j] = std::max(out[j], k_eps);
  }
  Pow(D, out, lambda1, out);
  for (int64_t j = 0; j < D; j++) {
    out[j] -= 1.0;
  }
  caffe2::math::Div(D, out, lambda1, out, &context_);
}

template <>
template <typename T>
void BatchBoxCoxOp<CPUContext>::BoxCoxZeroLambda(
    int64_t D,
    const T* data_ptr,
    const T* lambda2,
    T k_eps,
    T* output_ptr) {
  caffe2::math::Add(D, data_ptr, lambda2, output_ptr, &context_);
  for (int64_t j = 0; j < D; j++) {
    output_ptr[j] = std::max(output_ptr[j], k_eps);
  }
  caffe2::math::Log(D, output_ptr, output_ptr, &context_);
}

template <>
template <typename T>
void BatchBoxCoxOp<CPUContext>::BoxCoxMixedLambda(
    const T* data_ptr,
    const vector<int>& nonzeros,
    const vector<int>& zeros,
    const T* lambda1,
    const T* lambda2,
    const T* lambda2_z,
    T k_eps,
    T* buffer,
    T* output_ptr) {
  PackV(nonzeros.size(), data_ptr, nonzeros.data(), buffer);
  BoxCoxNonzeroLambda(nonzeros.size(), buffer, lambda1, lambda2, k_eps, buffer);
  UnpackV(nonzeros.size(), buffer, output_ptr, nonzeros.data());

  PackV(zeros.size(), data_ptr, zeros.data(), buffer);
  BoxCoxZeroLambda(zeros.size(), buffer, lambda2_z, k_eps, buffer);
  UnpackV(zeros.size(), buffer, output_ptr, zeros.data());
}

// Helpers to access cached buffers.
#define DEFINE_CACHED_BUFFERS(T, tag)                                         \
  template <>                                                                 \
  template <>                                                                 \
  BatchBoxCoxOp<CPUContext>::TypedCachedBuffers<T>&                           \
  BatchBoxCoxOp<CPUContext>::GetBuffers<T>() {                                \
    if (!buffers_ || buffers_->type_ != tag) {                                \
      buffers_.reset(new BatchBoxCoxOp<CPUContext>::TypedCachedBuffers<T>()); \
      buffers_->type_ = tag;                                                  \
    }                                                                         \
    return *static_cast<TypedCachedBuffers<T>*>(buffers_.get());              \
  }
DEFINE_CACHED_BUFFERS(float, 1);
DEFINE_CACHED_BUFFERS(double, 2);
#undef DEFINE_CACHED_BUFFERS

#endif // CAFFE2_USE_MKL

namespace {

REGISTER_CPU_OPERATOR(BatchBoxCox, BatchBoxCoxOp<CPUContext>);
OPERATOR_SCHEMA(BatchBoxCox)
    .NumInputs(3)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Input `data` is a N * D matrix. Apply box-cox transform for each column.
`lambda1` and `lambda2` is of size D that defines the hyper-parameters for
the transform of each column `x` of the input `data`:

    ln(x + lambda2), if lambda1 == 0
    ((x + lambda2)^lambda1 - 1)/lambda1, if lambda1 != 0

)DOC")
    .Input(0, "data", "input float or double N * D matrix")
    .Input(1, "lambda1", "tensor of size D with the same type as data")
    .Input(2, "lambda2", "tensor of size D with the same type as data")
    .Output(0, "output", "output matrix that applied box-cox transform");

GRADIENT_NOT_IMPLEMENTED_YET(BatchBoxCox);
} // namespace
} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    BatchBoxCox,
    "_caffe2::BatchBoxCox(Tensor data, Tensor lambda1, Tensor lambda2, int min_block_size = 256) -> Tensor results",
    caffe2::BatchBoxCoxOp<caffe2::CPUContext>);
