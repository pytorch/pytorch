#include "caffe2/operators/elementwise_ops.h"

#include <algorithm>

namespace caffe2 {

std::vector<int> ComputeBinaryBroadcastForwardDims(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims) {
  const int ndim = std::max(A_dims.size(), B_dims.size());
  std::vector<int> C_dims(ndim);
  int i = A_dims.size() - 1;
  int j = B_dims.size() - 1;
  int k = ndim - 1;
  for (; i >= 0 && j >= 0; --k) {
    CAFFE_ENFORCE(A_dims[i] == B_dims[j] || A_dims[i] == 1 || B_dims[j] == 1);
    C_dims[k] = std::max(A_dims[i--], B_dims[j--]);
  }
  for (; i >= 0; --i) {
    C_dims[k--] = A_dims[i];
  }
  for (; j >= 0; --j) {
    C_dims[k--] = B_dims[j];
  }
  return C_dims;
}

void ComputeBinaryBroadcastBackwardAxes(
    const std::vector<int>& A_dims,
    const std::vector<int>& B_dims,
    std::vector<int>* A_axes,
    std::vector<int>* B_axes) {
  A_axes->clear();
  B_axes->clear();
  const int ndim = std::max(A_dims.size(), B_dims.size());
  int i = A_dims.size() - 1;
  int j = B_dims.size() - 1;
  int k = ndim - 1;
  for (; i >= 0 && j >= 0; --k) {
    CAFFE_ENFORCE(A_dims[i] == B_dims[j] || A_dims[i] == 1 || B_dims[j] == 1);
    if (A_dims[i] != B_dims[j]) {
      if (A_dims[i] == 1) {
        A_axes->push_back(k);
      }
      if (B_dims[j] == 1) {
        B_axes->push_back(k);
      }
    }
    --i;
    --j;
  }
  if (i < 0) {
    for (; k >= 0; --k) {
      A_axes->push_back(k);
    }
  } else {
    for (; k >= 0; --k) {
      B_axes->push_back(k);
    }
  }
  std::reverse(A_axes->begin(), A_axes->end());
  std::reverse(B_axes->begin(), B_axes->end());
}

REGISTER_CPU_OPERATOR(
    Not,
    UnaryElementwiseOp<BoolTypes, CPUContext, NotFunctor<CPUContext>>);
REGISTER_CPU_OPERATOR(
    Sign,
    UnaryElementwiseOp<NumericTypes, CPUContext, SignFunctor<CPUContext>>);

#define REGISTER_CPU_COMPARE_OPERATOR(Op)                     \
  REGISTER_CPU_OPERATOR(                                      \
      Op,                                                     \
      BinaryElementwiseOp<                                    \
          TensorTypes<bool, int32_t, int64_t, float, double>, \
          CPUContext,                                         \
          Op##Functor<CPUContext>,                            \
          FixedType<bool>>)

REGISTER_CPU_COMPARE_OPERATOR(EQ);
REGISTER_CPU_COMPARE_OPERATOR(NE);
REGISTER_CPU_COMPARE_OPERATOR(LT);
REGISTER_CPU_COMPARE_OPERATOR(LE);
REGISTER_CPU_COMPARE_OPERATOR(GT);
REGISTER_CPU_COMPARE_OPERATOR(GE);

#undef REGISTER_CPU_COMPARE_OPERATOR

#define REGISTER_CPU_LOGICAL_BINARY_OPERATOR(Op) \
  REGISTER_CPU_OPERATOR(                         \
      Op, BinaryElementwiseOp<BoolTypes, CPUContext, Op##Functor<CPUContext>>)

REGISTER_CPU_LOGICAL_BINARY_OPERATOR(And);
REGISTER_CPU_LOGICAL_BINARY_OPERATOR(Or);
REGISTER_CPU_LOGICAL_BINARY_OPERATOR(Xor);

#undef REGISTER_CPU_LOGICAL_BINARY_OPERATOR

#define REGISTER_CPU_BITWISE_BINARY_OPERATOR(Op) \
  REGISTER_CPU_OPERATOR(                         \
      Op,                                        \
      BinaryElementwiseOp<IntBoolTypes, CPUContext, Op##Functor<CPUContext>>)

REGISTER_CPU_BITWISE_BINARY_OPERATOR(BitwiseAnd);
REGISTER_CPU_BITWISE_BINARY_OPERATOR(BitwiseOr);
REGISTER_CPU_BITWISE_BINARY_OPERATOR(BitwiseXor);

#undef REGISTER_CPU_BITWISE_BINARY_OPERATOR

template <typename T>
void SRLHelper::sum2one(const T* x, T* y, size_t n) {
  *y = ConstEigenArrayMap<T>(x, n, 1).sum();
}

template <typename T>
void SRLHelper::RunWithBroadcastFront(
    const T* x,
    T* y,
    size_t pre,
    size_t n,
    CPUContext*) {
  EigenArrayMap<T>(y, n, 1) = ConstEigenArrayMap<T>(x, n, pre).rowwise().sum();
}

template <typename T>
void SRLHelper::RunWithBroadcastBack(
    const T* x,
    T* y,
    size_t post,
    size_t n,
    CPUContext*) {
  EigenArrayMap<T>(y, 1, n) = ConstEigenArrayMap<T>(x, post, n).colwise().sum();
}

template <typename T>
void SRLHelper::RunWithBroadcast2(
    const T* a,
    T* y,
    size_t pre,
    size_t n,
    size_t post,
    CPUContext*) {
  for (int i = 0; i < n; ++i) {
    y[i] = 0;
    for (int j = 0; j < pre; ++j) {
      for (int k = 0; k < post; ++k) {
        y[i] += a[(j * n + i) * post + k];
      }
    }
  }
}

template <>
template <typename T>
bool SumReduceLikeOp<CPUContext>::DoRunWithType() {
  const auto& A = Input(0);
  const auto& B = Input(1);
  auto* C = Output(0);
  CAFFE_ENFORCE(&B != C, "In-place is not allowed.");
  C->ResizeLike(B);
  const T* Adata = A.template data<T>();
  auto* Cdata = C->template mutable_data<T>();
  if (B.size() == 1) {
    auto count = A.size();
    SRLHelper::sum2one<T>(Adata, Cdata, count);
  } else {
    size_t pre, n, post;
    std::tie(pre, n, post) = ComputeLegacyBroadcastSizes(A, B, axis_);
    if (post == 1) {
      SRLHelper::RunWithBroadcastFront<T>(Adata, Cdata, pre, n, &context_);
    } else if (pre == 1) {
      SRLHelper::RunWithBroadcastBack<T>(Adata, Cdata, post, n, &context_);
    } else {
      SRLHelper::RunWithBroadcast2<T>(Adata, Cdata, pre, n, post, &context_);
    }
  }
  return true;
}

REGISTER_CPU_OPERATOR(SumReduceLike, SumReduceLikeOp<CPUContext>);

} // namespace caffe2
