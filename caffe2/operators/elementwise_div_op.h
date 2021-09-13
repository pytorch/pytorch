#ifndef CAFFE2_OPERATORS_ELEMENTWISE_DIV_OP_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_DIV_OP_H_

#include <vector>

#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
struct DivFunctor {
  explicit DivFunctor(bool allow_broadcast_fastpath=false)
    : allow_broadcast_fastpath_(allow_broadcast_fastpath) {}

  template <typename TIn, typename TOut>
  bool Forward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TIn* A,
      const TIn* B,
      TOut* C,
      Context* context) const {
    math::Div(
        A_dims.size(),
        A_dims.data(),
        B_dims.size(),
        B_dims.data(),
        A,
        B,
        C,
        context);
    return true;
  }

  template <typename TGrad, typename TIn, typename TOut>
  bool Backward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TGrad* dC_data,
      const TIn* A_data,
      const TIn* B_data,
      const TOut* C_data,
      TGrad* dA_data,
      TGrad* dB_data,
      Context* context) const;

  const bool allow_broadcast_fastpath_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ELEMENTWISE_DIV_OP_H_
