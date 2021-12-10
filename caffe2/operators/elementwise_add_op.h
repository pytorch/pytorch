#ifndef CAFFE2_OPERATORS_ELEMENTWISE_ADD_OP_H_
#define CAFFE2_OPERATORS_ELEMENTWISE_ADD_OP_H_

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe2/operators/elementwise_ops.h"
#include "caffe2/operators/elementwise_ops_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
struct AddFunctor {
  template <typename TIn, typename TOut>
  bool Forward(
      const std::vector<int>& A_dims,
      const std::vector<int>& B_dims,
      const TIn* A,
      const TIn* B,
      TOut* C,
      Context* context) const {
    math::Add(
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
      const TGrad* dC,
      const TIn* /* A */,
      const TIn* /* B */,
      const TOut* /* C */,
      TGrad* dA,
      TGrad* dB,
      Context* context) const {
    const std::vector<int> C_dims =
        elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
            A_dims, B_dims);
    std::vector<int> A_back_dims;
    std::vector<int> B_back_dims;
    elementwise_ops_utils::ComputeBinaryBroadcastBackwardDims(
        A_dims, B_dims, &A_back_dims, &B_back_dims);
    math::ReduceSum(
        C_dims.size(),
        C_dims.data(),
        A_back_dims.data(),
        TGrad(1),
        dC,
        dA,
        context,
        true);
    math::ReduceSum(
        C_dims.size(),
        C_dims.data(),
        B_back_dims.data(),
        TGrad(1),
        dC,
        dB,
        context,
        true);
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ELEMENTWISE_ADD_OP_H_
