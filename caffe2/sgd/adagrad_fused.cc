#include "caffe2/sgd/adagrad_fused.h"

namespace caffe2 {

namespace {

struct adagrad_update_prefetch_inlined {
  void operator()(
      int N,
      const float* w,
      const float* w_n, // prefetch ptr
      const float* g,
      const float* h,
      const float* h_n, // prefetch ptr
      float* nw,
      float* nw_n, // prefetch ptr
      float* nh,
      float* nh_n, // prefetch ptr
      float epsilon,
      float lr) {
    return internal::adagrad_update_prefetch_inlined(
        N, w, w_n, g, h, h_n, nw, nw_n, nh, nh_n, epsilon, lr);
  }
};

OPERATOR_SCHEMA(SparseAdagradFusedWithSparseLengthsSumGradient)
    .NumInputs(6)
    .NumOutputs(2)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

Fused operator of
SparseLengthsIndicesInGradientSumGradient (gradient of SparseLengthsSum) +
SparseAdagrad.

Given inputs (param, moment, indices, grad, lr), runs the sparse AdaGrad
update on (param, grad, moment[indices], lr), and returns (new_param,
new_moment) as in the dense case. Additional input (lengths) is for fused
SparseLengthsIndicesInGradientSumGradient operator.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Moment history")
    .Input(
        2,
        "indices",
        "Integer vector containing indices of the first dimension of param for the slices that are being updated")
    .Input(3, "grad", "Gradient computed")
    .Input(4, "lr", "learning rate")
    .Input(
        5,
        "lengths",
        "Non negative vector with sum of elements equal to indices length")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment", "Updated moment")
    .Arg("epsilon", "Default 1e-5");

REGISTER_CPU_OPERATOR(
    SparseAdagradFusedWithSparseLengthsSumGradient,
    SparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        float,
        int,
        adagrad_update_prefetch_inlined>);

OPERATOR_SCHEMA(SparseAdagradFusedWithSparseLengthsWeightedSumGradient)
    .NumInputs(7)
    .NumOutputs(3)
    .EnforceInplace({{0, 0}, {1, 1}})
    .SetDoc(R"DOC(

Fused operator of SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient
(gradient of SparseLengthsWeightedSum) + SparseAdagrad, where weights are
positional weights computed with LengthsRangeFill + Gather pattern.

Given inputs (param, moment, indices, grad, lr), runs the sparse AdaGrad
update on (param, grad, moment[indices], lr), and returns (new_param,
new_moment) as in the dense case.
There're auxiliary inputs (aux_param) for which gradient is computed
and returns (aux_grad).
Yet additional input (lengths) is for fused
SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient operator.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Moment history")
    .Input(2, "aux_param", "Auxiliary parameters to be updated")
    .Input(
        3,
        "indices",
        "Integer vector containing indices of the first dimension of param for the slices that are being updated")
    .Input(4, "grad", "Gradient computed")
    .Input(5, "lr", "learning rate")
    .Input(
        6,
        "lengths",
        "Non negative vector with sum of elements equal to indices length")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment", "Updated moment")
    .Output(2, "aux_grad", "Auxiliary gradient")
    .Arg("epsilon", "Default 1e-5");

REGISTER_CPU_OPERATOR(
    SparseAdagradFusedWithSparseLengthsWeightedSumGradient,
    SparseAdagradFusedWithSparseLengthsWeightedSumGradientOp<
        float,
        float,
        int,
        adagrad_update_prefetch_inlined>);

OPERATOR_SCHEMA(SparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox)
    .NumInputs(7)
    .NumOutputs(3)
    .EnforceInplace({{0, 0}, {1, 1}})
    .SetDoc(R"DOC(

Approximately fused operator of
SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient
(gradient of SparseLengthsWeightedSum) + SparseAdagrad, where weights are
positional weights computed with LengthsRangeFill + Gather pattern.

Given inputs (param, moment, indices, grad, lr), runs the sparse AdaGrad
update on (param, grad, moment[indices], lr), and returns (new_param,
new_moment) as in the dense case.
There's race condition w.r.t. ordering between reading params and writing to
param, hence the name Approx.
There're auxiliary inputs (aux_param) for which gradient is computed and
returns (aux_grad).
Yet additional input (lengths) is for fused
SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient operator.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Moment history")
    .Input(2, "aux_param", "Auxiliary parameters to be updated")
    .Input(
        3,
        "indices",
        "Integer vector containing indices of the first dimension of param for the slices that are being updated")
    .Input(4, "grad", "Gradient computed")
    .Input(5, "lr", "learning rate")
    .Input(
        6,
        "lengths",
        "Non negative vector with sum of elements equal to indices length")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment", "Updated moment")
    .Output(2, "aux_grad", "Auxiliary gradients")
    .Arg("epsilon", "Default 1e-5");

REGISTER_CPU_OPERATOR(
    SparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox,
    SparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp<
        float,
        float,
        int,
        adagrad_update_prefetch_inlined>);

} // namespace
} // namespace caffe2
