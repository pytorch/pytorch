#include "caffe2/sgd/rowwise_adagrad_fused.h"

namespace caffe2 {

OPERATOR_SCHEMA(RowWiseSparseAdagradFusedWithSparseLengthsSumGradient)
    .NumInputs(6)
    .NumOutputs(2)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

Fused operator of
SparseLengthsIndicesInGradientSumGradient (gradient of SparseLengthsSum) +
RowWiseSparseAdagrad.

Given inputs (param, moment, indices, grad, lr), runs the row-wise sparse
AdaGrad update on (param, grad, moment[indices], lr), and returns (new_param,
new_moment) as in the dense case. Additional input (lengths) is for fused
SparseLengthsSumGradient operator.

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
    .Arg(
        "round_option",
        "rounding option: 0 for nearest rounding, 1 for stochastic rounding")
    .Arg("epsilon", "Default 1e-5");

REGISTER_CPU_OPERATOR(
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradient,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        float,
        int,
        rowwise_adagrad_update_inlined,
        /*is_mean=*/false>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradient,
    SIMD,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        float,
        int,
        rowwise_adagrad_update_inlined,
        /*is_mean=*/false>);

// Match the GPU Approx op, here Approx and Exact are the same for
// RowWiseSparseAdagradFusedWithSparseLengthsSumGradient op
OPERATOR_SCHEMA(RowWiseSparseAdagradFusedWithSparseLengthsSumGradientApprox)
    .NumInputs(6)
    .NumOutputs(2)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

Fused operator of
SparseLengthsIndicesInGradientSumGradient (gradient of SparseLengthsSum) +
RowWiseSparseAdagrad.

Given inputs (param, moment, indices, grad, lr), runs the row-wise sparse
AdaGrad update on (param, grad, moment[indices], lr), and returns (new_param,
new_moment) as in the dense case. Additional input (lengths) is for fused
SparseLengthsSumGradient operator.

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
    .Arg(
        "round_option",
        "rounding option: 0 for nearest rounding, 1 for stochastic rounding")
    .Arg("epsilon", "Default 1e-5");

REGISTER_CPU_OPERATOR(
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientApprox,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        float,
        int,
        rowwise_adagrad_update_inlined,
        /*is_mean=*/false>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientApprox,
    SIMD,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        float,
        int,
        rowwise_adagrad_update_inlined,
        /*is_mean=*/false>);

OPERATOR_SCHEMA(RowWiseSparseAdagradFusedWithSparseLengthsMeanGradient)
    .NumInputs(6)
    .NumOutputs(2)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

Fused operator of
SparseLengthsIndicesInGradientMeanGradient (gradient of SparseLengthsMean) +
RowWiseSparseAdagrad.

Given inputs (param, moment, indices, grad, lr), runs the row-wise sparse
AdaGrad update on (param, grad, moment[indices], lr), and returns (new_param,
new_moment) as in the dense case. Additional input (lengths) is for fused
SparseLengthsMeanGradient operator.

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
    RowWiseSparseAdagradFusedWithSparseLengthsMeanGradient,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        float,
        int,
        rowwise_adagrad_update_inlined,
        /*is_mean=*/true>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    RowWiseSparseAdagradFusedWithSparseLengthsMeanGradient,
    SIMD,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        float,
        int,
        rowwise_adagrad_update_inlined,
        /*is_mean=*/true>);

// Match the GPU Approx op, here Approx and Exact are the same for
// RowWiseSparseAdagradFusedWithSparseLengthsMeanGradient op
OPERATOR_SCHEMA(RowWiseSparseAdagradFusedWithSparseLengthsMeanGradientApprox)
    .NumInputs(6)
    .NumOutputs(2)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

Fused operator of
SparseLengthsIndicesInGradientMeanGradient (gradient of SparseLengthsMean) +
RowWiseSparseAdagrad.

Given inputs (param, moment, indices, grad, lr), runs the row-wise sparse
AdaGrad update on (param, grad, moment[indices], lr), and returns (new_param,
new_moment) as in the dense case. Additional input (lengths) is for fused
SparseLengthsMeanGradient operator.

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
    .Arg(
        "round_option",
        "rounding option: 0 for nearest rounding, 1 for stochastic rounding")
    .Arg("epsilon", "Default 1e-5");

REGISTER_CPU_OPERATOR(
    RowWiseSparseAdagradFusedWithSparseLengthsMeanGradientApprox,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        float,
        int,
        rowwise_adagrad_update_inlined,
        /*is_mean=*/true>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    RowWiseSparseAdagradFusedWithSparseLengthsMeanGradientApprox,
    SIMD,
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        float,
        int,
        rowwise_adagrad_update_inlined,
        /*is_mean=*/true>);

OPERATOR_SCHEMA(RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradient)
    .NumInputs(7)
    .NumOutputs(3)
    .EnforceInplace({{0, 0}, {1, 1}})
    .SetDoc(R"DOC(

Fused operator of SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient
(gradient of SparseLengthsWeightedSum) + RowWiseSparseAdagrad, where weights are
positional weights computed with LengthsRangeFill + Gather pattern.

Given inputs (param, moment, indices, grad, lr), runs the row-wise sparse
AdaGrad update on (param, grad, moment[indices], lr), and returns (new_param,
new_moment) as in the dense case.
There're auxiliary inputs (aux_param) for which gradient is computed and
returns (aux_grad).
Yet additional input (lengths) is for fused SparseLengthsWeightedSumGradient
operator.

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
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradient,
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientOp<
        float,
        float,
        int,
        rowwise_adagrad_update_inlined>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradient,
    SIMD,
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientOp<
        float,
        float,
        int,
        rowwise_adagrad_update_inlined>);

OPERATOR_SCHEMA(
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox)
    .NumInputs(7)
    .NumOutputs(3)
    .EnforceInplace({{0, 0}, {1, 1}})
    .SetDoc(R"DOC(

Approximately fused operator of
SparseLengthsIndicesInGradientWeightedSumWithMainInputGradient
(gradient of SparseLengthsWeightedSum) + RowWiseSparseAdagrad, where weights are
positional weights computed with LengthsRangeFill + Gather pattern.

Given inputs (param, moment, indices, grad, lr), runs the row-wise sparse
AdaGrad update on (param, grad, moment[indices], lr), and returns (new_param,
new_moment) as in the dense case.
There's race condition w.r.t. ordering between reading params and writing to
param, hence the name Approx.
There're auxiliary inputs (aux_param) for which gradient is computed
and returns (aux_grad).
Yet additional input (lengths) is for fused SparseLengthsWeightedSumGradient
operator.

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
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox,
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp<
        float,
        float,
        int,
        rowwise_adagrad_update_inlined>);

REGISTER_CPU_OPERATOR_WITH_ENGINE(
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox,
    SIMD,
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApproxOp<
        float,
        float,
        int,
        rowwise_adagrad_update_inlined>);

} // namespace caffe2
