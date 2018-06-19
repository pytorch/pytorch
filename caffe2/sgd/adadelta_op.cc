/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "adadelta_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(Adadelta, AdadeltaOp<float, CPUContext>);
OPERATOR_SCHEMA(Adadelta)
    .NumInputs(4)
    .NumOutputs(3)
    .AllowInplace({{0, 0}, {1, 1}, {2, 2}})
    .SetDoc(R"DOC(

Computes the Adadelta update for an input gradient and accumulated
history. Concretely, given inputs (param, grad, moment, moment_delta),
computes

    new_moment = decay * moment + square(grad)
    new_grad = grad * (sqrt(moment_delta) + epsilon) / (sqrt(new_moment) + epsilon)
    new_moment_delta = decay * moment_delta + square(delta_param)
    new_param = param + new_grad
and returns (new_param, new_moment, new_moment_delta).

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Moment of squared gradient")
    .Input(2, "moment_delta", "Delta_param history")
    .Input(3, "grad", "Gradient computed")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment", "Updated moment")
    .Output(2, "output_moment_delta", "Updated moment_delta")
    .Arg("epsilon", "Default 1e-5")
    .Arg(
        "decay",
        "Default 1. If it is in (0, 1), the gradient square sum "
        "is decayed by this factor.");

REGISTER_CPU_OPERATOR(SparseAdadelta, SparseAdadeltaOp<float, CPUContext>);
OPERATOR_SCHEMA(SparseAdadelta)
    .NumInputs(5)
    .NumOutputs(3)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

Given inputs (param, moment, moment_delta, indices, grad), runs the dense Adadelta
update on (param, grad, moment[indices], moment_delta[indices]), and returns (new_param,
new_moment, new_moment_delta) as in the dense case.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Moment of squared gradient")
    .Input(2, "moment_delta", "Delta_param history")
    .Input(3, "indices", "Sparse indices")
    .Input(4, "grad", "Gradient computed")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment_grad", "Updated moment")
    .Output(2, "output_moment_delta", "Updated moment_delta")
    .Arg("epsilon", "Default 1e-5")
    .Arg(
        "decay",
        "Default 1. If it is in (0, 1), the gradient square sum "
        "is decayed by this factor.");

REGISTER_CPU_OPERATOR(
    RowWiseSparseAdadelta,
    RowWiseSparseAdadeltaOp<float, CPUContext>);
OPERATOR_SCHEMA(RowWiseSparseAdadelta)
    .NumInputs(5)
    .NumOutputs(3)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(

Given inputs (param, moment, indices, grad, lr), runs a modified sparse Adadelta
update on (param, grad, moment[indices], lr), and returns (new_param,
new_momwnr), where moment is a 1D tensor with length equal to the number of
rows in param: shape(moment) == shape(param)[0]. Each element of moment is
applied to an entire row of param, and the new moment is calculated by adding
the average squared sum of gradients across each row. Note that indices must
also be a 1D tensor indexing into the rows of param.

)DOC")
    .Input(0, "param", "Parameters to be updated")
    .Input(1, "moment", "Moment of squared gradient")
    .Input(2, "moment_delta", "Delta_param history")
    .Input(3, "indices", "Sparse indices")
    .Input(4, "grad", "Gradient computed")
    .Output(0, "output_param", "Updated parameters")
    .Output(1, "output_moment_grad", "Updated moment")
    .Output(2, "output_moment_delta", "Updated moment")
    .Arg("epsilon", "Default 1e-5")
    .Arg(
        "decay",
        "Default 1. If it is in (0, 1), the gradient square sum "
        "is decayed by this factor.");

SHOULD_NOT_DO_GRADIENT(Adadelta);
SHOULD_NOT_DO_GRADIENT(SparseAdadelta);
SHOULD_NOT_DO_GRADIENT(RowWiseSparseAdadelta);
} // namespace caffe2
