#include "caffe2/operators/utility_ops.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(Print, PrintOp<CPUContext>);
REGISTER_CPU_OPERATOR(Flatten, FlattenOp<CPUContext>);
REGISTER_CPU_OPERATOR(Alias, AliasOp<CPUContext>);
REGISTER_CPU_OPERATOR(ResizeLike, ResizeLikeOp<CPUContext>);
REGISTER_CPU_OPERATOR(Sum, SumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(SumInt, SumOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(WeightedSum, WeightedSumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    ScatterWeightedSum,
    ScatterWeightedSumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ScatterAssign, ScatterAssignOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(Copy, CopyOp<CPUContext, CPUContext, CPUContext>);
REGISTER_CPU_OPERATOR(Shape, ShapeOp<CPUContext>);
REGISTER_CPU_OPERATOR(HasElements, HasElementsOp<CPUContext>);
REGISTER_CPU_OPERATOR(Gather, GatherOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(Unique, UniqueOp<CPUContext>);
REGISTER_CPU_OPERATOR(LengthsToSegmentIds, LengthsToSegmentIdsOp<CPUContext>);
REGISTER_CPU_OPERATOR(SegmentIdsToLengths, SegmentIdsToLengthsOp<CPUContext>);
REGISTER_CPU_OPERATOR(Slice, SliceOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(Squeeze, SqueezeOp<CPUContext>);
REGISTER_CPU_OPERATOR(ExpandDims, ExpandDimsOp<CPUContext>);

OPERATOR_SCHEMA(Print)
    .NumInputs(1)
    .NumOutputs(0)
    .SetDoc("Logs shape and contents of input tensor to stderr or to a file.")
    .Arg(
        "to_file",
        "(bool) if 1, saves contents to the root folder of the current "
        "workspace, appending the tensor contents to a file named after "
        "the blob name. Otherwise, logs to stderr.")
    .Input(0, "tensor", "The tensor to print.");

OPERATOR_SCHEMA(Flatten)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Flattens the input tensor into a 2D matrix, keeping the first dimension
unchanged.
)DOC")
    .Input(0, "input", "A tensor of rank >= 2.")
    .Output(
        0,
        "output",
        "A tensor of rank 2 with the contents of the input tensor, "
        "with first dimension equal first dimension of input, and remaining "
        "input dimensions flatenned into the inner dimension of the output.");

OPERATOR_SCHEMA(Alias)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Makes the output and the input share the same underlying storage.

WARNING: in general, in caffe2's operator interface different tensors should
have different underlying storage, which is the assumption made by
components such as the dependency engine and memory optimization. Thus, in
normal situations you should not use the AliasOp, especially in a normal
forward-backward pass.

The Alias op is provided so one can achieve true asynchrony, such as
Hogwild, in a graph. But make sure you understand all the implications
similar to multi-thread computation before you use it explicitly.
)DOC")
    .Input(0, "input", "Input tensor whose storage will be shared.")
    .Output(0, "output", "Tensor of same shape as input, sharing its storage.");

OPERATOR_SCHEMA(ResizeLike)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Produces tensor condaining data of first input and shape of second input.
)DOC")
    .Input(0, "data", "Tensor whose data will be copied into the output.")
    .Input(1, "shape_tensor", "Tensor whose shape will be applied to output.")
    .Output(0, "output", "Tensor with data of input 0 and shape of input 1.");

OPERATOR_SCHEMA(SumInt)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(Sum)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Element-wise sum of each of the input tensors. The first input tensor can be
used in-place as the output tensor, in which case the sum will be done in
place and results will be accumulated in input0. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "First of the input tensors. Can be inplace.")
    .Output(0, "sum", "Output tensor. Same dimension as inputs.");

OPERATOR_SCHEMA(WeightedSum)
    .NumInputs([](int n) { return (n > 0 && n % 2 == 0); })
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Element-wise weighted sum of several data, weight tensor pairs.
Input should be in the form X_0, weight_0, X_1, weight_1, ... where X_i all
have the same shape, and weight_i are size 1 tensors that specifies the weight
of each vector. Note that if one wants to do in-place computation, it could
only be done with X_0 also as the output, but not other X_i.
)DOC")
    .Input(0, "data_0", "First of the input tensors.")
    .Input(0, "weight_0", "Weight of the first input in the sum.")
    .Output(0, "output", "Result containing weighted elem-wise sum of inputs.");

OPERATOR_SCHEMA(ScatterWeightedSum)
    .NumInputs([](int n) { return (n > 3 && (n - 3) % 2 == 0); })
    .NumOutputs(1)
    .EnforceInplace({{0, 0}})
    .SetDoc(R"DOC(
Similar to WeightedSum, computes the weighted sum of several tensors, with
the difference that inputs are sliced tensors. The first tensor has to be
in-place and only slices of it on the first dimension as indexed by INDICES
will be updated.

Note: The op pretty much ignores the exact shapes of the input arguments and
cares only about sizes. It's done for performance consideration to avoid
unnecessary reshapes. Only first dimension of X_0 is important, let's call it
N. If M is the total size of X_0 and K is the size of INDICES then X_i is
assumed to be of shape K x (M / N) regardless of the real shape.

Note: Each update in INDICES is applied independently which means that if
duplicated elements are present in INDICES the corresponding slice of X_0
will be scaled multiple times. Manual collapsing of INDICES is required
beforehand if necessary.

Note: Updates are applied sequentially by inputs which might have undesired
consequences if the input tensor is accessed concurrently by different op
(e.g. when doing Hogwild). Other threads might see intermediate results even
on individual slice level, e.g. X_0 scaled by weight_0 but without any
updates applied.

Currently only works on CPU because of access to INDICES.
)DOC")
    .Input(0, "X_0", "Tensor to be updated.")
    .Input(
        1,
        "Weight_0",
        "Scalar weight for X_0, applied only to slices affected.")
    .Input(
        2,
        "INDICES",
        "1-D list of indices on the first dimension of X_0 "
        "that need to be updated")
    .Input(3, "X_1", "Update slices, with shape len(INDICES) + shape(X_0)[1:]")
    .Input(4, "Weight_1", "Scalar weight for X_1 update")
    .Output(0, "X_0", "Has to be exactly the same tensor as the input 0")
    .EnforceInplace({{0, 0}});

OPERATOR_SCHEMA(ScatterAssign)
    .NumInputs(3)
    .NumOutputs(1)
    .EnforceInplace({{0, 0}})
    .SetDoc(R"DOC(
Update slices of the tensor in-place by overriding current value.

Note: The op pretty much ignores the exact shapes of the input arguments and
cares only about sizes. It's done for performance consideration to avoid
unnecessary reshapes. Only first dimension of X_0 is important, let's call it
N. If M is the total size of X_0 and K is the size of INDICES then X_i is
assumed to be of shape K x (M / N) regardless of the real shape.

Note: Each update in INDICES is applied independently which means that if
duplicated elements are present in INDICES arbitrary one will win.

Currently only works on CPU because of access to INDICES.
)DOC")
    .Input(0, "DATA", "Tensor to be updated.")
    .Input(
        1,
        "INDICES",
        "1-D list of indices on the first dimension"
        "of X_0 that need to be updated")
    .Input(
        2,
        "SLICES",
        "Update slices, with shape len(INDICES) + shape(X_0)[1:]")
    .Output(0, "DATA", "Has to be exactly the same tensor as the input 0");

OPERATOR_SCHEMA(Copy)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc("Copy input tensor into output, potentially across devices.")
    .Input(0, "input", "The input tensor.")
    .Output(0, "output", "Tensor that will contain a copy of the input.");

OPERATOR_SCHEMA(Shape)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc("Produce a 1D int64 tensor with the shape of the input tensor.");

OPERATOR_SCHEMA(HasElements)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc("Returns true iff the input tensor has size > 0")
    .Input(0, "tensor", "Tensor of any type.")
    .Output(
        0,
        "has_elements",
        "Scalar bool tensor. True if input is not empty.");

OPERATOR_SCHEMA(Gather)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given DATA tensor of rank r >= 1, and INDICES tensor of rank q, gather
entries of the outer-most dimension of DATA indexed by INDICES, and concatenate
them in an output tensor of rank q + (r - 1).

Example:
  DATA  = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  INDICES = [
      [0, 1],
      [1, 2],
  ]
  OUTPUT = [
      [
          [1.0, 1.2],
          [2.3, 3.4],
      ],
      [
          [2.3, 3.4],
          [4.5, 5.7],
      ],
  ]
)DOC")
    .Input(0, "DATA", "Tensor of rank r >= 1.")
    .Input(1, "INDICES", "Tensor of int32/int64 indices, of any rank q.")
    .Output(0, "OUTPUT", "Tensor of rank q + (r - 1).");

OPERATOR_SCHEMA(Unique)
    .NumInputs(1)
    .NumOutputs(1, 2)
    .SetDoc(R"DOC(
Deduplicates input indices vector and optionally produces reverse remapping.
There's no guarantees on the ordering of the output indices.
)DOC")
    .Input(0, "indices", "1D tensor of int32 or int64 indices.")
    .Output(0, "unique_indices", "1D tensor of deduped entries.");

OPERATOR_SCHEMA(LengthsToSegmentIds)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given a vector of segment lengths, returns a zero-based, consecutive vector
of segment_ids. For example, [1, 3, 0, 2] will produce [0, 1, 1, 1, 3, 3].
In general, the inverse operation is SegmentIdsToLengths. Notice though that
trailing empty sequence lengths can't be properly recovered from segment ids.
)DOC")
    .Input(0, "lengths", "1D tensor of int32 or int64 segment lengths.")
    .Output(0, "has_elements", "Scalar bool. True iff input is not empty.");

OPERATOR_SCHEMA(SegmentIdsToLengths)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Transfers a vector of segment ids to a vector of segment lengths. This operation
supports non-consecutive segment ids. Segments not appearing in the input vector
will have length 0. The range of segments of interest has ids [0, last id in the
input vector].
In general, for consecutive, zero-based segment IDs, this is the inverse
operation of LengthsToSegmentIds, except that a vector of segment IDs
cannot represent empty segments at the end.
)DOC")
    .Input(0, "segment_ids", "1-D int32_t or int64_t tensor of segment ids")
    .Output(0, "segment_lengths", "1-D int64_t tensor of segment lengths");

OPERATOR_SCHEMA(Slice)
    .NumInputs(3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Produces a slice of the input tensor. Currently, only slicing in a single
dimension is supported.
Slices are passed as 2 1D vectors with starting and end indices for each
dimension of the input `data` tensor. End indices are non-inclusive. If
a negative value is passed for any of the start or end indices, it
represent number of elements before the end of that dimension.

Example:

  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 3]

  result = [
      [2, 3],
      [6, 7],
  ]
)DOC")
    .Input(0, "data", "Tensor of data to extract slices from.")
    .Input(1, "starts", "1D tensor: start-indices for each dimension of data.")
    .Input(2, "ends", "1D tensor: end-indices for each dimension of data.")
    .Output(0, "output", "Sliced data tensor.");

OPERATOR_SCHEMA(Squeeze)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Remove single-dimensional entries from the shape of a tensor.
Takes an optional parameter `dims` with a list of dimension to squeeze.
If `dims` is not provided, all singleton dimensions are squeezed.
If the same blob is provided in input and output, the operation is copy-free.
This is the exact inverse operation of ExpandDims given the same `dims` arg.
)DOC")
    .Input(0, "data", "Tensors with at least max(dims) dimensions.")
    .Output(0, "squeezed", "Reshaped tensor with same data as input.");

OPERATOR_SCHEMA(ExpandDims)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Insert single-dimensional entries to the shape of a tensor.
Takes one required argument `dims`, a list of dimensions that will be inserted.
Dimension indices in `dims` are as seen in the output tensor. For example:

  Given a tensor such that tensor.Shape() = [3, 4, 5], then
  ExpandDims(tensor, dims=[0, 4]).Shape() == [1, 3, 4, 5, 1])

If the same blob is provided in input and output, the operation is copy-free.
)DOC")
    .Input(0, "data", "Original tensor")
    .Output(0, "expanded", "Reshaped tensor with same data as input.");

SHOULD_NOT_DO_GRADIENT(Print);
SHOULD_NOT_DO_GRADIENT(Shape);
SHOULD_NOT_DO_GRADIENT(HasElements);

class GetSqueezeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ExpandDims",
        "",
        vector<string>{GO(0)},
        vector<string>{GI(0)},
        Def().arg());
  }
};
REGISTER_GRADIENT(Squeeze, GetSqueezeGradient);

class GetExpandDimsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Squeeze",
        "",
        vector<string>{GO(0)},
        vector<string>{GI(0)},
        Def().arg());
  }
};
REGISTER_GRADIENT(ExpandDims, GetExpandDimsGradient);

class GetFlattenGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ResizeLike", "", vector<string>{GO(0), I(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Flatten, GetFlattenGradient);

class GetAliasGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // We will simply pass-along the gradient. Nothing needs to
    // be calculated.
    SetDense(0, GO(0));
    return vector<OperatorDef>();
  }
};
REGISTER_GRADIENT(Alias, GetAliasGradient);

SHOULD_NOT_DO_GRADIENT(ResizeLike);

class GetSumGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    for (auto i = 0; i < def_.input_size(); ++i) {
      SetDense(i, GO(0));
    }
    return vector<OperatorDef>();
  }
};
REGISTER_GRADIENT(Sum, GetSumGradient);

// TODO(jiayq): Weighted sum is originally intended to be used in SGD, but in
// theory, its gradient DOES exist. Should we enable the gradient?
SHOULD_NOT_DO_GRADIENT(WeightedSum);
SHOULD_NOT_DO_GRADIENT(ScatterWeightedSum);
SHOULD_NOT_DO_GRADIENT(ScatterAssign);

// TODO(jiayq): Copy is a bit tricky because one need to figure out correctly
// where the input lies (e.g. for muji, which gpu). Right now I am marking it
// as not gradient ready.
SHOULD_NOT_DO_GRADIENT(Copy);

class GetGatherGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    // For now we don't do any reshaping as the consumer of this op would
    // probably be ScatterUpdate which is intenionally ignores shapes. We might
    // need to revisit it in the future for correctness purposes. The right
    // shape for the output woild be to flatten INDICES and collapse first X
    // dims of GRAD
    using Op = GatherOp<float, CPUContext>;
    SetSparse(Op::DATA, I(Op::INDICES), GO(0));
    return vector<OperatorDef>();
  }
};
REGISTER_GRADIENT(Gather, GetGatherGradient);
SHOULD_NOT_DO_GRADIENT(Unique);
SHOULD_NOT_DO_GRADIENT(LengthsToSegmentIds);
SHOULD_NOT_DO_GRADIENT(SegmentIdsToLengths);
// TODO(azzolini): Add support for slice gradient
SHOULD_NOT_DO_GRADIENT(Slice);

} // namespace

} // namespace caffe2
