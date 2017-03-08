#include "caffe2/operators/utility_ops.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(WallClockTime, WallClockTimeOp<CPUContext>);
REGISTER_CPU_OPERATOR(Print, PrintOp<CPUContext>);
REGISTER_CPU_OPERATOR(Flatten, FlattenOp<CPUContext>);
REGISTER_CPU_OPERATOR(FlattenToVec, FlattenToVecOp<CPUContext>);

REGISTER_CPU_OPERATOR(Alias, AliasOp<CPUContext>);
REGISTER_CPU_OPERATOR(ResizeLike, ResizeLikeOp<CPUContext>);
REGISTER_CPU_OPERATOR(Sum, SumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(SumInt, SumOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(SumElements, SumElementsOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    SumElementsGradient,
    SumElementsGradientOp<float, CPUContext>);

REGISTER_CPU_OPERATOR(WeightedSum, WeightedSumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(
    ScatterWeightedSum,
    ScatterWeightedSumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(Max, MaxOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(MaxGradient, MaxGradientOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ScatterAssign, ScatterAssignOp<float, CPUContext>);
// From whatever the current context, ensure the output is TensorCPU
REGISTER_CPU_OPERATOR(
    EnsureCPUOutput,
    CopyOp<CPUContext, CPUContext, CPUContext>);
// From CPU, copy it to whatever the current context
REGISTER_CPU_OPERATOR(
    CopyFromCPUInput,
    CopyOp<CPUContext, CPUContext, CPUContext>);
REGISTER_CPU_OPERATOR(
    CopyOnDeviceLike,
    CopyOnDeviceLikeOp<CPUContext, CPUContext, CPUContext>);
REGISTER_CPU_OPERATOR(Copy, CopyOp<CPUContext, CPUContext, CPUContext>);
REGISTER_CPU_OPERATOR(Shape, ShapeOp<CPUContext>);
REGISTER_CPU_OPERATOR(Reshape, ReshapeOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(LengthsToShape, LengthsToShapeOp<CPUContext>);
REGISTER_CPU_OPERATOR(HasElements, HasElementsOp<CPUContext>);
REGISTER_CPU_OPERATOR(IsEmpty, IsEmptyOp<CPUContext>);
REGISTER_CPU_OPERATOR(Gather, GatherOp<CPUContext>);
REGISTER_CPU_OPERATOR(GatherRanges, GatherRangesOp<CPUContext>);
REGISTER_CPU_OPERATOR(Unique, UniqueOp<CPUContext>);
REGISTER_CPU_OPERATOR(LengthsToSegmentIds, LengthsToSegmentIdsOp<CPUContext>);
REGISTER_CPU_OPERATOR(LengthsToRanges, LengthsToRangesOp<CPUContext>);
REGISTER_CPU_OPERATOR(SegmentIdsToLengths, SegmentIdsToLengthsOp<CPUContext>);
REGISTER_CPU_OPERATOR(SegmentIdsToRanges, SegmentIdsToRangesOp<CPUContext>);
REGISTER_CPU_OPERATOR(Slice, SliceOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(Squeeze, SqueezeOp<CPUContext>);
REGISTER_CPU_OPERATOR(ExpandDims, ExpandDimsOp<CPUContext>);
REGISTER_CPU_OPERATOR(LengthsToWeights, LengthsToWeightsOp<CPUContext>);
REGISTER_CPU_OPERATOR(EnsureDense, EnsureDenseOp<CPUContext>);

OPERATOR_SCHEMA(WallClockTime)
    .NumInputs(0)
    .NumOutputs(1)
    .SetDoc("Time since epoch in nanoseconds.")
    .Output(0, "time", "The time in nanoseconds.");

REGISTER_CPU_OPERATOR(UnsafeCoalesce, UnsafeCoalesceOp<CPUContext>);

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

OPERATOR_SCHEMA(LengthsToShape).NumInputs(1).NumOutputs(1);

OPERATOR_SCHEMA(Reshape)
    .NumInputs(1, 2)
    .NumOutputs(2)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Reshape the input tensor similar to numpy.reshape.

It takes a tensor as input and an optional tensor specifying the new shape.
When the second input is absent, an extra argument `shape` must be specified.
It outputs the reshaped tensor as well as the original shape.

At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is going to be copied
from the input tensor.
)DOC")
    .Arg("shape", "New shape")
    .Input(0, "data", "An input tensor.")
    .Input(1, "new_shape", "New shape.")
    .Output(0, "reshaped", "Reshaped data.")
    .Output(1, "old_shape", "Original shape.");

class GetReshapeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Reshape",
        "",
        vector<string>{GO(0), O(1)},
        vector<string>{GI(0), "_" + GI(0) + "_dims"});
  }

  // Argument `shape` is no longer needed in backprop.
  bool CopyArguments() const override {
    return false;
  }
};

REGISTER_GRADIENT(Reshape, GetReshapeGradient);

OPERATOR_SCHEMA(SumElements)
    .NumInputs(1)
    .NumOutputs(1)
    .ScalarType(TensorProto::FLOAT)
    .SetDoc("Sums the elements of the input tensor.")
    .Arg("average", "whether to average or not")
    .Input(0, "X", "Tensor to sum up")
    .Output(0, "sum", "Scalar sum");

class GetSumElementsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "SumElementsGradient",
        "",
        vector<string>{I(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(SumElements, GetSumElementsGradient);

OPERATOR_SCHEMA(Flatten)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(
        [](const OperatorDef&, const vector<TensorShape>& in) {
          vector<TensorShape> out(1);
          int total = 1;
          std::size_t index = 0;
          for (auto d : in[0].dims()) {
            // skip the first element
            if (index++ == 0) {
              continue;
            }
            total *= d;
          }
          out[0].add_dims(in[0].dims(0));
          out[0].add_dims(total);
          return out;
        })
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

OPERATOR_SCHEMA(FlattenToVec)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          vector<TensorShape> out(1);
          int total = 1;
          for (auto d : in[0].dims()) {
            total *= d;
          }
          out[0].add_dims(total);
          return out;
        })
    .SetDoc(R"DOC(
Flattens the input tensor into a 1D vector.
)DOC")
    .Input(0, "input", "A tensor of rank >= 1.")
    .Output(
        0,
        "output",
        "A tensor of rank 1 with the contents of the input tensor");

OPERATOR_SCHEMA(Alias)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
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
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          vector<TensorShape> out(1);
          out.push_back(in[1]);
          out[0].set_data_type(in[0].data_type());
          return out;
        })
    .SetDoc(R"DOC(
Produces tensor condaining data of first input and shape of second input.
)DOC")
    .Input(0, "data", "Tensor whose data will be copied into the output.")
    .Input(1, "shape_tensor", "Tensor whose shape will be applied to output.")
    .Output(0, "output", "Tensor with data of input 0 and shape of input 1.");

OPERATOR_SCHEMA(SumInt)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          vector<TensorShape> out(1);
          out.push_back(in[0]);
          out[0].set_data_type(TensorProto::INT32);
          return out;
        })
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(Sum)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShapeOfInput(0)
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
    .IdenticalTypeAndShapeOfInput(0)
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

OPERATOR_SCHEMA(Max)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1)
    .IdenticalTypeAndShapeOfInput(0)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Element-wise max of each of the input tensors. The first input tensor can be
used in-place as the output tensor, in which case the max will be done in
place and results will be accumulated in input0. All inputs and outputs must
have the same shape and data type.
)DOC")
    .Input(0, "data_0", "First of the input tensors. Can be inplace.")
    .Output(0, "max", "Output tensor. Same dimension as inputs.");

OPERATOR_SCHEMA(MaxGradient).NumInputs(3, INT_MAX).NumOutputs(1, INT_MAX);

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
    .IdenticalTypeAndShape()
    .SetDoc("Copy input tensor into output, potentially across devices.")
    .Input(0, "input", "The input tensor.")
    .Output(0, "output", "Tensor that will contain a copy of the input.");

OPERATOR_SCHEMA(CopyGPUToCPU)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Copy tensor for GPU to CPU context. Must be run under GPU device option.
)DOC")
    .Input(0, "input", "The input tensor.")
    .Output(0, "output", "Tensor that will contain a copy of the input.");

OPERATOR_SCHEMA(CopyCPUToGPU)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Copy tensor for CPU to GPU context. Must be run under GPU device option.
)DOC")
    .Input(0, "input", "The input tensor.")
    .Output(0, "output", "Tensor that will contain a copy of the input.");

OPERATOR_SCHEMA(EnsureCPUOutput)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Take an input tensor in the current Context (GPU or CPU) and create an output
which is always a TensorCPU. This may involves cross-device MemCpy.
)DOC")
    .Input(0, "input", "The input CUDA or CPU tensor.")
    .Output(0, "output", "TensorCPU that is a copy of the input.");

OPERATOR_SCHEMA(CopyFromCPUInput)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Take a CPU input tensor and copy it to an output in the current
Context (GPU or CPU). This may involves cross-device MemCpy.
)DOC")
    .Input(0, "input", "The input CPU tensor.")
    .Output(0, "output", "either a TensorCUDA or a TensorCPU");

OPERATOR_SCHEMA(CopyOnDeviceLike)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc("Copy input tensor into output to the specific device.")
    .Input(0, "input", "The input tensor.")
    .Input(1, "dst", "Tensor, on which device the copy will be performed.")
    .Output(0, "output", "Tensor that will contain a copy of the input.");

OPERATOR_SCHEMA(Shape)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          vector<TensorShape> out(1);
          out[0].add_dims(in[0].dims().size());
          out[0].set_data_type(TensorProto::INT32);
          return out;
        })
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

OPERATOR_SCHEMA(IsEmpty)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc("Returns true iff the input tensor has size == 0")
    .Input(0, "tensor", "Tensor of any type.")
    .Output(0, "is_empty", "Scalar bool tensor. True if input is empty.");

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

OPERATOR_SCHEMA(GatherRanges)
    .NumInputs(2)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Given DATA tensor of rank 1, and RANGES tensor of rank 3, gather
corresponding ranges into a 1-D tensor OUTPUT.

RANGES dimentions description:
1: represents list of examples within a batch
2: represents list features
3: two values which are start and length or a range (to be applied on DATA)

Another output LENGTHS represents each example length within OUTPUT

Example:
  DATA  = [1, 2, 3, 4, 5, 6]
  RANGES = [
    [
      [0, 1],
      [2, 2],
    ],
    [
      [4, 1],
      [5, 1],
    ]
  ]
  OUTPUT = [1, 3, 4, 5, 6]
  LENGTHS = [3, 2]
)DOC")
    .Input(0, "DATA", "Tensor of rank 1.")
    .Input(
        1,
        "RANGES",
        "Tensor of int32/int64 ranges, of dims (N, M, 2). "
        "Where N is number of examples and M is a size of each example. "
        "Last dimention represents a range in the format (start, lengths)")
    .Output(0, "OUTPUT", "1-D tensor of size sum of range lengths")
    .Output(
        1,
        "LENGTHS",
        "1-D tensor of size N with lengths over gathered data"
        " for each row in a batch. sum(LENGTHS) == OUTPUT.size()");

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
    .Output(0, "segment_ids", "1D tensor of length `sum(lengths)`");

OPERATOR_SCHEMA(LengthsToRanges)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given a vector of segment lengths, calculates offsets of each segment and packs
them next to the lengths. For the input vector of length N the output is a Nx2
matrix with (offset, lengths) packaged for each segment.

For example, `[1, 3, 0, 2]` transforms into `[[0, 1], [1, 3], [4, 0], [4, 2]]`.
)DOC")
    .Input(0, "lengths", "1D tensor of int32 segment lengths.")
    .Output(
        0,
        "ranges",
        "2D tensor of shape len(lengths) X 2 and the same type as `lengths`");

OPERATOR_SCHEMA(SegmentIdsToLengths)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Transfers a vector of segment ids to a vector of segment lengths. This operation
supports non-consecutive segment ids. Segments not appearing in the input vector
will have length 0. If the second input is provided, the number of segments =
the size of its first dimension. Otherwise, the number of segments = the last
index in the first input vector + 1.

In general, for consecutive, zero-based segment IDs, this is the inverse
operation of LengthsToSegmentIds, except that a vector of segment IDs
cannot represent empty segments at the end (if the second input is absent).
)DOC")
    .Input(0, "segment_ids", "1-D int32_t or int64_t tensor of segment ids")
    .Input(
        1,
        "data (optional)",
        "if provided, number of segments = the size of its first dimension")
    .Output(0, "lengths", "1-D int64_t tensor of segment lengths");

OPERATOR_SCHEMA(SegmentIdsToRanges)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Transfers a vector of segment ids to a vector of segment ranges. This operation
supports non-consecutive segment ids. Segments not appearing in the input vector
will have length 0. If the second input is provided, the number of segments =
the size of its first dimension. Otherwise, the number of segments = the last
index in the first input vector + 1.
)DOC")
    .Input(0, "segment_ids", "1-D int32_t or int64_t tensor of segment ids")
    .Input(
        1,
        "data (optional)",
        "if provided, number of segments = the size of its first dimension")
    .Output(0, "lengths", "1-D int64_t tensor of segment lengths");

OPERATOR_SCHEMA(LengthsToWeights)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("power", "n of 1/pow(length,n) for normalization")
    .SetDoc(R"DOC(
Similar as LengthsToSegmentIds but output vector of segment
weights derived by lengths. i.e 1/pow(length, power)
)DOC")
    .Input(0, "lengths", "1-D int32_t or int64_t tensor of lengths")
    .Output(0, "a vector of weights", "1-D float tensor of weights by length");

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
Takes a  parameter `dims` with a list of dimension to squeeze.
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

SHOULD_NOT_DO_GRADIENT(WallClockTime);

OPERATOR_SCHEMA(UnsafeCoalesce)
    .NumInputsOutputs([](int inputs, int outputs) {
      return inputs + 1 == outputs;
    })
    .AllowInplace([](int input, int output) { return input == output; })
    .SetDoc(R"DOC(
Coalesce the N inputs into N outputs and a single coalesced output blob.

This allows operations that operate over multiple small kernels (e.g.
biases in a deep CNN) to be coalesced into a single larger operation,
amortizing the kernel launch overhead, synchronization costs for
distributed computation, etc.

The operator:

- computes the total size of the coalesced blob by summing the input sizes
- allocates the coalesced output blob as the total size
- copies the input vectors into the coalesced blob, at the correct offset.
- aliases each Output(i) to- point into the coalesced blob, at the
  corresponding offset for Input(i).

This is 'unsafe' as the output vectors are aliased, so use with
caution.

)DOC");

OPERATOR_SCHEMA(EnsureDense)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
This operator converts dense or sparse gradients to dense ones.
Therefore, sparse gradient can be back propagated to Operators that consume
dense gradients only (e.g., FCGradient).

The operator's behaviors:
- In forward, simply pass in place or copy input to the output.
- In backward, if the gradient passed-in is sparse gradient, change it to
  dense gradient in linear time; otherwise, simply pass the dense gradient.
)DOC")
    .Input(0, "input", "Input tensors.")
    .Output(0, "output", "Output tensor. Same dimension as inputs.");

class GetEnsureDenseGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(
        GradOut(0).IsSparse() || GradOut(0).IsDense(),
        "Input gradient ",
        O(0),
        " should be either sparse or dense.");

    if (GradOut(0).IsDense()) {
      SetDense(0, GO(0));
      return vector<OperatorDef>();
    } else {
      return SingleGradientDef(
          "SparseToDense",
          "",
          vector<string>{GO_I(0), GO_V(0)},
          vector<string>{GI(0)});
    }
  }
};
REGISTER_GRADIENT(EnsureDense, GetEnsureDenseGradient);

SHOULD_NOT_DO_GRADIENT(Print);
SHOULD_NOT_DO_GRADIENT(Shape);
SHOULD_NOT_DO_GRADIENT(HasElements);
SHOULD_NOT_DO_GRADIENT(IsEmpty);
SHOULD_NOT_DO_GRADIENT(LengthsToShape);
SHOULD_NOT_DO_GRADIENT(UnsafeCoalesce);

class GetSqueezeGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ExpandDims", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Squeeze, GetSqueezeGradient);

class GetExpandDimsGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Squeeze", "", vector<string>{GO(0)}, vector<string>{GI(0)});
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

class GetMaxGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    auto gradInputs = vector<string>();
    auto inputs = vector<string>{O(0), GO(0)};
    for (int i = 0; i < def_.input_size(); i++) {
      gradInputs.push_back(GI(i));
      inputs.push_back(I(i));
    }
    return SingleGradientDef("MaxGradient", "", inputs, gradInputs);
  }
};
REGISTER_GRADIENT(Max, GetMaxGradient);

class GetGatherGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    ArgumentHelper argsHelper(def_);
    const bool dense_gradient =
        argsHelper.GetSingleArgument<bool>("dense_gradient", false);

    using Op = GatherOp<CPUContext>;

    if (dense_gradient) {
      return vector<OperatorDef>{CreateOperatorDef(
          "SparseToDense",
          "",
          vector<string>{I(Op::INDICES), GO(0), I(Op::DATA)},
          vector<string>{GI(Op::DATA)})};
    } else {
      // For now we don't do any reshaping as the consumer of this op would
      // probably be ScatterUpdate which is intenionally ignores shapes. We
      // might need to revisit it in the future for correctness purposes. The
      // right shape for the output woild be to flatten INDICES and collapse
      // first X dims of GRAD
      SetSparse(Op::DATA, I(Op::INDICES), GO(0));
      return vector<OperatorDef>();
    }
  }
};
REGISTER_GRADIENT(Gather, GetGatherGradient);

struct GetFlattenToVecGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "ResizeLike", "", vector<string>{GO(0), I(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(FlattenToVec, GetFlattenToVecGradient);

struct GetCopyGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CopyOnDeviceLike",
        "",
        vector<string>{GO(0), I(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(Copy, GetCopyGradient);

struct GetGPUToCPUGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CopyCPUToGPU", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(CopyGPUToCPU, GetGPUToCPUGradient);

struct GetCPUToGPUGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "CopyGPUToCPU", "", vector<string>{GO(0)}, vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(CopyCPUToGPU, GetCPUToGPUGradient);

SHOULD_NOT_DO_GRADIENT(Unique);
SHOULD_NOT_DO_GRADIENT(LengthsToSegmentIds);
SHOULD_NOT_DO_GRADIENT(SegmentIdsToLengths);
SHOULD_NOT_DO_GRADIENT(SegmentIdsToRanges);
SHOULD_NOT_DO_GRADIENT(SegmentIdsToLengthWeights);
// TODO(azzolini): Add support for slice gradient
SHOULD_NOT_DO_GRADIENT(Slice);
SHOULD_NOT_DO_GRADIENT(GatherRangesOp);

} // namespace

template <typename T, class Context>
bool MaxOp<T, Context>::Compute() {
  auto& input0 = Input(0);
  const int N = input0.size();
  T* output_data = Output(0)->template mutable_data<T>();

  for (int i = 1; i < InputSize(); i++) {
    auto input_data = Input(i).template data<T>();
    EigenVectorMap<T> output_vec(output_data, N);
    output_vec = output_vec.cwiseMax(ConstEigenVectorMap<T>(input_data, N));
  }

  return true;
}

template <typename T, class Context>
bool MaxGradientOp<T, Context>::RunOnDevice() {
  auto& output = Input(0);
  auto& grad_output = Input(1);
  const int kInputStartOffset = 2;

  const T* data = output.template data<T>();
  ConstEigenArrayMap<T> output_array(
      output.template data<T>(), 1, output.size());
  ConstEigenArrayMap<T> grad_out_array(
      grad_output.template data<T>(), 1, grad_output.size());

  for (int i = 0; i < OutputSize(); i++) {
    auto& input = Input(i + kInputStartOffset);
    ConstEigenArrayMap<T> input_array(
        input.template data<T>(), 1, input.size());

    auto* grad_input = Output(i);
    grad_input->ResizeLike(input);
    EigenArrayMap<T> grad_in_array(
        grad_input->template mutable_data<T>(), 1, grad_input->size());
    grad_in_array = grad_out_array *
        input_array.cwiseEqual(output_array).template cast<T>();
  }
  return true;
}

} // namespace caffe2
