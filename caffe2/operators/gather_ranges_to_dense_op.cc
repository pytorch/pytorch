#include "caffe2/operators/gather_ranges_to_dense_op.h"

namespace caffe2 {
namespace {

OPERATOR_SCHEMA(GatherRangesToDense)
    .NumInputs(2, 3)
    .NumOutputs(1, INT_MAX)
    .SetDoc(R"DOC(
Given DATA tensor of rank 1, and RANGES tensor of rank 3, gather values
corresponding to each range into a separate output tensor. If the optional input
KEY tensor is also given, the output will be sorted by KEY for each example.

RANGES dimensions description:
1: represents list of examples within a batch
2: represents list features
3: two values which are start and length or a range (to be applied on DATA)

Each feature has fixed lengths which are passed as lengths argument and a
separate tensor will be produced for each feature.
i.e. DATA.dim(1) = len(lengths) = NumOuptuts.

Missing features (represented by empty ranges) filled with default_value.

Example 1:
  DATA  = [1, 2, 3, 4, 5, 6, 7, 8]
  RANGES = [
    [
      [2, 4],
      [0, 2],
    ],
    [
      [0, 0],
      [6, 2],
    ]
  ]
  lengths = [4, 2]
  OUTPUT[0] = [[3, 4, 5, 6], [0, 0, 0, 0]]
  OUTPUT[1] = [[1, 2], [7, 8]]

Example 2 (with KEY):
DATA  = [1, 2, 3, 4, 5, 6, 7, 8]
KEY   = [0, 1, 3, 2, 1, 0, 1, 0]
RANGES = [
  [
    [2, 4],
    [0, 2],
  ],
  [
    [0, 0],
    [6, 2],
  ]
]
lengths = [4, 2]
OUTPUT[0] = [[6, 5, 4, 3], [0, 0, 0, 0]]
OUTPUT[1] = [[1, 2], [8, 7]]

Contrast Example 2 with Example 1. For each data point per feature, the values
are sorted by the corresponding KEY.
)DOC")
    .Input(0, "DATA", "Tensor of rank 1.")
    .Input(
        1,
        "RANGES",
        "Tensor of int32/int64 ranges, of dims (N, M, 2). "
        "Where N is number of examples and M is a size of each example. "
        "Last dimension represents a range in the format (start, lengths)")
    .Input(2, "KEY", "Tensor of rank 1 and type int64.")
    .Output(0, "OUTPUT", "1-D tensor of size sum of range lengths")
    .Arg("lengths", "Expected lengths for ranges")
    .Arg(
        "min_observation",
        "The number of observations needed before deciding that the ratio of "
        "mismatched ranges is alarming, also determines whether an info "
        "sumarizing the empty and mismatch ratio will be printed at the end.")
    .Arg(
        "max_mismatched_ratio",
        "An error is raised when ratio of mismatched ranges exceeds this.")
    .Arg(
        "max_empty_ratio",
        "An error is raised when ratio of empty ranges exceeds this (default is"
        " 1, which means by default no error will be triggered).")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto lengths = helper.GetRepeatedArgument<int>("lengths");
      CAFFE_ENFORCE_EQ(in[0].dims_size(), 1, "DATA should be 1-D tensor.");
      CAFFE_ENFORCE_EQ(in[1].dims_size(), 3, "RANGES should be 3-D tensor.");
      if (in.size() > 2) {
        CAFFE_ENFORCE_EQ(in[2].dims_size(), 1, "KEY should be 1-D tensor.");
      }
      CAFFE_ENFORCE_GT(lengths.size(), 0, "lengths should be non-empty.");
      std::vector<TensorShape> out(lengths.size());
      for (int i = 0; i < lengths.size(); ++i) {
        out[i].set_data_type(in[0].data_type());
        out[i].add_dims(in[1].dims(0));
        out[i].add_dims(lengths[i]);
      }
      return out;
    });

REGISTER_CPU_OPERATOR(GatherRangesToDense, GatherRangesToDenseOp<CPUContext>);
NO_GRADIENT(GatherRangesToDense);

} // namespace
} // namespace caffe2

using GatherRangesToDenseCPUOp =
    caffe2::GatherRangesToDenseOp<caffe2::CPUContext>;

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    GatherRangesToDense,
    "_caffe2::GatherRangesToDense(Tensor data, Tensor ranges, Tensor? key, int[] lengths, int min_observation, float max_mismatched_ratio, float max_empty_ratio) -> Tensor[] outputs",
    GatherRangesToDenseCPUOp);
