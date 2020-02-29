#include "batch_bucketize_op.h"

#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <>
bool BatchBucketizeOp<CPUContext>::RunOnDevice() {
  auto& feature = Input(FEATURE);
  auto& indices = Input(INDICES);
  auto& boundaries = Input(BOUNDARIES);
  auto& lengths = Input(LENGTHS);

  CAFFE_ENFORCE_EQ(lengths.dim(), 1);
  CAFFE_ENFORCE_EQ(indices.dim(), 1);
  CAFFE_ENFORCE_EQ(boundaries.dim(), 1);
  CAFFE_ENFORCE_EQ(feature.dim(), 2);
  CAFFE_ENFORCE_EQ(lengths.numel(), indices.numel());

  const auto* lengths_data = lengths.template data<int32_t>();
  const auto* indices_data = indices.template data<int32_t>();
  const auto* boundaries_data = boundaries.template data<float>();
  const auto* feature_data = feature.template data<float>();
  auto batch_size = feature.size(0);
  auto feature_dim = feature.size(1);
  auto output_dim = indices.numel();

  int64_t length_sum = 0;
  for (int64_t i = 0; i < lengths.numel(); i++) {
    CAFFE_ENFORCE_GE(feature_dim, indices_data[i]);
    length_sum += lengths_data[i];
  }
  CAFFE_ENFORCE_EQ(length_sum, boundaries.numel());

  int64_t lower_bound = 0;
  auto* output = Output(O, {batch_size, output_dim}, at::dtype<int32_t>());
  auto* output_data = output->template mutable_data<int32_t>();

  for (int64_t i = 0; i < batch_size; i++) {
    lower_bound = 0;
    for (int64_t j = 0; j < output_dim; j++) {
      for (int64_t k = 0; k <= lengths_data[j]; k++) {
        if (k == lengths_data[j] ||
            feature_data[i * feature_dim + indices_data[j]] <=
                boundaries_data[lower_bound + k]) {
          output_data[i * output_dim + j] = k;
          break;
        } else {
          continue;
        }
      }
      lower_bound += lengths_data[j];
    }
  }
  return true;
}

REGISTER_CPU_OPERATOR(BatchBucketize, BatchBucketizeOp<CPUContext>);

OPERATOR_SCHEMA(BatchBucketize)
    .NumInputs(4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Bucketize the float_features into sparse features.
The float_features is a N * D tensor where N is the batch_size, and D is the feature_dim.
The indices is a 1D tensor containing the indices of the features that need to be bucketized.
The lengths is a 1D tensor that splits the following 'boundaries' argument.
The boundaries is a 1D tensor containing the border list for each feature.

With in each batch, `indices` should not have duplicate number,
and the number of elements in `indices` should be less than or equal to `D`.
Each element in `lengths` vector (lengths[`i`]) represents
the number of boundaries in the sub border list.
The sum of all elements in `lengths` must be equal to the size of  `boundaries`.
If lengths[0] = 2, the first sub border list is [0.5, 1.0], which separate the
value to (-inf, 0.5], (0,5, 1.0], (1.0, inf). The bucketized feature will have
three possible values (i.e. 0, 1, 2).


For example, with input:

  float_features = [[1.42, 2.07, 3.19, 0.55, 4.32],
                    [4.57, 2.30, 0.84, 4.48, 3.09],
                    [0.89, 0.26, 2.41, 0.47, 1.05],
                    [0.03, 2.97, 2.43, 4.36, 3.11],
                    [2.74, 5.77, 0.90, 2.63, 0.38]]
  indices = [0, 1, 4]
  lengths = [2, 3, 1]
  boundaries =  [0.5, 1.0, 1.5, 2.5, 3.5, 2.5]

The output is:

  output =[[2, 1, 1],
           [2, 1, 1],
           [1, 0, 0],
           [0, 2, 1],
           [2, 3, 0]]

after running this operator.
)DOC")
    .Input(
        0,
        "float_features",
        "2-D dense tensor, the second dimension must be greater or equal to the indices dimension")
    .Input(
        1,
        "indices",
        "Flatten tensor, containing the indices of `float_features` to be bucketized. The datatype must be int32.")
    .Input(
        2,
        "lengths",
        "Flatten tensor, the size must be equal to that of `indices`. The datatype must be int32.")
    .Input(
        3,
        "boundaries",
        "Flatten tensor, dimension has to match the sum of lengths")
    .Output(
        0,
        "bucktized_feat",
        "2-D dense tensor, with 1st dim = float_features.dim(0), 2nd dim = size(indices)"
        "in the arg list, the tensor is of the same data type as `feature`.");

NO_GRADIENT(BatchBucketize);

} // namespace caffe2
