#pragma once

#include <immintrin.h>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class ConcatAddMulReplaceNaNClipOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ConcatAddMulReplaceNaNClipOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    if (HasArgument("clip_min")) {
      min_ = static_cast<float>(this->template GetSingleArgument<float>(
          "clip_min", std::numeric_limits<float>::lowest()));
    }
    if (HasArgument("clip_max")) {
      max_ = static_cast<float>(this->template GetSingleArgument<float>(
          "clip_max", std::numeric_limits<float>::max()));
    }
  }

  bool RunOnDevice() {
    auto concat_input_start = 2;
    auto axis_ = 1;

    Tensor* split = Output(
        1,
        vector<int64_t>(1, InputSize() - concat_input_start),
        at::dtype<int>());
    int* axis_data = split->template mutable_data<int>();

    auto& add_input = Input(0);
    auto& mul_input = Input(1);

    auto& concat_input_0 = Input(2);

    int adj_size = concat_input_0.dim();
    int canonical_axis = canonical_axis_index_(axis_, adj_size);
    CAFFE_ENFORCE_LT(canonical_axis, adj_size, "Axis not in input ndim range.");
    for (int i = concat_input_start + 1; i < InputSize(); ++i) {
      CAFFE_ENFORCE(
          Input(i).dtype() == concat_input_0.dtype(),
          "All inputs must have the same type, expected: ",
          concat_input_0.dtype().name(),
          " but got: ",
          Input(i).dtype().name(),
          " for input: ",
          i);
    }
    int before = 1, after = 1;
    vector<int64_t> output_dims(concat_input_0.sizes().vec());
    for (int i = 0; i < concat_input_0.dim(); ++i) {
      if (i == canonical_axis) {
        continue;
      }
      int dim = concat_input_0.dim32(i);
      if (i < canonical_axis) {
        before *= dim;
      } else { // i > canonical_axis
        after *= dim;
      }
      // check the input dims are compatible.
      for (int j = concat_input_start; j < InputSize(); ++j) {
        int dim_j = Input(j).dim32(i);
        CAFFE_ENFORCE(
            dim == dim_j,
            "Expect dimension = ",
            dim,
            " got ",
            dim_j,
            " at axis = ",
            i,
            " for input: ",
            j,
            ". The input tensors can only have different dimensions "
            "when arg 'add_axis' = 0 and along the axis = ",
            canonical_axis,
            " <",
            Input(0).sizes(),
            "> vs <",
            Input(j).sizes(),
            ">.");
      }
    }

    CAFFE_ENFORCE(
        concat_input_0.dim() <= 2,
        "Cannot handle fused concat with dim > 2, please update your fusion logic");

    int output_channels = 0;
    for (int i = concat_input_start; i < InputSize(); ++i) {
      axis_data[i - concat_input_start] = Input(i).dim32(canonical_axis);
      output_channels += Input(i).dim32(canonical_axis);
    }
    output_dims[canonical_axis] = output_channels;
    auto* output = Output(0, output_dims, at::dtype<float>());

    size_t output_offset = 0;
    for (int i = concat_input_start; i < InputSize(); ++i) {
      auto& input = Input(i);
      auto axis_dim = input.dim32(canonical_axis);
      math::CopyMatrix<Context>(
          input.itemsize(),
          before,
          axis_dim * after,
          input.raw_data(),
          axis_dim * after,
          static_cast<char*>(output->raw_mutable_data(concat_input_0.dtype())) +
              output_offset,
          output_channels * after,
          &context_,
          concat_input_0.dtype().copy());
      output_offset += axis_dim * after * input.itemsize();
    }

    float* output_data = output->template mutable_data<float>();
    const float* add_input_data = add_input.template data<float>();
    const float* mul_input_data = mul_input.template data<float>();

    const auto _max_mask = _mm256_set1_ps(max_);
    const auto _min_mask = _mm256_set1_ps(min_);
    const auto _zeros = _mm256_set1_ps(0.f);

    output_offset = 0;
    for (auto outer = 0; outer < before; ++outer) {
      auto axis_dim = output->dim32(canonical_axis);
      size_t inner_size = axis_dim * after;
      auto inner = 0;
      for (; inner < inner_size; inner += 8) {
        if (inner + 7 >= inner_size) {
          break;
        }
        auto elem = _mm256_loadu_ps(&(output_data[output_offset + inner]));
        auto add_elem = _mm256_loadu_ps(&(add_input_data[inner]));
        auto mul_elem = _mm256_loadu_ps(&(mul_input_data[inner]));
        auto added = _mm256_add_ps(elem, add_elem);
        auto mulled = _mm256_mul_ps(added, mul_elem);
        // ordered non-signaling compare returns false on NaN
        auto mask = _mm256_cmp_ps(mulled, mulled, _CMP_EQ_OQ);
        auto removed_nan = _mm256_blendv_ps(_zeros, mulled, mask);
        auto out_val =
            _mm256_max_ps(_mm256_min_ps(_max_mask, removed_nan), _min_mask);
        _mm256_storeu_ps(&output_data[output_offset + inner], out_val);
      }

#if defined(_OPENMP)
#pragma omp simd
#endif
      for (auto inner_omp = inner; inner_omp < inner_size; ++inner_omp) {
        float elem = output_data[output_offset + inner_omp];
        float add_elem = add_input_data[inner_omp];
        float mul_elem = mul_input_data[inner_omp];
        float clipped = (elem + add_elem) * mul_elem;
        if (std::isnan(clipped)) {
          clipped = 0;
        }
        if (clipped > max_) {
          clipped = max_;
        } else if (clipped < min_) {
          clipped = min_;
        }
        output->template mutable_data<float>()[output_offset + inner_omp] = clipped;
      }
      output_offset += axis_dim * after;
    }
    return true;
  }

 protected:
  float min_;
  float max_;
};

} // namespace caffe2
