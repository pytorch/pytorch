#ifndef CAFFE2_OPERATORS_MEAN_OPS_H_
#define CAFFE2_OPERATORS_MEAN_OPS_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/operators/elementwise_ops_utils.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

template <class Context>
class MeanOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(MeanOp)

  template <typename T>
  bool DoRunWithType() {
    std::vector<std::vector<int>> input_dims(InputSize());
    for (int i = 0; i < InputSize(); i++) {
      std::copy(
          Input(i).sizes().cbegin(),
          Input(i).sizes().cend(),
          std::back_inserter(input_dims[i]));
    }
    auto output_dims = ComputeBroadcastDims(input_dims);
    auto* output = Output(0, output_dims, at::dtype<T>());
    auto* output_data = output->template mutable_data<T>();
    if (IsInputOutputAlias(0, 0)) {
      CAFFE_ENFORCE(
          (output_dims.size() == input_dims[0].size()) &&
          std::equal(
              output_dims.begin(),
              output_dims.begin() + output_dims.size(),
              input_dims[0].begin()),
          "Cannot broadcast output to the first input.");
    }
    std::vector<int> output_dims_int;
    std::copy(
        output_dims.begin(),
        output_dims.end(),
        std::back_inserter(output_dims_int));
    math::Broadcast(
        input_dims[0].size(),
        input_dims[0].data(),
        output_dims_int.size(),
        output_dims_int.data(),
        T(1),
        Input(0).template data<T>(),
        output_data,
        &context_);
    for (int i = 1; i < InputSize(); i++) {
      math::Add(
          input_dims[i].size(),
          input_dims[i].data(),
          output_dims_int.size(),
          output_dims_int.data(),
          Input(i).template data<T>(),
          output_data,
          output_data,
          &context_);
    }
    math::Scale(
        output->numel(),
        1.0f / InputSize(),
        output_data,
        output_data,
        &context_);

    return true;
  }


 private:
  std::vector<int64_t> ComputeBroadcastDims(
      const std::vector<std::vector<int>>& input_dims) const {
    const int ninp = input_dims.size();
    std::vector<int> input_sizes(ninp);
    for (int i = 0; i < ninp; i++) {
      input_sizes[i] = input_dims[i].size() - 1;
    }
    const int ndim = *std::max_element(input_sizes.begin(), input_sizes.end());
    std::vector<int64_t> output_dims(ndim + 1);
    for (int k = ndim; k >= 0; k--) {
      int max_dim = -1;
      for (int i = 0; i < ninp; i++) {
        int isz = input_sizes[i];
        if (isz >= 0) {
          CAFFE_ENFORCE(
              max_dim == -1 || max_dim == 1 || input_dims[i][isz] == max_dim ||
              input_dims[i][isz] == 1);
          if (input_dims[i][isz] == 0) {
            max_dim = 0;
          } else {
            max_dim = std::max(max_dim, input_dims[i][isz]);
          }
        }
        input_sizes[i]--;
      }
      output_dims[k] = max_dim;
    }
    return output_dims;
  }

  bool RunOnDevice() override {
    if (Input(0).template IsType<float>()) {
      return DoRunWithType<float>();
    } else {
      CAFFE_THROW(
          "Mean operator only supports 32-bit float, but",
          " input was of type ",
          Input(0).dtype().name());
    }
  }
};

template <class Context>
class MeanGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  MeanGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  template <typename T>
  bool DoRunWithType() {
    auto& dY = Input(0);
    const auto* dY_data = dY.template data<T>();
    int num_inputs = OutputSize();
    const T scale = T(1) / static_cast<T>(num_inputs);

    if (InputSize() == 1) {
      // Handling legacy case for backwards compatibility
      auto* dX0 = Output(0, dY.sizes(), at::dtype<T>());
      int size = dY.numel();
      math::Scale(
          size, scale, dY_data, dX0->template mutable_data<T>(), &context_);

      // Copy the rest dX
      for (int i = 1; i < num_inputs; i++) {
        auto* cur_dX = Output(i);
        cur_dX->ResizeLike(dY);
        cur_dX->CopyFrom(*dX0, true /*async*/);
      }
      return true;
    }

    const std::vector<int> output_dims(dY.sizes().cbegin(), dY.sizes().cend());
    for (int i = num_inputs; i >= 1; i--) {
      std::vector<int> input_dims;
      const auto& original_input_size = Input(i).sizes();
      std::copy(
          original_input_size.cbegin(),
          original_input_size.cend(),
          std::back_inserter(input_dims));
      std::vector<int> input_axes;
      std::vector<int> output_axes;
      elementwise_ops_utils::ComputeBinaryBroadcastBackwardAxes(
          input_dims, output_dims, &input_axes, &output_axes);
      auto* dX = Output(i - 1, original_input_size, at::dtype<T>());
      auto* dX_data = dX->template mutable_data<T>();
      math::ReduceSum<T, Context>(
          output_dims.size(),
          output_dims.data(),
          input_axes.size(),
          input_axes.data(),
          scale,
          dY_data,
          dX_data,
          &context_);
    }
    return true;
  }

  bool RunOnDevice() override {
    if (Input(0).template IsType<float>()) {
      return DoRunWithType<float>();
    } else {
      CAFFE_THROW(
          "Mean operator only supports 32-bit float, but",
          " input was of type ",
          Input(0).dtype().name());
    }
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MEAN_OPS_H_
