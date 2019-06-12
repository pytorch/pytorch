#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class GatherFused8BitRowwiseOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(GatherFused8BitRowwiseOp);

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, this->template Input<Tensor>(INDICES, CPU));
  }

  template <typename Index>
  bool DoRunWithType() {
    const auto& data = Input(DATA);
    const auto& indices = Input(INDICES);
    auto* output = Output(0);

    CAFFE_ENFORCE_EQ(data.dim(), 2, "DATA must be a matrix");
    CAFFE_ENFORCE_EQ(indices.dim(), 1, "INDICES must be a vector");
    CAFFE_ENFORCE_GT(data.size(1), 8, "DATA must have more than 8 columns");
    // Subtract 8 from the #columns of data for the 4 bytes for scale and 4
    // bytes for bias that we use in the fused representation (per row).
    const std::vector<int64_t> shape = {indices.size(0), data.size(1) - 8};
    output->Resize(shape);

    int block_size = shape[1];
    auto block_bytesize = data.size_from_dim(1) * data.dtype().itemsize();
    int N = indices.numel();

    const uint8_t* src_base = data.template data<uint8_t>();
    const Index* idxs = indices.template data<Index>();
    auto out = output->template mutable_data<float>();

    for (int i = 0; i < N; ++i) {
      auto idx = idxs[i];
      CAFFE_ENFORCE(
          0 <= idx && idx < data.size(0),
          "INDICES element is out of DATA bounds, id=",
          idx,
          " data_dim=",
          data.size(0));
      const uint8_t* src = src_base + idx * block_bytesize;
      ConstEigenVectorArrayMap<uint8_t> input_row_values(src, shape[1]);
      ConstEigenVectorArrayMap<float> input_row_scale_bias(
          reinterpret_cast<const float*>(src + shape[1]), 2);

      EigenVectorArrayMap<float> output_row(out + i * shape[1], shape[1]);

      output_row = input_row_values.cast<float>() * input_row_scale_bias(0) +
          input_row_scale_bias(1);
    }
    return true;
  }

  INPUT_TAGS(DATA, INDICES);
};

} // namespace caffe2
