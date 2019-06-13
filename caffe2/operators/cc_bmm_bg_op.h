#ifndef CAFFE2_FB_OPERATORS_CC_BMM_BG_H_
#define CAFFE2_FB_OPERATORS_CC_BMM_BG_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

using T = float;
using TInd = int;
using Engine = DefaultEngine;

template <class Context>
class ConcatBatchMatMulBatchGatherOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  ConcatBatchMatMulBatchGatherOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override;

 protected:
  int axis_ = 1;
  int add_axis_ = 1;

  bool trans_a_ = 0;
  bool trans_b_ = 1;
  bool broadcast_ = 0;
};

template <class Context>
bool ConcatBatchMatMulBatchGatherOp<Context>::RunOnDevice() {
  auto& indices = Input(0);
  auto& input_zero = Input(1);
  int adj_size = input_zero.dim() + 1;
  int canonical_axis = 1;
  CAFFE_ENFORCE_LT(canonical_axis, adj_size, "Axis not in input ndim range.");
  for (int i = 2; i < InputSize(); ++i) {
    CAFFE_ENFORCE(
        Input(i).dtype() == input_zero.dtype(),
        "All inputs must have the same type, expected: ",
        input_zero.dtype().name(),
        " but got: ",
        Input(i).dtype().name(),
        " for input: ",
        i);
  }

  int before = 1, after = 1;
  for (int i = 0; i < input_zero.dim(); ++i) {
    int dim = input_zero.dim32(i);
    if (i < canonical_axis) {
      before *= dim;
    } else { // i > canonical_axis || i == canonical_axis && add_axis_
      after *= dim;
    }
    // check the input dims are compatible.
    for (int j = 2; j < InputSize(); ++j) {
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
          input_zero.sizes(),
          "> vs <",
          Input(j).sizes(),
          ">.");
    }
  }

  auto ndata = InputSize() - 1;
  auto batch_size = before;
  auto embed_size = after;
  auto gather_size = indices.sizes()[0];

  vector<int64_t> output_dims;
  output_dims.push_back(batch_size);
  output_dims.insert(
      output_dims.begin() + 1, indices.sizes().begin(), indices.sizes().end());
  auto* output = Output(0, output_dims, at::dtype<T>());
  // std::stringstream ss;
  // ss << "[";
  // for(int i = 0; i < output_dims.size(); i++) ss << output_dims[i];
  // ss << "]";
  // LOG(INFO) << "output size: " << ss.str();

  auto* output_data = output->template mutable_data<T>();
  auto* indices_data = indices.template data<TInd>();
#pragma omp parallel
  {
    std::vector<T> scratch_input(ndata * embed_size);
    std::vector<T> scratch_output(ndata * ndata);

#pragma omp for
    for (int b = 0; b < batch_size; ++b) {
      // concat input to scratch
      for (int i = 1; i < InputSize(); ++i) {
        auto* input_data = Input(i).template data<T>();
        memcpy(
            &scratch_input[(i - 1) * embed_size],
            input_data + b * embed_size,
            embed_size * Input(i).itemsize());
      }
      // call mkl gemm
      math::Gemm<T, Context, Engine>(
          CblasNoTrans,
          CblasTrans,
          ndata,
          ndata,
          embed_size,
          1,
          &scratch_input[0],
          &scratch_input[0],
          0,
          &scratch_output[0],
          &context_);
      // do gather

      int64_t output_offset = b * gather_size;
      for (int i = 0; i < gather_size; i++) {
        output_data[output_offset + i] = scratch_output[indices_data[i]];
      }
    }
  }
  return true;
}

} // namespace caffe2

#endif // CAFFE2_FB_OPERATORS_CC_BMM_BG_H_
