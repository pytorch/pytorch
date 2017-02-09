#ifndef CAFFE2_OPERATORS_SPARSE_TO_DENSE_OP_H_
#define CAFFE2_OPERATORS_SPARSE_TO_DENSE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class SparseToDenseOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_DISPATCH_HELPER;

  SparseToDenseOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        output_first_dim_(
            OperatorBase::GetSingleArgument<int>("output_first_dim", 0)) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename TInd>
  int GetOutputFirstDim(
      const TInd* sparse_indices_vec,
      const int32_t sparse_indices_len) {
    if (output_first_dim_ > 0) {
      CAFFE_ENFORCE_EQ(InputSize(), 2);
      return output_first_dim_;
    }
    if (InputSize() == 3) {
      auto& data_to_infer_dim = Input(DATA_TO_INFER_DIM);
      CAFFE_ENFORCE_GE(data_to_infer_dim.ndim(), 1);
      return data_to_infer_dim.dim32(0);
    }
    if (sparse_indices_len <= 0) {
      return 0;
    }
    return 1 +
        *std::max_element(
               sparse_indices_vec, sparse_indices_vec + sparse_indices_len);
  }

  template <typename TInd>
  bool DoRunWithType() {
    auto& sparse_indices = Input(INDICES);
    CAFFE_ENFORCE_EQ(sparse_indices.ndim(), 1);
    auto& sparse_values = Input(VALUES);
    CAFFE_ENFORCE_GE(sparse_values.ndim(), 1);
    CAFFE_ENFORCE_EQ(sparse_indices.size(), sparse_values.dim(0));

    const TInd* sparse_indices_vec = sparse_indices.template data<TInd>();
    const int32_t sparse_indices_len = sparse_indices.dim32(0);
    const int output_first_dim =
        GetOutputFirstDim(sparse_indices_vec, sparse_indices_len);

    auto shape = sparse_values.dims();
    shape[0] = output_first_dim;
    auto* output = Output(0);
    output->Resize(shape);

    char* output_data =
        static_cast<char*>(output->raw_mutable_data(sparse_values.meta()));
    if (sparse_values.meta().copy() == nullptr) {
      // If it is not nullptr, the tensor is already initialized by contructor.
      math::Set(output->nbytes(), '\0', output_data, &context_);
    }
    const int block_nitems = sparse_values.size_from_dim(1);
    const int block_nbytes = block_nitems * sparse_values.itemsize();
    const char* sparse_values_vec =
        static_cast<const char*>(sparse_values.raw_data());

    for (int32_t i = 0; i < sparse_indices_len; i++) {
      const TInd idx = sparse_indices_vec[i];
      CAFFE_ENFORCE_GE(idx, 0);
      CAFFE_ENFORCE_LT(idx, output_first_dim);
      context_.template CopyItems<Context, Context>(
          sparse_values.meta(),
          block_nitems,
          sparse_values_vec + i * block_nbytes,
          output_data + idx * block_nbytes);
    }

    return true;
  }

 private:
  int output_first_dim_;

  INPUT_TAGS(INDICES, VALUES, DATA_TO_INFER_DIM);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SPARSE_TO_DENSE_OP_H_
