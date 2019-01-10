#ifndef CAFFE2_OPERATORS_RESHAPE_OP_H_
#define CAFFE2_OPERATORS_RESHAPE_OP_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// Takes a shape and data tensor and reshapes it
template <typename F, class Context>
class ReshapeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ReshapeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        new_shape_(OperatorBase::GetRepeatedArgument<int64_t>("shape")) {}

  bool RunOnDevice() override {
    if (InputSize() == 2) {
      return DispatchHelper<TensorTypes<int, int64_t>>::call(this, Input(1));
    }
    CAFFE_ENFORCE(
        OperatorBase::HasArgument("shape"), "Argument `shape` is missing.");
    return this->template DoRunWithType<int64_t>();
  }

  template <typename T>
  bool DoRunWithType() {
    DoRunWithTypeImpl<T>(Input(0), Output(0));
    return true;
  }

 protected:
  template <typename T>
  void DoRunWithTypeImpl(
      const Tensor<Context>& input,
      Tensor<Context>* output) {
    vector<int64_t> actual_new_shape = new_shape_;
    if (InputSize() == 2) {
      CAFFE_ENFORCE(
          !OperatorBase::HasArgument("shape"),
          "New shape is specified by the input blob, do not pass in "
          "the argument `shape`.");

      auto& shape = Input(1);
      CAFFE_ENFORCE(shape.ndim() == 1, "Shape should be 1-D");

      const T* shape_data = shape.template data<T>();

      // Bit awkward, but needed so works on both CPU and CUDA contexts
      std::vector<T> tmpv(shape.size());
      context_.template CopyBytes<Context, CPUContext>(
          shape.size() * sizeof(T), shape_data, &tmpv[0]);
      actual_new_shape.assign(tmpv.begin(), tmpv.begin() + shape.size());
    }

    // Copy over the dimensions for those that are specified zero.
    for (int i = 0; i < actual_new_shape.size(); ++i) {
      if (actual_new_shape[i] == 0) {
        actual_new_shape[i] = input.dim(i);
      }
    }

    // Checks if the new shape is valid and fills in the missing dimension
    // specified by -1.
    // NOTE: At most one dimension can be -1.
    auto total_size = input.size_from_dim(0);
    T size = 1;
    int unknown_idx = -1;
    for (int i = 0; i < actual_new_shape.size(); ++i) {
      const auto dim = actual_new_shape[i];
      if (dim == -1) {
        CAFFE_ENFORCE(
            unknown_idx == -1,
            "Argument `shape` has more than one missing dimension.");
        unknown_idx = i;
      } else {
        size *= dim;
      }
    }

    if (unknown_idx != -1) {
      CAFFE_ENFORCE(
          total_size % size == 0,
          "Argument `shape` does not agree with the input data.",
          " (",
          total_size,
          " vs ",
          size,
          ")");
      actual_new_shape[unknown_idx] = total_size / size;
    } else {
      CAFFE_ENFORCE_EQ(
          total_size,
          size,
          "Argument `shape` does not agree with the input data.",
          " (",
          total_size,
          " != ",
          size,
          ")");
    }

    // Write the original shape to the second output.
    auto* old_shape = Output(1);
    old_shape->Resize(input.ndim());
    T* old_shape_data = old_shape->template mutable_data<T>();
    for (int i = 0; i < input.ndim(); ++i) {
      math::Set<T, Context>(1, input.dim(i), old_shape_data + i, &context_);
    }

    output->Resize(actual_new_shape);
    if (output != &input) {
      // If we are not doing in-place computation, a copy is needed.
      context_.template CopyItems<Context, Context>(
          input.meta(),
          input.size(),
          input.raw_data(),
          output->raw_mutable_data(input.meta()));
    }
  }

 private:
  vector<int64_t> new_shape_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RESHAPE_OP_H_
