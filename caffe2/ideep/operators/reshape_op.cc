#include <caffe2/ideep/ideep_utils.h>

using namespace caffe2;

namespace {

// Takes a shape and data tensor and reshapes it
class IDEEPReshapeOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPReshapeOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        new_shape_(OperatorBase::GetRepeatedArgument<itensor::dim>("shape")) {}

  bool RunOnDevice() override {
    ideep::tensor::dims actual_new_shape = new_shape_;
    if (InputSize() == 2) {
      CAFFE_ENFORCE(
          !OperatorBase::HasArgument("shape"),
          "New shape is specified by the input blob, do not pass in "
          "the argument `shape`.");

      // shape info live on CPU
      auto& shape = OperatorBase::Input<TensorCPU>(1, CPU);
      CAFFE_ENFORCE(shape.ndim() == 1, "Shape should be 1-D");
      actual_new_shape.reserve(shape.size());
      if (shape.template IsType<int>()) {
        const int* shape_data = shape.template data<int>();
        actual_new_shape.assign(shape_data, shape_data + shape.size());
      } else if (shape.template IsType<int64_t>()) {
        const int64_t* shape_data = shape.template data<int64_t>();
        for (int i = 0; i < shape.size(); ++i) {
          actual_new_shape.push_back(static_cast<int>(shape_data[i]));
        }
      } else {
        CAFFE_THROW(
            "IDEEP reshape only supports shape data in int32_t or int64_t");
      }
    } else {
      CAFFE_ENFORCE(
          OperatorBase::HasArgument("shape"), "Argument `shape` is missing.");
    }

    auto& input = Input(0);
    // Copy over the dimensions for those that are specified zero.
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
    for (int i = 0; i < actual_new_shape.size() && i < input.ndims(); ++i) {
      if (actual_new_shape[i] == 0) {
        actual_new_shape[i] = input.get_dim(i);
      }
    }

    // Checks if the new shape is valid and fills in the missing dimension
    // specified by -1.
    // NOTE: At most one dimension can be -1.
    auto total_size = input.get_nelems();
    int size = 1;
    int unknown_idx = -1;
    // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
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
    if (size == 0 && total_size != 0) {
      CAFFE_THROW(
          "Can not reshape a non-zero size (",
          total_size,
          ") tensor to zero size.");
    }

    if (unknown_idx != -1) {
      CAFFE_ENFORCE_NE(
          size,
          0,
          "New shape at dim ",
          unknown_idx,
          " can not be inferred since new size is zero.");
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
    // shape info live on CPU
    TensorCPU* old_shape = OperatorBase::Output<TensorCPU>(1, CPU);
    old_shape->Resize(input.ndims());
    int* old_shape_data = old_shape->template mutable_data<int>();
    for (int i = 0; i < input.ndims(); ++i) {
      old_shape_data[i] = input.get_dim(i);
    }

    auto* output = Output(0);
    if (output != &input) {
      // If we are not doing in-place computation, a copy is needed.
      output->reinit_like(input);
      ideep::direct_copy::compute(input, *output);
    }

    output->reshape(actual_new_shape);
    return true;
  }

 private:
  ideep::tensor::dims new_shape_;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_IDEEP_OPERATOR(Reshape, IDEEPReshapeOp);

} // namespace
