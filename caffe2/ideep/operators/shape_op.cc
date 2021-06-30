#include <caffe2/ideep/ideep_utils.h>

using namespace caffe2;

namespace {

// RecordShapeOp records the shape of the input tensor to a vector of int. You
// mostly don't need this operator explicitly, and it is mostly used in the
// autodiff process.
class IDEEPShapeOp : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPShapeOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        axes_(OperatorBase ::GetRepeatedArgument<int>("axes")) {}

  bool RunOnDevice() override {
    int numDims = 0;
    int numAxes = axes_.size();
    vector<int64_t> dims;
    const char* data_dims = nullptr;
    auto* output = OperatorBase::Output<Tensor>(OUTPUT, CPU);

    if (OperatorBase::InputBlob(DATA).template IsType<itensor>()) {
      auto& data = Input(DATA);
      numDims = data.ndims();
      auto idims = data.get_dims();
      dims.assign(idims.begin(), idims.end());
      data_dims = reinterpret_cast<const char*>(dims.data());
    } else {
      auto& data = OperatorBase::Input<Tensor>(DATA, CPU);
      numDims = data.dim();
      data_dims = reinterpret_cast<const char*>(data.sizes().data());
    }

    if (numAxes == 0) {
      output->Resize(numDims);
      int64_t* output_data = output->template mutable_data<int64_t>();
      context_.CopyBytesSameDevice(
          numDims * sizeof(int64_t), data_dims, output_data);
      return true;
    }

    output->Resize(numAxes);
    auto out = reinterpret_cast<char*>(output->template mutable_data<int64_t>());
    for (int i = 0; i < numAxes; i++) {
      auto axis = axes_[i];
      CAFFE_ENFORCE_LT(axis, numDims, "Axis out of range");
      CAFFE_ENFORCE_GE(axis, 0, "Each axis should be non-negative");
      context_.CopyBytesSameDevice(
          sizeof(int64_t), data_dims + axis * sizeof(int64_t), out);
      out += sizeof(int64_t);
    }

    return true;
  }

 private:
  vector<int> axes_;

  INPUT_TAGS(DATA);
  OUTPUT_TAGS(OUTPUT);
};


// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_IDEEP_OPERATOR(Shape, IDEEPShapeOp);

} // namespace
