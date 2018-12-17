#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

class IDEEPConcatOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPConcatOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws) {
    CAFFE_ENFORCE(
      !(OperatorBase::HasArgument("axis") && OperatorBase::HasArgument("order")),
        "You shouldn't specify both the dim to concat, and the order "
        "in the case of 4-D images.");
    if (OperatorBase::HasArgument("axis")) {
      axis_ = OperatorBase::GetSingleArgument<int>("axis", -1);
      add_axis_ = OperatorBase::GetSingleArgument<int>("add_axis", 0);
    } else {
      axis_ = 1;
      add_axis_ = 0;
    }
    CAFFE_ENFORCE_GE(axis_, 0);
  }
  virtual ~IDEEPConcatOp() {}

  bool RunOnDevice() override {
    auto* output = Output(OUTPUT);
    TensorCPU* axis_info = OperatorBase::Output<TensorCPU>(AXIS_INFO, CPU);

    vector<itensor> inputs;
    for (int i = 0; i < InputSize(); ++i) {
      if (OperatorBase::InputBlob(i).template IsType<itensor>()) {
        inputs.emplace_back(Input(i));
      } else {
        CAFFE_ENFORCE(
            BlobIsTensorType(OperatorBase::InputBlob(i), CPU),
            "Expect cpu tensor if not itensor");
        auto& tensor_cpu = OperatorBase::Input<Tensor>(i, CPU);
        CAFFE_ENFORCE(
            tensor_cpu.sizes().size() == 0 || tensor_cpu.numel() == 0,
            "Expect zero dim tensor");
      }
    }

    auto axis_vdata = ideep::concat::compute(inputs, axis_, add_axis_, *output);
    axis_info->Resize(vector<int64_t>(1, InputSize()));
    int* axis_data = axis_info->template mutable_data<int>();
    for (int i = 0; i < axis_vdata.size(); i++) {
      axis_data[i] = axis_vdata[i];
    }

    return true;
  }

 private:
  int axis_;
  int add_axis_;

  INPUT_TAGS(INPUT0);
  OUTPUT_TAGS(OUTPUT, AXIS_INFO);
};

class IDEEPSplitOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPSplitOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        axis_offset_(OperatorBase::GetRepeatedArgument<int>("split")) {
    CAFFE_ENFORCE(
      !(OperatorBase::HasArgument("axis") && OperatorBase::HasArgument("order")),
        "You shouldn't specify both the dim to split, and the order "
        "in the case of 4-D images.");
    if (OperatorBase::HasArgument("axis")) {
      axis_ = OperatorBase::GetSingleArgument<int>("axis", -1);
      // only exists for computing the gradient of a Concat with 'add_axis'
      add_axis_ = OperatorBase::GetSingleArgument<int>("add_axis", 0);
    } else {
      axis_ = 1;
      add_axis_ = 0;
    }
    CAFFE_ENFORCE_GE(axis_, 0);
  }
  virtual ~IDEEPSplitOp() {}

  bool RunOnDevice() override {
    const auto& input = Input(INPUT);
    CAFFE_ENFORCE_LT(axis_, input.ndims(), "Axis not in input ndim range.");
    const int input_channels = input.get_dim(axis_);
    vector<int> axis_vdata(OutputSize(), 0);
    if (InputSize() == 2) {
      // We obtain split from the input tensor.
      CAFFE_ENFORCE_EQ(
          axis_offset_.size(),
          0,
          "If you set split with an input blob, do not pass in "
          "split in the argument.");
      auto& axis_info = OperatorBase::Input<Tensor>(AXIS_INFO, CPU);
      CAFFE_ENFORCE_EQ(axis_info.numel(), OutputSize());
      auto* axis_data = axis_info.template data<int>();
      axis_vdata.assign(axis_data, axis_data + OutputSize());
    } else if (axis_offset_.size() == 0) {
      CAFFE_ENFORCE_EQ(
          input_channels % OutputSize(),
          0,
          "If you did not specify split explicitly, the number of "
          "input channels should be divisible by the output size.");
      axis_vdata.assign(OutputSize(), input_channels / OutputSize());
    } else {
      // We obtain split from the parameters.
      CAFFE_ENFORCE_EQ(
          axis_offset_.size(),
          OutputSize(),
          "The number of splits specified should be equal to the "
          "number of outputs.");
      axis_vdata = axis_offset_;
    }

    CAFFE_ENFORCE_EQ(
        add_axis_ ? OutputSize()
                  : std::accumulate(
                    axis_vdata.data(), axis_vdata.data() + OutputSize(), 0),
        input_channels,
        "Sum of split dimensions do not match: should be ",
        input_channels);

    auto iten_vector = ideep::spliter::compute(
        input, axis_vdata, axis_, add_axis_);
    CAFFE_ENFORCE_EQ(
        iten_vector.size(),
        OutputSize(),
        "Output size does not match: should be ",
        OutputSize());

    for (int i = 0; i < OutputSize(); i++) {
      auto* output = Output(i);
      *output = iten_vector[i];
    }

    return true;
  }

 private:
  int axis_;
  int add_axis_;
  vector<int> axis_offset_;

  INPUT_TAGS(INPUT, AXIS_INFO);
};


REGISTER_IDEEP_OPERATOR(Concat, IDEEPConcatOp);
REGISTER_IDEEP_OPERATOR(Split, IDEEPSplitOp);

} // namespace caffe2
