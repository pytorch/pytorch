#include "caffe2/operators/utility_ops.h"
#include "caffe2/core/operator.h"
#include "caffe2/ideep/ideep_utils.h"

using namespace caffe2;

namespace {

class CopyCPUToIDEEPOp final : public IDEEPOperator {
 public:
  USE_SIMPLE_IDEEP_CTOR_DTOR(CopyCPUToIDEEPOp);
  USE_IDEEP_DEF_ALIASES();

  bool RunOnDevice() override {
    const auto& X = OperatorBase::Input<Tensor>(0, CPU);
    auto* Y = OperatorBase::OutputBlob(0);
    itensor::dims src_dims(X.sizes().begin(), X.sizes().end());
    if (!(Y->template IsType<itensor>() &&
          Y->Get<itensor>().get_data_type() == itensor::data_type::f32) ||
        Y->Get<itensor>().get_dims() != src_dims) {
      Y->Reset(new itensor());
      Y->GetMutable<itensor>()->resize(src_dims, itensor::data_type::f32);
    }
    Y->GetMutable<itensor>()->feed_from(
        src_dims, itensor::data_type::f32, X.raw_data());
    return true;
  }
};

class IDEEPCopyOp final : public IDEEPOperator {
 public:
  USE_SIMPLE_IDEEP_CTOR_DTOR(IDEEPCopyOp);
  USE_IDEEP_DEF_ALIASES();

  bool RunOnDevice() override {
    const auto& X = OperatorBase::Input<itensor>(0);
    auto* Y = Output(0);
    if (X != *Y) {
      Y->reinit_like(X);
      ideep::direct_copy::compute(X, *Y);
    }

    return true;
  }
};

class CopyIDEEPToCPUOp final : public IDEEPOperator {
 public:
  USE_SIMPLE_IDEEP_CTOR_DTOR(CopyIDEEPToCPUOp);
  USE_IDEEP_DEF_ALIASES();
  bool RunOnDevice() override {
    const auto& input_blob = OperatorBase::InputBlob(0);
    if (BlobIsTensorType(input_blob, CPU)) {
      VLOG(2) << "Directing sharing of TensorCPU";
      const auto& X = OperatorBase::Input<Tensor>(0, CPU);
      OutputTensorCopyFrom(0, at::device(CPU), X);
    } else {
      const auto& X = OperatorBase::Input<itensor>(0);
      if (X.get_data_type() == itensor::data_type::f32) {
        std::vector<int64_t> dims;
        for (int i = 0; i < X.get_dims().size(); ++i) {
          dims.push_back(X.get_dims()[i]);
        }
        auto* Y =
            OperatorBase::OutputTensor(0, dims, at::dtype<float>().device(CPU));
        X.to_public(Y->template mutable_data<float>());
      } else {
        CAFFE_THROW("Unsupported ideep type: ",
                    static_cast<int>(X.get_data_type()));
      }
    }
    return true;
  }
};

class IDEEPWeightedSumOp : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPWeightedSumOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws) {}
  bool RunOnDevice() override {
    CAFFE_ENFORCE_EQ(InputSize() % 2, 0);
    auto ndims = Input(0).ndims();
    auto nelems = Input(0).get_nelems();
    auto w_nelems = Input(1).get_nelems();
    CAFFE_ENFORCE_GT(nelems, 0);
    CAFFE_ENFORCE_EQ(w_nelems, 1);
    auto* output = Output(0);
    std::vector<float> scales;
    scales.reserve(InputSize() / 2);
    std::vector<itensor> inputs;
    inputs.reserve(InputSize() / 2);
    for (int i = 0; i < InputSize(); i += 2) {
      auto& X = Input(i);
      CAFFE_ENFORCE(X.ndims() == ndims);
      CAFFE_ENFORCE(X.get_nelems() == nelems);
      CAFFE_ENFORCE(Input(i + 1).get_nelems() == w_nelems);
      inputs.push_back(X);
      auto scale = static_cast<float *>(Input(i + 1).get_data_handle());
      scales.push_back(scale[0]);
    }

    ideep::sum::compute(scales, inputs, *output);

    return true;
  }
};

REGISTER_IDEEP_OPERATOR(CopyCPUToIDEEP, CopyCPUToIDEEPOp);
REGISTER_IDEEP_OPERATOR(CopyIDEEPToCPU, CopyIDEEPToCPUOp);
REGISTER_IDEEP_OPERATOR(Copy, IDEEPCopyOp);
REGISTER_IDEEP_OPERATOR(WeightedSum, IDEEPWeightedSumOp);

OPERATOR_SCHEMA(CopyCPUToIDEEP)
    .NumInputs(1)
    .NumOutputs(1)
    .Input(0, "cpu_blob", "The input TensorCPU to copy")
    .Output(0, "ideep_blob", "The output IDEEP tensort to copy to");
OPERATOR_SCHEMA(CopyIDEEPToCPU)
    .NumInputs(1)
    .NumOutputs(1)
    .Input(0, "ideep_blob", "The input IDEEP tensort to copy")
    .Output(0, "cpu_blob", "The output TensorCPU to copy to");

} // namespace
