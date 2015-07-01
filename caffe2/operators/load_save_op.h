#ifndef CAFFE2_OPERATORS_LOAD_SAVE_OP_H_
#define CAFFE2_OPERATORS_LOAD_SAVE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/proto_utils.h"
#include "glog/logging.h"

namespace caffe2 {

// LoadFloatTensorOp is a very simple operator that loads a TensorProto stored
// on disk. The TensorProto should only be stored in float form.
template <class DeviceContext>
class LoadFloatTensorOp final : public Operator<float, DeviceContext> {
 public:
  LoadFloatTensorOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<float, DeviceContext>(operator_def, ws),
        filename_(OperatorBase::GetSingleArgument<string>("filename", "")) {
    CHECK_GT(filename_.size(), 0) << "Must specify an input file.";
  }

  bool RunOnDevice() override {
    TensorProtos protos;
    CHECK(ReadProtoFromFile(filename_, &protos));
    // TODO(Yangqing): Add capability to allow loading a subset of the protos.
    CHECK_EQ(protos.protos_size(), OperatorBase::OutputSize())
        << "Inconsistent number of tensors.";
    int i = 0;
    for (const auto& proto : protos.protos()) {
      CHECK_GT(proto.dims_size(), 0);
      CHECK_EQ(proto.data_type(), TensorProto::FLOAT);
      auto* output = OperatorBase::Output<Tensor<float, DeviceContext> >(i);
      output->Reshape(vector<int>(proto.dims().begin(), proto.dims().end()));
      CHECK_EQ(output->size(), proto.float_data_size());
      this->device_context_.template Copy<float, DeviceContext, CPUContext>(
          output->mutable_data(), proto.float_data().data(), output->size());
      VLOG(1) << "Loaded tensor " << this->def().output(i);
      ++i;
    }
    return true;
  }

 private:
  string filename_;
  INPUT_OUTPUT_STATS(0, 0, 1, INT_MAX);
  DISABLE_COPY_AND_ASSIGN(LoadFloatTensorOp);
};

// SaveFloatTensorOp is a very simple operator that loads a TensorProto stored
// on disk. The TensorProto should only be stored in float form.
template <class DeviceContext>
class SaveFloatTensorOp final : public Operator<float, DeviceContext> {
 public:
  SaveFloatTensorOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<float, DeviceContext>(operator_def, ws),
        filename_(OperatorBase::GetSingleArgument<string>("filename", "")) {}

  bool RunOnDevice() override {
    TensorProtos protos;
    for (int i = 0; i < OperatorBase::InputSize(); ++i) {
      auto& input = OperatorBase::Input<Tensor<float, DeviceContext> >(i);
      auto* proto = protos.add_protos();
      proto->set_data_type(TensorProto::FLOAT);
      proto->set_name(OperatorBase::def().input(i));
      for (int dim : input.dims()) {
        proto->add_dims(dim);
      }
      // Note(Yangqing): there is no way in protobuffer to resize a repeated
      // field, so we have to do reserve and insert dummy zeros.
      proto->mutable_float_data()->Reserve(input.size());
      for (int i = 0; i < input.size(); ++i) {
        proto->add_float_data(0);
      }
      this->device_context_.template Copy<float, CPUContext, DeviceContext>(
          proto->mutable_float_data()->mutable_data(),
          input.data(), input.size());
    }
    WriteProtoToBinaryFile(protos, filename_);
    return true;
  }

 private:
  string filename_;
  INPUT_OUTPUT_STATS(1, INT_MAX, 0, 0);
  DISABLE_COPY_AND_ASSIGN(SaveFloatTensorOp);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_LOAD_SAVE_OP_H_
