#pragma once

#include <caffe2/core/common.h>
#include <caffe2/core/context.h>
#include <caffe2/core/operator.h>
#include <caffe2/ideep/ideep_utils.h>
#include <caffe2/proto/caffe2_pb.h>

namespace caffe2 {

/**
 * @brief A templated class to allow one to wrap a CPU operator as an IDEEP
 * operator.
 *
 * This class can be used when one does not have the IDEEP implementation ready
 * yet for an operator. Essentially, what this op does is to automatically
 * deal with data copy for you. Plausibly, this causes a lot of overhead and
 * is not optimal, so you should use this operator mostly for quick prototyping
 * purpose.
 *
 * All the input and output of the original operator should be TensorCPU.
 *
 * Example usage: if you have a class MyMagicOp that is CPU based, and you use
 * the registration code
 *     REGISTER_CPU_OPERATOR(MyMagic, MyMagicOp);
 * to register the CPU side, you can create its corresponding IDEEP operator
 * (with performance hits of course) via
 *     REGISTER_IDEEP_OPERATOR(MyMagic,
 *                            IDEEPFallbackOp<MyMagicOp>);
 *
 * Advanced usage: if you want to have some specific outputs never copied, you
 * can use the SkipOutputCopy template argument to do that. For example, if
 * MyMagic produces two outputs and the first output is always going to live on
 * the CPU, you can do
 *     REGISTER_IDEEP_OPERATOR(MyMagic,
 *                            IDEEPFallbackOp<MyMagicOp, SkipIndices<0>>);
 */
template <class CPUOp, typename SkipOutputCopy = SkipIndices<>>
class C10_EXPORT IDEEPFallbackOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPFallbackOp(const OperatorDef& def, Workspace* ws)
      : IDEEPOperator(def, ws) {
    CAFFE_ENFORCE_EQ(def.device_option().device_type(), PROTO_IDEEP);
    base_def_.CopyFrom(def);
    // base_def_ runs on CPU, so we will set its device option to CPU.
    // Copy to allow random_seed to be correctly propagated.
    base_def_.mutable_device_option()->CopyFrom(def.device_option());
    base_def_.mutable_device_option()->set_device_type(PROTO_CPU);
    // Create output blobs in parent workspace,
    // then forward output blobs to local workspace.
    std::unordered_map<string, string> forwarded_output_blobs;
    for (int i = 0; i < base_def_.output_size(); i++) {
      // For in-place case, the in/output tensor for local_ws must be
      // re-created, instead of forwarding from current workspace.
      string parent_name(base_def_.output(i));
      if (!SkipOutputCopy::Contains(i)) {
        parent_name += "_cpu_output_blob_" + base_def_.type();
      }
      local_output_blobs_.push_back(ws->CreateBlob(parent_name));
      CHECK_NOTNULL(local_output_blobs_.back());
      forwarded_output_blobs[base_def_.output(i)] = parent_name;
      output_inplace_.push_back(false);
      for (const string &input_name : base_def_.input()) {
        if (input_name == base_def_.output(i)) {
          output_inplace_[i] = true;
          break;
        }
      }
    }
    local_ws_.reset(new Workspace(ws, forwarded_output_blobs));
    // Set up the symbols for the local workspace.
    for (const string& name : base_def_.input()) {
      local_input_blobs_.push_back(local_ws_->CreateBlob(name));
      CHECK_NOTNULL(local_input_blobs_.back());
    }
    input_share_.resize(local_input_blobs_.size(), false);
    base_op_.reset(new CPUOp(base_def_, local_ws_.get()));
  }

  bool RunOnDevice() override {
    for (int i = 0; i < InputSize(); ++i) {
      if (InputIsType<itensor>(i)
          && (Input(i).has_scale()
            || Input(i).get_data_type() == idtype::f32)) {
        auto& input = Input(i);
        if (input_share_[i]) {
          local_input_blobs_[i]->Reset();
          input_share_[i] = false;
        }
        auto dtensor = BlobGetMutableTensor(local_input_blobs_[i], CPU);
        dtensor->Resize(input.get_dims());
        // If fallback from INT8, the public format of original input is nhwc.
        // While the required format is nchw, need to reorder to nchw.
        if (input.get_public_format() == iformat::nhwc) {
          itensor temp_ten ({input.get_dims(), idtype::f32, iformat::nchw},
              dtensor->template mutable_data<float>());
          temp_ten.feed_from(input);
        } else if (!input.need_reorder()) {
          CAFFE_ENFORCE(!input.has_scale(),
              "Incorrect invocation of get_data_handle");
          dtensor->ShareExternalPointer(
              static_cast<float*>(input.get_data_handle()));
        } else {
          input.to_public(dtensor->template mutable_data<float>());
        }
      } else {
        VLOG(1) << "Input " << i << " is not ideep::tensor. Skipping copy.";
        if (OperatorBase::Inputs()[i]->GetRaw() != local_input_blobs_[i]->GetRaw()) {
          // Note(jiayq): This removes a const but conceptually
          // local_input_blobs will only be used as const blob input for the
          // base op so we are still fine.
          local_input_blobs_[i]->ShareExternal(
              const_cast<void *>(OperatorBase::Inputs()[i]->GetRaw()),
              OperatorBase::Inputs()[i]->meta());
        }
        input_share_[i] = true;
      }
    }

    // Some CPU ops inherited from OperatorBase directly might need this default
    // input argument '0' like 'PrefetchOperator'.
    if (!base_op_->Run(0)) {
      LOG(ERROR) << "Base op run failed in IDEEPFallbackOp. Def: "
                 << ProtoDebugString(this->debug_def());
      return false;
    }

    for (int i = 0; i < OutputSize(); ++i) {
      if (SkipOutputCopy::Contains(i)) {
        VLOG(1) << "Copy output: index " << i << " skipped.";
        continue;
      }
      CAFFE_ENFORCE(
          BlobIsTensorType(*local_output_blobs_[i], CPU),
          "IDEEP fallback op currently does not support non-TensorCPU "
          "output type who needs copying.");
      const auto& src = local_output_blobs_[i]->template Get<TensorCPU>();
      auto src_dims = src.sizes().vec();
      if (src.template IsType<float>() && src.dim() != 0 && base_op_->type() != "Python") {
        Blob* dst = OperatorBase::OutputBlob(i);
        // The output tensor must be ideep tensor with public format.
        // If reusing ideep tensor with non-public format, the tensor buffer
        // will be interpreted incorrectly.
        if (!dst->template IsType<itensor>() ||
            !dst->template Get<itensor>().is_public_format()) {
          dst->Reset(new itensor());
        }

        itensor::dims dst_dims (src_dims.begin(), src_dims.end());
        auto dtensor = dst->template GetMutable<itensor>();
        if (dtensor->get_dims() != dst_dims) {
          dtensor->resize(dst_dims, idtype::f32);
        }
        if (output_inplace_[i]) {
          dtensor->feed_from(dst_dims, idtype::f32,
              const_cast<void*>(src.raw_data()));
        } else {
          CAFFE_ENFORCE(!dtensor->has_scale(),
              "Incorrect invocation of set_data_handle");
          dtensor->set_data_handle(const_cast<void *>(src.raw_data()));
        }
      } else {
        VLOG(2) << "Output " << base_def_.output(i) << " as CPUTensor";
        Blob* dst = OperatorBase::OutputBlob(i);
        if (output_inplace_[i]) {
          auto dtensor = BlobGetMutableTensor(dst, CPU);
          dtensor->CopyFrom(src);
        } else {
          dst->Reset(new Tensor(CPU));
          BlobSetTensor(dst, src.Alias());
        }
      }
    }
    return true;
  }

 protected:
  vector<Blob*> local_input_blobs_;
  vector<Blob*> local_output_blobs_;
  vector<bool> output_inplace_;
  vector<bool> input_share_;
  std::unique_ptr<CPUOp> base_op_;
  std::unique_ptr<Workspace> local_ws_;
  OperatorDef base_def_;
};

} // namespace caffe2
