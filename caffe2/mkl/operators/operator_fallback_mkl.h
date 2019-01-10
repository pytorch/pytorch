#ifndef CAFFE2_OPERATORS_OPERATOR_FALLBACK_H_
#define CAFFE2_OPERATORS_OPERATOR_FALLBACK_H_

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/mkl/mkl_utils.h"
#include "caffe2/proto/caffe2.pb.h"

#ifdef CAFFE2_HAS_MKL_DNN
namespace caffe2 {
namespace mkl {

/**
 * @brief A templated class to allow one to wrap a CPU operator as an MKL
 * operator.
 *
 * This class can be used when one does not have the MKL implementation ready
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
 * to register the CPU side, you can create its corresponding MKL operator
 * (with performance hits of course) via
 *     REGISTER_MKL_OPERATOR(MyMagic,
 *                            MKLFallbackOp<MyMagicOp>);
 *
 * Advanced usage: if you want to have some specific outputs never copied, you
 * can use the SkipOutputCopy template argument to do that. For example, if
 * MyMagic produces two outputs and the first output is always going to live on
 * the CPU, you can do
 *     REGISTER_MKL_OPERATOR(MyMagic,
 *                            MKLFallbackOp<MyMagicOp, SkipIndices<0>>);
 */
template <class CPUOp, typename SkipOutputCopy = SkipIndices<>>
class MKLFallbackOp final : public Operator<MKLContext> {
 public:
  USE_OPERATOR_FUNCTIONS(MKLContext);
  MKLFallbackOp(const OperatorDef& def, Workspace* ws)
      : Operator<MKLContext>(def, ws) {
    CAFFE_ENFORCE_EQ(def.device_option().device_type(), MKLDNN);
    OperatorDef base_def_(def);
    // base_def_ runs on CPU, so we will set its device option to CPU.
    // Copy to allow random_seed to be correctly propagated.
    base_def_.mutable_device_option()->CopyFrom(def.device_option());
    base_def_.mutable_device_option()->set_device_type(CPU);
    // Set up the symbols for the local workspace.
    for (const string& name : def.input()) {
      local_input_blobs_.push_back(local_ws_.CreateBlob(name));
      CHECK_NOTNULL(local_input_blobs_.back());
    }
    base_op_.reset(new CPUOp(base_def_, &local_ws_));
    for (const string& name : def.output()) {
      local_output_blobs_.push_back(local_ws_.GetBlob(name));
      CHECK_NOTNULL(local_output_blobs_.back());
    }
  }

  bool RunOnDevice() override {
    for (int i = 0; i < InputSize(); ++i) {
      if (OperatorBase::InputIsType<MKLMemory<float>>(i)) {
        OperatorBase::Input<MKLMemory<float>>(i).CopyTo(
            local_input_blobs_[i]->template GetMutable<TensorCPU>());
      } else if (OperatorBase::InputIsType<MKLMemory<double>>(i)) {
        OperatorBase::Input<MKLMemory<double>>(i).CopyTo(
            local_input_blobs_[i]->template GetMutable<TensorCPU>());
      } else {
        VLOG(1) << "Input " << i << " is not MKLMemory. Skipping copy.";
        // Note(jiayq): This removes a const but conceptually
        // local_input_blobs will only be used as const blob input for the
        // base op so we are still fine.
        local_input_blobs_[i]->ShareExternal(
            const_cast<void*>(OperatorBase::Inputs()[i]->GetRaw()),
            OperatorBase::Inputs()[i]->meta());
      }
    }

    if (!base_op_->Run()) {
      LOG(ERROR) << "Base op run failed in MKLFallbackOp. Def: "
                 << ProtoDebugString(this->debug_def());
      return false;
    }

    for (int i = 0; i < OutputSize(); ++i) {
      if (SkipOutputCopy::Contains(i)) {
        VLOG(1) << "Copy output: index " << i << " skipped.";
        continue;
      }
      CAFFE_ENFORCE(
          local_output_blobs_[i]->template IsType<TensorCPU>(),
          "MKL fallback op currently does not support non-TensorCPU "
          "output type who needs copying.");
      const auto& src = local_output_blobs_[i]->template Get<TensorCPU>();
      if (src.template IsType<float>()) {
        Blob* dst = OperatorBase::OutputBlob(i);
        if (!dst->template IsType<MKLMemory<float>>() ||
            dst->Get<MKLMemory<float>>().dims() != src.dims()) {
          dst->Reset(new MKLMemory<float>(src.dims()));
        }
        dst->GetMutable<MKLMemory<float>>()->CopyFrom(src);
      } else if (src.template IsType<double>()) {
        Blob* dst = OperatorBase::OutputBlob(i);
        if (!dst->template IsType<MKLMemory<double>>() ||
            dst->Get<MKLMemory<double>>().dims() != src.dims()) {
          dst->Reset(new MKLMemory<double>(src.dims()));
        }
        dst->GetMutable<MKLMemory<double>>()->CopyFrom(src);
      } else {
        CAFFE_THROW("MKLMemory only supports float and double.");
      }
    }
    return true;
  }

 protected:
  Workspace local_ws_;
  vector<Blob*> local_input_blobs_;
  vector<Blob*> local_output_blobs_;
  std::unique_ptr<CPUOp> base_op_;
};
}

} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
#endif // CAFFE2_OPERATORS_OPERATOR_FALLBACK_H_
