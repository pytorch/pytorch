#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/server/caffe2_dnnlowp_utils.h"
#include "caffe2/operators/quantized/server/dnnlowp.h"

namespace caffe2 {

bool ShouldFp32FallbackToNCHW(const OperatorDef& def);

/**
 * Wrap a floating-point operator with quantized inputs with type T.
 * This class is to measure quantization error against fp32 reference.
 */
template<typename OpType, typename T>
class OpWrapper {
 public :
  OpWrapper(OperatorBase* op, dnnlowp::QuantizationFactory* qfactory)
      : op_(op), qfactory_(qfactory) {
    for (auto name : op->debug_def().input()) {
      local_input_blobs_.push_back(local_ws_.CreateBlob(name));
      CHECK_NOTNULL(local_input_blobs_.back());
    }
    OperatorDef def = op->debug_def();
    if (ShouldFp32FallbackToNCHW(def)) {
      // C2 default Conv operator doesn't support 3D convolution in NHWC
      Argument* arg = GetMutableArgument("order", false, &def);
      arg->set_s("NCHW");
      std::string new_order =
          ArgumentHelper::GetSingleArgument<OperatorDef, std::string>(
              def, "order", "");
      assert(new_order == "NCHW");
    }
    local_op_.reset(new OpType(def, &local_ws_));
    for (auto name : def.output()) {
      local_output_blobs_.push_back(local_ws_.GetBlob(name));
      CHECK_NOTNULL(local_output_blobs_.back());
    }
  }

  void DequantizeInput() {
    const OperatorDef& def = op_->debug_def();
    CPUContext context(def.device_option());
    bool fallback_to_nchw = ShouldFp32FallbackToNCHW(def);

    for (int i = 0; i < op_->InputSize(); ++i) {
      if (op_->InputIsType<int8::Int8TensorCPU>(i)) {
        const TensorCPU& qtensor = op_->Input<int8::Int8TensorCPU>(i).t;
        TensorCPU *float_tensor =
          BlobGetMutableTensor(local_input_blobs_[i], CPU);
        if (fallback_to_nchw && i < 2) {
          // NHWC2NCHW for input
          std::vector<T> temp(qtensor.numel());

          int ndim = qtensor.dim();
          std::vector<int> dims(qtensor.sizes().begin(), qtensor.sizes().end());
          std::vector<int> axes(ndim);
          axes[0] = 0;
          axes[1] = ndim - 1;
          for (auto j = 1; j < ndim - 1; ++j) {
            axes[j + 1] = j;
          }

          std::vector<int> new_dims(ndim);
          for (auto j = 0; j < ndim; ++j) {
            new_dims[j] = dims[axes[j]];
          }
          float_tensor->Resize(new_dims);

          math::Transpose(
              ndim,
              dims.data(),
              axes.data(),
              qtensor.data<T>(),
              temp.data(),
              &context);

          Dequantize(
              temp.data(),
              float_tensor->template mutable_data<float>(),
              qtensor.numel(),
              dnnlowp::GetInputTensorQuantizationParamsOf(op_, i, qfactory_));
        } else {
          // FIXME: doesn't work for bias so we shouldn't quantize bias before
          // model loading.
          float_tensor->ResizeLike(qtensor);
          Dequantize(
              qtensor.data<T>(),
              float_tensor->template mutable_data<float>(),
              qtensor.numel(),
              dnnlowp::GetInputTensorQuantizationParamsOf(op_, i, qfactory_));
        }
      } else {
        if (fallback_to_nchw && i < 2) {
          // NHWC2NCHW for input
          const TensorCPU& in_tensor = op_->Input<Tensor>(i, CPU);
          TensorCPU* float_tensor =
            BlobGetMutableTensor(local_input_blobs_[i], CPU);

          int ndim = in_tensor.dim();
          std::vector<int> dims(
              in_tensor.sizes().begin(), in_tensor.sizes().end());
          std::vector<int> axes(ndim);
          axes[0] = 0;
          axes[1] = ndim - 1;
          for (int j = 1; j < ndim - 1; ++j) {
            axes[j + 1] = j;
          }

          std::vector<int> new_dims(ndim);
          for (auto j = 0; j < ndim; ++j) {
            new_dims[j] = dims[axes[j]];
          }
          float_tensor->Resize(new_dims);

          math::Transpose(
              ndim,
              dims.data(),
              axes.data(),
              in_tensor.data<float>(),
              float_tensor->mutable_data<float>(),
              &context);
        } else {
          local_input_blobs_[i]->ShareExternal(
              const_cast<void*>(op_->Inputs()[i]->GetRaw()),
              op_->Inputs()[i]->meta());
        }
      }
    }
  }

  OpType *Get() { return local_op_.get(); }

  dnnlowp::TensorQuantizationParams
    GetOutputQuantizationParams(
        dnnlowp::QuantizationFactory *qfactory, int index = 0) {
    using namespace dnnlowp;

    float min, max;
    auto& out_tensor = local_output_blobs_[index]->template Get<TensorCPU>();
    FindMinMax(
        out_tensor.template data<float>(), &min, &max, out_tensor.numel());
    if (op_->OperatorBase::GetSingleArgument<std::string>("followed_by", "") ==
        "Relu") {
      min = std::max(0.0f, min);
      max = std::max(0.0f, max);
    }

    return qfactory->ChooseQuantizationParams(min, max);
  }

 private :
  OperatorBase* op_; /* container quantized op */
  Workspace local_ws_;
  std::vector<Blob*> local_input_blobs_;
  std::vector<Blob*> local_output_blobs_;
  std::unique_ptr<OpType> local_op_; /* contained fp32 reference op */
  dnnlowp::QuantizationFactory* qfactory_;
};

} // caffe2
