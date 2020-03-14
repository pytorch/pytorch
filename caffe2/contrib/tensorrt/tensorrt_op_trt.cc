#include "caffe2/contrib/tensorrt/tensorrt_op_trt.h"

#include <numeric>
#include <unordered_map>

#include "caffe2/contrib/tensorrt/tensorrt_tranformer.h"
#include "caffe2/core/logging.h"
#include "onnx/onnx_pb.h"

namespace caffe2 {

namespace {
// Note that input of trt tensor is in CHW format, while our tensor is NCHW
// \return -1 if there is dimension mismatch between C2 tensor and trt tensor.
// Otherwise, return the product of CHW dimensions
int64_t CheckDims(
    const nvinfer1::Dims& nv_dims,
    at::ArrayRef<int64_t> c2_dims) {
  if (nv_dims.nbDims + 1 != c2_dims.size()) {
    CAFFE_THROW(
        "Mismatched dimensions between TRT input (",
        nv_dims.nbDims + 1,
        ") and C2 input (",
        c2_dims.size(),
        ")");
  }
  int64_t chw = 1;
  for (int i = 0; i < nv_dims.nbDims; ++i) {
    if (nv_dims.d[i] != c2_dims[i + 1]) {
      CAFFE_THROW(
          "Mismatched value at dimension ",
          i,
          "  between TRT input (",
          nv_dims.d[i],
          ") and C2 input (",
          c2_dims[i + 1],
          ")");
    }
    chw *= nv_dims.d[i];
  }
  return chw;
}

} // namespace

// Upon construction, we build the inference engine by deserializing from
// protobuf string. And since we know the input/output blobs, we can do the
// binding here too.
TensorRTOp::TensorRTOp(const OperatorDef& operator_def, Workspace* ws)
    : Operator<CUDAContext>(operator_def, ws),
      logger_(
          (nvinfer1::ILogger::Severity)(OperatorBase::GetSingleArgument<int>(
              "log_verbosity",
              FLAGS_caffe2_log_level))),
      max_batch_size_(
          OperatorBase::GetSingleArgument<int>("max_batch_size", 1)) {
  {
    auto engine_string =
        OperatorBase::GetSingleArgument<std::string>("backend_buffer", "");
    if (!engine_string.empty()) {
      auto trt_runtime =
          tensorrt::TrtObject(nvinfer1::createInferRuntime(logger_));
      // TODO(support trt plugin factory)
      trt_engine_ = tensorrt::TrtObject(trt_runtime->deserializeCudaEngine(
          engine_string.data(), engine_string.size(), nullptr));
    } else {
      auto onnx_model_str =
          OperatorBase::GetSingleArgument<std::string>("onnx_model", "");
      CAFFE_ENFORCE(!onnx_model_str.empty(), "onnx_model cannot be empty");
      auto debug_builder = OperatorBase::GetSingleArgument<int>("debug_builder", 0);
      auto max_workspace_size = OperatorBase::GetSingleArgument<int>(
          "max_workspace_size", 1024 * 1024 * 2);

      // Pull the weights from workspace and assembly it back to the onnx model,
      // notice that since we may have rewritten the net, we need to map the
      // weight names
      auto initializers = OperatorBase::GetRepeatedArgument<std::string>("initializers");
      CAFFE_ENFORCE_EQ(
          initializers.size() % 2, 0, "initializers should come in pairs");
      std::unordered_set<std::string> initializer_set;
      std::unordered_map<std::string, std::string> input_mapping;
      for (auto it = initializers.begin(); it != initializers.end(); ++it)  {
        auto key = *it++;
        input_mapping.emplace(key, *it);
        initializer_set.emplace(key);
      }
      Workspace mapped_ws(ws, input_mapping);
      ::ONNX_NAMESPACE::ModelProto onnx_model;
      ParseProtoFromLargeString(onnx_model_str, &onnx_model);
      BuildInitializationList(&mapped_ws, onnx_model.mutable_graph(), &initializer_set);
      onnx_model_str.clear();
      onnx_model.SerializeToString(&onnx_model_str);

      // Build the trt engine
      trt_engine_ = tensorrt::BuildTrtEngine(
          onnx_model_str,
          &logger_,
          max_batch_size_,
          max_workspace_size,
          debug_builder);
    }
  }

  CAFFE_ENFORCE(trt_engine_, "Cannot build TensorRT engine!");

  // match and bind the input/output
  const int num_bindings = trt_engine_->getNbBindings();
  int output_idx = 0;
  for (int b = 0; b < num_bindings; ++b) {
    nv_dims_.push_back(trt_engine_->getBindingDimensions(b));
    bool is_input = trt_engine_->bindingIsInput(b);
    is_input_.push_back(is_input);
    if (!is_input) {
      // For output, we try to get its output size hint
      const std::string key = c10::str("output_size_hint_", output_idx);
      auto output_size_hint = OperatorBase::GetRepeatedArgument<int>(key);
      if (!output_size_hint.empty()) {
        std::vector<int64_t> dims;
        for (const auto v : output_size_hint) {
          dims.push_back(v);
        }
        output_size_hints_.emplace(output_idx, std::move(dims));
      }
      ++output_idx;
    }
  }

  trt_executor_ = tensorrt::TrtObject(trt_engine_->createExecutionContext());
}

void TensorRTOp::MaybeAdjustOutputShape(
    int output_idx,
    std::vector<int64_t>* dims) {
  const auto it = output_size_hints_.find(output_idx);
  if (it != output_size_hints_.end()) {
    const auto& dims_hint = it->second;
    auto total_trt = std::accumulate(
        dims->begin(), dims->end(), (int64_t)(1), std::multiplies<int64_t>());
    auto total_c2 = std::accumulate(
        dims_hint.begin(),
        dims_hint.end(),
        (int64_t)(1),
        std::multiplies<int64_t>());
    CAFFE_ENFORCE_EQ(
        total_trt,
        total_c2,
        "The total size of TensorRT op output and hint don't match: ",
        total_trt,
        " vs ",
        total_c2);

    // We conform to the output shape hints. NB: We might need an explicit
    // reshape op for this
    *dims = dims_hint;
  }
}

bool TensorRTOp::RunOnDevice() {
  CAFFE_ENFORCE(trt_executor_);
  // Decide input batch size
  size_t N = 0;
  for (int i = 0; i < InputSize(); ++i) {
    const auto& input_tensor = Input(i);
    const auto tensor_dims = input_tensor.sizes();
    CAFFE_ENFORCE(!tensor_dims.empty(), "Input tensor cannot be empty");
    if (i == 0) {
      N = tensor_dims.front();
    } else {
      CAFFE_ENFORCE_EQ(
          N, tensor_dims.front(), "Mismatched batch size in input tensors");
    }
  }
  if (N > max_batch_size_ && !batch_warning_issued_) {
    LOG(WARNING) << "Batch size (" << N << ") is larger than max_batch_size ("
                 << max_batch_size_ << ") optimized for TensorRT operator. "
                 << "Performance may be sub-optimal.";
    batch_warning_issued_ = true;
  }

  // We need to do the binding at RunOnDevice time because we only know the
  // exact shapes of the tensors now. In addition, since TensorRT engine has
  // max_batch_size, we need to call that multiple times if input batch size
  // exceeeds this limit.
  CAFFE_ENFORCE_EQ(is_input_.size(), nv_dims_.size());
  std::vector<void*> bindings;
  bindings.reserve(is_input_.size());
  auto batch_size = max_batch_size_;
  for (size_t offset = 0; offset < N; offset += batch_size) {
    bindings.clear();
    batch_size = std::min<size_t>(N - offset, max_batch_size_);
    VLOG(2) << "Offset: " << offset << ", batch_size: " << batch_size
            << ", N: " << N;
    int input_idx = 0;
    int output_idx = 0;
    for (auto i = 0; i < is_input_.size(); ++i) {
      const auto& dims = nv_dims_[i];
      if (is_input_[i]) {
        // input, check input dimensions
        const auto& input_tensor = Input(input_idx++);
        const float* input_data = input_tensor.data<float>();
        const auto tensor_dims = input_tensor.sizes();
        auto chw = CheckDims(dims, tensor_dims);
        bindings.push_back((void*)(input_data + offset * chw));
      } else {
        // output, we need to allocate the output tensor at first batch run
        auto* output_tensor = Output(output_idx);
        std::vector<int64_t> tensor_dims;
        tensor_dims.push_back(N);
        int64_t chw = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
          tensor_dims.push_back(dims.d[i]);
          chw *= dims.d[i];
        }

        if (offset == 0) {
          MaybeAdjustOutputShape(output_idx, &tensor_dims);
          output_tensor->Resize(tensor_dims);
        }
        ++output_idx;
        float* output_data = output_tensor->mutable_data<float>();
        bindings.push_back((void*)(output_data + offset * chw));
      }
    }

    CAFFE_ENFORCE_EQ(bindings.size(), InputSize() + OutputSize());
    if (!trt_executor_->execute(batch_size, bindings.data())) {
      CAFFE_THROW("Error running the TensorRT executor");
    }
  }
  return true;
}

OPERATOR_SCHEMA(TensorRT)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .SetDoc(R"DOC(
The TensorRT operator is a black-box operator serialized from prebuilt TensorRT
Engine string. It will take the input, do the computation by calling TensorRT
inference engine and generate the outputs.

This is a GPU only operator.
)DOC")
    .Arg(
        "log_verbosity",
        "(int default 0) verbosity of the TensorRt engine log.")
    .Arg(
        "backend_buffer",
        "(string default=\"\" blob for serialized TensorRT engine."
        "Note that serialized engine is not compatible across platform and "
        "different TensorRT version.")
    .Arg(
        "max_batch_size",
        "(int default 0) Batch size set by the TensorRT engine builder."
        "It must be no larger than the max_batch_size of the engine builder so "
        "it is better not to edit this manually.");

REGISTER_CUDA_OPERATOR(TensorRT, TensorRTOp);
} // namespace caffe2
