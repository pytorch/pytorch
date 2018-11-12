#pragma once

#include <unordered_map>

#include "onnx/onnx_pb.h"

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/onnx/onnxifi_init.h"
#include "caffe2/utils/string_utils.h"

namespace caffe2 {

template <typename T, typename Context>
class OnnxifiOp final : public Operator<Context> {
  struct TensorInfo {
    TensorInfo() {}
    TensorInfo(TensorInfo&&) = default;
    TensorInfo& operator=(TensorInfo&&) = default;
    std::vector<uint64_t> dims;
    uint64_t onnxifi_type;
  };

 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  OnnxifiOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    lib_ = onnx::initOnnxifiLibrary();
    CAFFE_ENFORCE(lib_, "Cannot initialize ONNXIFI library");
    auto onnx_model_str =
        this->template GetSingleArgument<std::string>("onnx_model", "");
    CAFFE_ENFORCE(!onnx_model_str.empty(), "onnx_model cannot be empty");

    // Setup input/output descriptor templates
    for (const auto& input : operator_def.input()) {
      input_desc_.push_back(onnxTensorDescriptorV1());
      input_desc_.back().name = input.c_str();
    }
    int output_idx = 0;
    for (const auto& output : operator_def.output()) {
      output_desc_.push_back(onnxTensorDescriptorV1());
      output_desc_.back().name = output.c_str();

      // For output, we try to get its output size hint
      const std::string key = c10::str("output_shape_hint_", output_idx);
      auto output_shape_hint = this->template GetRepeatedArgument<int>(key);
      if (!output_shape_hint.empty()) {
        TensorInfo info;
        info.onnxifi_type = output_shape_hint.front();
        for (int i = 1; i < output_shape_hint.size(); ++i) {
          info.dims.push_back(output_shape_hint[i]);
        }
        output_shape_hints_.emplace(output_idx, std::move(info));
      }
      ++output_idx;
    }

    // Encode arguments starting with "custom_" to backend
    std::vector<uint64_t> property_pointers;
    std::vector<int64_t> int_args;
    std::vector<float> float_args;
    BuildPropertyList(operator_def, &property_pointers, &int_args, &float_args);

    // Pull the weights from workspace and feed it to the backend through
    // setGraphIO. Notice that since we may have rewritten the net, we need to
    // map the weight names
    auto initializers =
        this->template GetRepeatedArgument<std::string>("initializers");
    CAFFE_ENFORCE_EQ(
        initializers.size() % 2, 0, "initializers should come in pairs");
    std::unordered_set<std::string> initializer_set;
    std::unordered_map<std::string, std::string> input_mapping;
    for (auto it = initializers.begin(); it != initializers.end(); ++it) {
      auto key = *it++;
      input_mapping.emplace(key, *it);
      initializer_set.emplace(key);
    }
    Workspace mapped_ws(ws, input_mapping);
    std::vector<std::string> weight_names;
    std::vector<std::vector<uint64_t>> weight_shapes;
    auto weight_descs = BuildInitializationList(
        &mapped_ws, &initializer_set, &weight_names, &weight_shapes);

    // Build the Onnxifi engine
    // TODO: In spec, backends are hot-pluggable, so two calls to
    // onnxGetBackendIDs may result in different number of backend. And we
    // should retry until it get consistent. For now, we don't do that.
    CAFFE_ENFORCE_EQ(
        lib_->onnxGetBackendIDs(nullptr, &num_backends_),
        ONNXIFI_STATUS_FALLBACK);
    CAFFE_ENFORCE_GT(
        num_backends_, 0, "At least 1 onnxifi backend should be available");
    backend_ids_.resize(num_backends_);
    CAFFE_ENFORCE_EQ(
        lib_->onnxGetBackendIDs(backend_ids_.data(), &num_backends_),
        ONNXIFI_STATUS_SUCCESS);

    // TODO: choose backend id
    CAFFE_ENFORCE_EQ(
        lib_->onnxInitBackend(
            backend_ids_[0], property_pointers.data(), &backend_),
        ONNXIFI_STATUS_SUCCESS);
    CAFFE_ENFORCE_EQ(
        lib_->onnxInitGraph(
            backend_,
            nullptr,
            onnx_model_str.size(),
            (void*)(onnx_model_str.c_str()),
            weight_descs.size(),
            weight_descs.data(),
            &graph_),
        ONNXIFI_STATUS_SUCCESS);
  }

  ~OnnxifiOp() {
    if (graph_) {
      if (lib_->onnxReleaseGraph(graph_) != ONNXIFI_STATUS_SUCCESS) {
        LOG(ERROR) << "Error when calling onnxReleaseGraph";
      }
      graph_ = nullptr;
    }
    if (backend_) {
      if (lib_->onnxReleaseBackend(backend_) != ONNXIFI_STATUS_SUCCESS) {
        LOG(ERROR) << "Error when calling onnxReleaseBackend";
      }
      backend_ = nullptr;
    }
    for (unsigned i = 0; i < num_backends_; ++i) {
      if (lib_->onnxReleaseBackendID(backend_ids_[i]) != ONNXIFI_STATUS_SUCCESS) {
        LOG(ERROR) << "Error when calling onnxReleaseBackendID";
      }
    }
  }

  bool RunOnDevice() override;

 private:
  uint64_t SetOutputShapeAndType(int output_idx, std::vector<size_t>* dims) {
    uint64_t type = ONNXIFI_DATATYPE_FLOAT32;
    const auto it = output_shape_hints_.find(output_idx);
    if (it != output_shape_hints_.end()) {
      std::copy(
          it->second.dims.begin(),
          it->second.dims.end(),
          std::back_inserter(*dims));
      type = it->second.onnxifi_type;
    }
    return type;
  }

  void BuildPropertyList(
      const OperatorDef& /* unused */,
      std::vector<uint64_t>* property_list,
      std::vector<int64_t>* /* unused */,
      std::vector<float>* /* unused */) {
    property_list->push_back(ONNXIFI_BACKEND_PROPERTY_NONE);
  }

  std::vector<onnxTensorDescriptorV1> BuildInitializationList(
      Workspace* ws,
      std::unordered_set<std::string>* initialization_list,
      std::vector<std::string>* weight_names,
      std::vector<std::vector<uint64_t>>* weight_shapes);

  // pointer to loaded onnxifi library
  onnxifi_library* lib_{nullptr};

  std::vector<onnxBackendID> backend_ids_;
  onnxBackend backend_{nullptr};
  onnxGraph graph_{nullptr};
  size_t num_backends_{0};

  // input/output descriptors
  std::vector<onnxTensorDescriptorV1> input_desc_;
  std::vector<onnxTensorDescriptorV1> output_desc_;
  std::vector<std::vector<uint64_t>> input_shapes_;
  std::vector<std::vector<uint64_t>> output_shapes_;

  // output shape hints
  std::unordered_map<int, TensorInfo> output_shape_hints_;
};

} // namespace caffe2
