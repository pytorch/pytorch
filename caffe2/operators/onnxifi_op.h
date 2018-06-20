#pragma once

#include <unordered_map>

#include "onnx/onnx_pb.h"
#include "onnx/onnxifi.h"

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/onnx/onnxifi_manager.h"

namespace caffe2 {

template <typename T, typename Context>
class OnnxifiOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  OnnxifiOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    onnxifi_backend_ =
        OperatorBase::GetSingleArgument<std::string>("onnxifi_backend", "");
    location_ = OperatorBase::GetSingleArgument<std::string>(
        "onnxifi_backend_path", "");
    backend_idx_ =
        OperatorBase::GetSingleArgument<int>("onnxifi_backend_idx", 0);
    auto suffix = OperatorBase::GetSingleArgument<std::string>(
        "onnxifi_backend_suffix", "");
    CAFFE_ENFORCE(!onnxifi_backend_.empty(), "Unspecified onnxifi_backend");
    CAFFE_ENFORCE(!location_.empty(), "Unspecified onnxifi_backend_path");
    auto* onnxifi_manager = onnx::OnnxifiManager::get_onnxifi_manager();
    lib_ = onnxifi_manager->AddOnnxifiLibrary(onnxifi_backend_, location_, suffix);
    auto onnx_model_str =
        OperatorBase::GetSingleArgument<std::string>("onnx_model", "");
    CAFFE_ENFORCE(!onnx_model_str.empty(), "onnx_model cannot be empty");

    // Setup input/output descriptor templates
    for (const auto& input : operator_def.input()) {
      input_desc_.push_back(onnxTensorDescriptor());
      input_desc_.back().name = input.c_str();
    }
    int output_idx = 0;
    for (const auto& output : operator_def.output()) {
      output_desc_.push_back(onnxTensorDescriptor());
      output_desc_.back().name = output.c_str();

      // For output, we try to get its output size hint
      const std::string key = MakeString("output_size_hint_", output_idx);
      auto output_size_hint = OperatorBase::GetRepeatedArgument<int>(key);
      if (!output_size_hint.empty()) {
        std::vector<TIndex> dims;
        for (const auto v : output_size_hint) {
          dims.push_back(v);
        }
        output_size_hints_.emplace(output_idx, std::move(dims));
      }
      ++output_idx;
    }

    // TODO: Encode the rest of the arguments to backend

    // Pull the weights from workspace and assembly it back to the onnx model,
    // notice that since we may have rewritten the net, we need to map the
    // weight names
    // TODO: this step can be avoided by initGraphIO
    auto initializers =
        OperatorBase::GetRepeatedArgument<std::string>("initializers");
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
    ::ONNX_NAMESPACE::ModelProto onnx_model;
    ParseProtoFromLargeString(onnx_model_str, &onnx_model);
    BuildInitializationList(
        &mapped_ws, onnx_model.mutable_graph(), &initializer_set);
    onnx_model_str.clear();
    onnx_model.SerializeToString(&onnx_model_str);

    // Build the Onnxifi engine
    CAFFE_ENFORCE_EQ(
        lib_->onnxGetBackendIDs(&backend_id_, &num_backends_),
        ONNXIFI_STATUS_SUCCESS);
    CAFFE_ENFORCE_GT(num_backends_, backend_idx_);
    // TODO: feed encoded parameter list to backend
    CAFFE_ENFORCE_EQ(
        lib_->onnxInitBackend(backend_id_, NULL, &backend_),
        ONNXIFI_STATUS_SUCCESS);
    CAFFE_ENFORCE_EQ(
        lib_->onnxInitGraph(
            backend_,
            onnx_model_str.size(),
            (void*)(onnx_model_str.c_str()),
            0,
            NULL,
            &graph_),
        ONNXIFI_STATUS_SUCCESS);
  }

  ~OnnxifiOp() {
    if (backend_id_ && !num_backends_) {
      CAFFE_ENFORCE_EQ(
          lib_->onnxReleaseBackendID(backend_id_), ONNXIFI_STATUS_SUCCESS);
    }
    if (graph_) {
      CAFFE_ENFORCE_EQ(lib_->onnxReleaseGraph(graph_), ONNXIFI_STATUS_SUCCESS);
    }
    if (backend_) {
      CAFFE_ENFORCE_EQ(
          lib_->onnxReleaseBackend(backend_), ONNXIFI_STATUS_SUCCESS);
    }
  }

  bool RunOnDevice() override;

 private:
  void SetOutputShape(int output_idx, std::vector<TIndex>* dims) {
    const auto it = output_size_hints_.find(output_idx);
    if (it != output_size_hints_.end()) {
      *dims = it->second;
    }
  }

  void BuildInitializationList(
      Workspace* ws,
      ::ONNX_NAMESPACE::GraphProto* g,
      std::unordered_set<std::string>* initialization_list);

  std::string onnxifi_backend_;
  std::string location_;
  int backend_idx_{0};
  onnxifi_library* lib_{nullptr};

  onnxBackendID backend_id_{nullptr};
  onnxBackend backend_{nullptr};
  onnxGraph graph_{nullptr};
  size_t num_backends_{0};

  // input/output descriptors
  std::vector<onnxTensorDescriptor> input_desc_;
  std::vector<onnxTensorDescriptor> output_desc_;
  std::vector<std::vector<uint64_t>> input_shapes_;
  std::vector<std::vector<uint64_t>> output_shapes_;

  // output shape hints
  std::unordered_map<int, std::vector<TIndex>> output_size_hints_;
};

} // namespace caffe2
