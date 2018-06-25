#pragma once

#include <unordered_map>

#include "onnx/onnx_pb.h"
#include "onnx/onnxifi.h"

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/onnx/onnxifi_manager.h"
#include "caffe2/utils/string_utils.h"

namespace caffe2 {

template <typename T, typename Context>
class OnnxifiOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  OnnxifiOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    onnxifi_backend_ =
        OperatorBase::GetSingleArgument<std::string>("onnxifi_backend", "");
    CAFFE_ENFORCE(!onnxifi_backend_.empty(), "Unspecified onnxifi_backend");
    auto* onnxifi_manager = onnx::OnnxifiManager::get_onnxifi_manager();
    lib_ = onnxifi_manager->AddOnnxifiLibrary(onnxifi_backend_);
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

    // Encode arguments starting with "custom_" to backend
    std::vector<uint64_t> property_pointers;
    std::vector<int64_t> int_args;
    std::vector<float> float_args;
    BuildPropertyList(operator_def, &property_pointers, &int_args, &float_args);

    // Pull the weights from workspace and feed it to the backend through
    // setGraphIO. Notice that since we may have rewritten the net, we need to
    // map the weight names
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
    std::vector<std::string> weight_names;
    std::vector<std::vector<uint64_t>> weight_shapes;
    auto weight_descs = BuildInitializationList(
        &mapped_ws, &initializer_set, &weight_names, &weight_shapes);

    ::ONNX_NAMESPACE::ModelProto onnx_model;
    ParseProtoFromLargeString(onnx_model_str, &onnx_model);
    onnx_model_str.clear();
    onnx_model.SerializeToString(&onnx_model_str);

    // Build the Onnxifi engine
    CAFFE_ENFORCE_EQ(
        lib_->onnxGetBackendIDs(backend_ids_, &num_backends_),
        ONNXIFI_STATUS_SUCCESS);
    CAFFE_ENFORCE_LT(
        num_backends_, 0, "At least 1 onnxifi backend should be available");
    // TODO: choose backedn id
    CAFFE_ENFORCE_EQ(
        lib_->onnxInitBackend(backend_ids_[0], property_pointers.data(), &backend_),
        ONNXIFI_STATUS_SUCCESS);
    CAFFE_ENFORCE_EQ(
        lib_->onnxInitGraph(
            backend_,
            onnx_model_str.size(),
            (void*)(onnx_model_str.c_str()),
            weight_descs.size(),
            weight_descs.data(),
            &graph_),
        ONNXIFI_STATUS_SUCCESS);
  }

  ~OnnxifiOp() {
    if (graph_) {
      CAFFE_ENFORCE_EQ(lib_->onnxReleaseGraph(graph_), ONNXIFI_STATUS_SUCCESS);
      graph_ = nullptr;
    }
    if (backend_) {
      CAFFE_ENFORCE_EQ(
          lib_->onnxReleaseBackend(backend_), ONNXIFI_STATUS_SUCCESS);
      backend_ = nullptr;
    }
    for (unsigned i = 0U; i < num_backends_; ++i) {
      CAFFE_ENFORCE_EQ(
          lib_->onnxReleaseBackendID(backend_ids_[i]), ONNXIFI_STATUS_SUCCESS);
    }
    backend_ids_ = nullptr;
  }

  bool RunOnDevice() override;

 private:
  void SetOutputShape(int output_idx, std::vector<TIndex>* dims) {
    const auto it = output_size_hints_.find(output_idx);
    if (it != output_size_hints_.end()) {
      *dims = it->second;
    }
  }

  void BuildPropertyList(
      const OperatorDef& operator_def,
      std::vector<uint64_t>* property_list,
      std::vector<int64_t>* int_args,
      std::vector<float>* float_args) {
    for (const auto& arg: operator_def.arg()) {
      if (!StartsWith(arg.name(), "custom_")) {
        continue;
      }
      // Pick the name as ABC if the arg name is custom_ABC
      property_list->push_back((uint64_t)(arg.name().c_str() + 7));

      if (arg.has_s()) {
        property_list->push_back((uint64_t)(arg.s().c_str()));
      } else if (arg.has_i()) {
        int_args->push_back(arg.i());
        property_list->push_back((uint64_t)(&int_args->back()));
      } else if (arg.has_f()) {
        float_args->push_back(arg.f());
        property_list->push_back((uint64_t)(&float_args->back()));
      } else if (arg.floats_size()) {
        property_list->push_back((uint64_t)(arg.floats().data()));
      } else if (arg.ints_size()) {
        property_list->push_back((uint64_t)(arg.ints().data()));
      } else {
        // TODO: support string lists
        CAFFE_THROW(
            "Cannot pass string list to onnxifi backend yet");
      }
    }
    property_list->push_back(ONNXIFI_BACKEND_PROPERTY_NONE);
  }

  std::vector<onnxTensorDescriptor> BuildInitializationList(
      Workspace* ws,
      std::unordered_set<std::string>* initialization_list,
      std::vector<std::string>* weight_names,
      std::vector<std::vector<uint64_t>>* weight_shapes);

  std::string onnxifi_backend_;
  onnxifi_library* lib_{nullptr};

  onnxBackendID* backend_ids_{nullptr};
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
