#pragma once

#include <unordered_map>

#include "onnx/onnx_pb.h"

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/onnx/onnxifi_graph_info.h"
#include "caffe2/onnx/onnxifi_init.h"
#include "caffe2/utils/string_utils.h"

namespace caffe2 {

template <typename Context>
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
  explicit OnnxifiOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    lib_ = onnx::initOnnxifiLibrary();
    backend_graph_map_ptr_ = onnx::getOnnxBackendGraphMap();
    CAFFE_ENFORCE(lib_, "Cannot initialize ONNXIFI library");
    use_onnx_ = this->template GetSingleArgument<int>("use_onnx", 0);
    auto onnx_model_str =
        this->template GetSingleArgument<std::string>("onnx_model", "");
    CAFFE_ENFORCE(!onnx_model_str.empty(), "onnx_model cannot be empty");

    // Setup input/output descriptor templates
    input_names_ =
        this->template GetRepeatedArgument<std::string>("input_names");
    output_names_ =
        this->template GetRepeatedArgument<std::string>("output_names");
    CAFFE_ENFORCE_EQ(input_names_.size(), operator_def.input_size());
    CAFFE_ENFORCE_EQ(output_names_.size(), operator_def.output_size());
    for (const auto& input : input_names_) {
      input_desc_.push_back(onnxTensorDescriptorV1());
      input_desc_.back().name = input.c_str();
    }
    int output_idx = 0;
    for (const auto& output : output_names_) {
      output_desc_.push_back(onnxTensorDescriptorV1());
      output_desc_.back().name = output.c_str();

      // For output, we try to get its output size hint
      const std::string key = c10::str("output_shape_hint_", output_idx);
      auto output_shape_hint = this->template GetRepeatedArgument<int>(key);
      if (!output_shape_hint.empty()) {
        TensorInfo info;
        info.onnxifi_type = output_shape_hint.front();
        for (size_t i = 1; i < output_shape_hint.size(); ++i) {
          info.dims.push_back(output_shape_hint[i]);
        }
        output_shape_hints_.emplace(output_idx, std::move(info));
      }
      ++output_idx;
    }

    // Get output resizing hints
    adjust_output_batch_ =
        this->template GetSingleArgument<int>("adjust_output_batch", 0);
    auto output_resize_hints =
        this->template GetRepeatedArgument<int>("output_resize_hints");
    CAFFE_ENFORCE_EQ(
        output_resize_hints.size() % 2,
        0,
        "output_resize_hints must have even size: ",
        output_resize_hints.size());
    for (int i = 0; i < output_resize_hints.size(); ++i) {
      auto k = output_resize_hints[i++];
      batch_pos_map_.emplace(k, output_resize_hints[i]);
    }

    // Encode arguments starting with "custom_" to backend
    std::vector<uint64_t> property_pointers;
    std::vector<int64_t> int_args;
    std::vector<float> float_args;
    buildPropertyList(operator_def, &property_pointers, &int_args, &float_args);

    // Initialize the backend if it has not been already created. When we
    // initialized the backend, we will get the weights (initializers) from the
    // workspace and offload onto the backend. This should be done only once.
    // Subsequent call of this function with the same model id should find a
    // cached backend and therefore there is no need to repeat the above
    // process.
    buildBackendAndGraph(ws, property_pointers, onnx_model_str);
  }

  ~OnnxifiOp() {
    backend_graph_shared_ptr_.reset();
    backend_graph_map_ptr_->remove(op_id_string_);
#ifdef ONNXIFI_ENABLE_EXT
    traces_.reset();
#endif
  }

  bool RunOnDevice() override;

  void setEnableTracing(bool b) {
    enable_tracing_ = b;
  }

#ifdef ONNXIFI_ENABLE_EXT
  std::shared_ptr<onnxTraceEventList> traces() const {
    return traces_;
  }
#endif
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

  void buildPropertyList(
      const OperatorDef& /* unused */,
      std::vector<uint64_t>* property_list,
      std::vector<int64_t>* /* unused */,
      std::vector<float>* /* unused */) {
    property_list->push_back(ONNXIFI_BACKEND_PROPERTY_NONE);
  }

  void buildBackendAndGraph(
      Workspace* ws,
      const std::vector<uint64_t>& property_pointers,
      const std::string& onnx_model_str) {
    op_id_string_ =
        this->template GetSingleArgument<std::string>("model_id", "") + ":" +
        this->template GetSingleArgument<std::string>("net_pos", "");

    auto initializers =
        this->template GetRepeatedArgument<std::string>("initializers");
    // Build the Onnxifi engine
    auto backend_index = this->template GetSingleArgument<int>("backend_id", 0);
    auto creator = [this,
                    ws,
                    property_pointers,
                    backend_index,
                    &onnx_model_str,
                    &initializers]() {
      std::vector<onnxBackendID> backend_ids;
      size_t num_backends{0};
      CAFFE_ENFORCE_EQ(
          lib_->onnxGetBackendIDs(nullptr, &num_backends),
          ONNXIFI_STATUS_FALLBACK);
      CAFFE_ENFORCE_GT(
          num_backends, 0, "At least 1 onnxifi backend should be available");
      CAFFE_ENFORCE_LT(
          backend_index,
          num_backends,
          "Backend idx out of bound: ",
          backend_index,
          ", #backends: ",
          num_backends);
      backend_ids.resize(num_backends);
      CAFFE_ENFORCE_EQ(
          lib_->onnxGetBackendIDs(backend_ids.data(), &num_backends),
          ONNXIFI_STATUS_SUCCESS);

      onnxBackendID backend_id = backend_ids[backend_index];
      onnxBackend backend{nullptr};

      CAFFE_ENFORCE_EQ(
          lib_->onnxInitBackend(backend_id, property_pointers.data(), &backend),
          ONNXIFI_STATUS_SUCCESS);

      // Release unused backend ids.
      for (auto i = 0; i < num_backends; ++i) {
        if (i == backend_index) {
          continue;
        }
        lib_->onnxReleaseBackendID(backend_ids[i]);
      }

      // Get weights
      std::vector<std::string> weight_names;
      std::vector<std::vector<uint64_t>> weight_shapes;
      auto weight_descs = buildInitializationList(
          ws, initializers, &weight_names, &weight_shapes);

      onnxGraph graph{nullptr};
      CAFFE_ENFORCE_EQ(
          lib_->onnxInitGraph(
              backend,
              nullptr,
              onnx_model_str.size(),
              (const void*)(onnx_model_str.c_str()),
              weight_descs.size(),
              weight_descs.data(),
              &graph),
          ONNXIFI_STATUS_SUCCESS);

      return std::make_shared<onnx::BackendGraphInfo>(
          backend_id, backend, graph, lib_);
    };
    backend_graph_shared_ptr_ =
        backend_graph_map_ptr_->insert(op_id_string_, creator);

    backend_id_ = backend_graph_shared_ptr_->backend_id;
    backend_ = backend_graph_shared_ptr_->backend;
    graph_ = backend_graph_shared_ptr_->graph;

    getExtFunctionPointers();
  }

  /// Set up function pointer if onnxifi_ext is enabled
  void getExtFunctionPointers() {
#ifdef ONNXIFI_ENABLE_EXT
    onnxExtensionFunctionPointer p;
    if (lib_->onnxGetExtensionFunctionAddress(
            backend_id_, "onnxSetIOAndRunGraphFunction", &p) !=
        ONNXIFI_STATUS_SUCCESS) {
      onnxSetIOAndRunGraphPointer_ = nullptr;
    } else {
      onnxSetIOAndRunGraphPointer_ =
          reinterpret_cast<decltype(onnxSetIOAndRunGraphPointer_)>(p);
    }
    if (lib_->onnxGetExtensionFunctionAddress(
            backend_id_, "onnxReleaseTraceEventsFunction", &p) !=
        ONNXIFI_STATUS_SUCCESS) {
      onnxReleaseTraceEventsPointer_ = nullptr;
    } else {
      onnxReleaseTraceEventsPointer_ =
          reinterpret_cast<decltype(onnxReleaseTraceEventsPointer_)>(p);
    }
#endif
  }

  std::vector<int> extractOutputBatchSizes() const;

  void maybeAdjustOutputBatchSizes(
      const std::vector<int>& real_output_batch_sizes);

  std::vector<onnxTensorDescriptorV1> buildInitializationList(
      Workspace* ws,
      const std::vector<std::string>& initializers,
      std::vector<std::string>* weight_names,
      std::vector<std::vector<uint64_t>>* weight_shapes);

  // pointer to loaded onnxifi library
  onnxifi_library* lib_{nullptr};
  onnx::OnnxBackendGraphMap* backend_graph_map_ptr_;
  std::string op_id_string_;

  onnxBackendID backend_id_{nullptr};
  onnxBackend backend_{nullptr};
  onnxGraph graph_{nullptr};
  onnx::SharedPtrBackendGraphInfo backend_graph_shared_ptr_;

  // input/output descriptors
  std::vector<onnxTensorDescriptorV1> input_desc_;
  std::vector<onnxTensorDescriptorV1> output_desc_;

#ifdef ONNXIFI_ENABLE_EXT
  // onnxifi extension mode function pointer
  onnxStatus (*onnxSetIOAndRunGraphPointer_)(
      onnxGraph,
      uint32_t,
      const onnxTensorDescriptorV1*,
      uint32_t,
      const onnxTensorDescriptorV1*,
      onnxMemoryFenceV1*,
      onnxTraceEventList*);

  onnxStatus (*onnxReleaseTraceEventsPointer_)(onnxTraceEventList*);

  std::shared_ptr<onnxTraceEventList> traces_{nullptr};
#endif
  bool use_onnx_{false};

  // We bind the op input/output by position while ONNXIFI binds input/output by
  // names. In addition, op input/output names can be writtten by, for example,
  // memonger. We cache the original input/output name of ONNX object here and
  // bind them by position.
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  std::vector<std::vector<uint64_t>> input_shapes_;
  std::vector<std::vector<uint64_t>> output_shapes_;

  // output shape hints
  std::unordered_map<int, TensorInfo> output_shape_hints_;

  // Whether we need to resize outputs or not
  bool adjust_output_batch_{false};

  // Output resizing hint map
  // key: max batch size
  // value: position of the input where the real batch size can be extracted
  // from its first dimension
  std::unordered_map<int, int> batch_pos_map_;
  // Whether we enable tracing in one run of inference
  bool enable_tracing_{false};
};

} // namespace caffe2
