#pragma once

#include <unordered_map>

#include "onnx/onnx_pb.h"

#include "c10/util/SmallVector.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/onnx/onnxifi_graph_info.h"
#include "caffe2/onnx/onnxifi_init.h"
#include "caffe2/opt/shape_info.h"
#include "caffe2/utils/proto_utils.h"
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

  struct OutputReshapeInfo {
    std::vector<Tensor> begins;
    std::vector<Tensor> ends;
    std::vector<bool> fast_path;
    bool skip{false};
  };

 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  explicit OnnxifiOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        use_onnx_(this->template GetSingleArgument<int>("use_onnx", 0)),
        max_batch_size_(
            this->template GetSingleArgument<int>("max_batch_size", 0)),
        max_seq_size_(this->template GetSingleArgument<int>("max_seq_size", 0)),
        nominal_batch_idx_(
            this->template GetSingleArgument<int>("nominal_batch_idx", 0)) {
    lib_ = onnx::initOnnxifiLibrary();
    backend_graph_map_ptr_ = onnx::getOnnxBackendGraphMap();
    CAFFE_ENFORCE(lib_, "Cannot initialize ONNXIFI library");
    auto onnx_model_str =
        this->template GetSingleArgument<std::string>("onnx_model", "");
    CAFFE_ENFORCE(!onnx_model_str.empty(), "onnx_model cannot be empty");
    if (!use_onnx_) {
      CAFFE_ENFORCE(ParseProtoFromLargeString(onnx_model_str, &netdef_));
    }

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
    all_offsets_.reserve(ws->Blobs().size());
    all_scales_.reserve(ws->Blobs().size());
    input_shapes_.resize(input_names_.size());
    output_shapes_.resize(output_names_.size());
    output_reshape_info_.begins.reserve(output_names_.size());
    output_reshape_info_.ends.reserve(output_names_.size());
    output_reshape_info_.fast_path.reserve(output_names_.size());
    int output_idx = 0;
    for (const auto& output : output_names_) {
      output_desc_.push_back(onnxTensorDescriptorV1());
      output_desc_.back().name = output.c_str();

      // For output, we try to get its output size hint
      int64_t num_dims = 0;
      const std::string key = c10::str("output_shape_hint_", output_idx);
      auto output_shape_hint = this->template GetRepeatedArgument<int>(key);
      if (!output_shape_hint.empty()) {
        TensorInfo info;
        info.onnxifi_type = output_shape_hint.front();
        for (size_t i = 1; i < output_shape_hint.size(); ++i) {
          info.dims.push_back(output_shape_hint[i]);
        }
        num_dims = info.dims.size();
        output_shape_hints_.emplace(output_idx, std::move(info));
      }

      // Initialize the tensors used to slice the output
      output_reshape_info_.begins.emplace_back();
      ReinitializeTensor(
          &output_reshape_info_.begins.back(),
          {num_dims},
          at::dtype<int32_t>().device(CPU));
      output_reshape_info_.ends.emplace_back();
      ReinitializeTensor(
          &output_reshape_info_.ends.back(),
          {num_dims},
          at::dtype<int32_t>().device(CPU));
      output_reshape_info_.fast_path.push_back(false);
      ++output_idx;
    }

    // Get output resizing hints
    adjust_output_batch_ =
        this->template GetSingleArgument<int>("adjust_output_batch", 0);

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
      for (size_t i = 0; i < num_backends; ++i) {
        if (i == backend_index) {
          continue;
        }
        lib_->onnxReleaseBackendID(backend_ids[i]);
      }

      // Get weights
      std::vector<std::string> weight_names;
      std::vector<std::vector<uint64_t>> weight_shapes;
      auto weight_descs = buildInitializationList(
          ws,
          initializers,
          &weight_names,
          &weight_shapes,
          &all_scales_,
          &all_offsets_);

      // Extra weight shapes
      std::unordered_map<std::string, ShapeInfo> weight_shape_info;
      for (size_t i = 0; i < weight_names.size(); ++i) {
        TensorShape shape;
        const auto& shape0 = weight_shapes[i];
        for (const auto d : shape0) {
          shape.add_dims(d);
        }
        weight_shape_info[weight_names[i]] = ShapeInfo(
            std::vector<TensorBoundShape::DimType>(
                shape0.size(), TensorBoundShape_DimType_CONSTANT),
            std::move(shape));
      }

      Blob* defered_blob_reader = nullptr;
      if (ws->HasBlob("__DEFERRED_BLOB_READER__")) {
        defered_blob_reader = ws->GetBlob("__DEFERRED_BLOB_READER__");
      }
      onnxGraph graph{nullptr};
      CAFFE_ENFORCE_EQ(
          lib_->onnxInitGraph(
              backend,
              nullptr,
              onnx_model_str.size(),
              (const void*)(onnx_model_str.c_str()),
              weight_descs.size(),
              weight_descs.data(),
              &graph,
              defered_blob_reader),
          ONNXIFI_STATUS_SUCCESS);

      return std::make_shared<onnx::BackendGraphInfo>(
          backend_id, backend, graph, lib_, std::move(weight_shape_info));
    };
    backend_graph_shared_ptr_ =
        backend_graph_map_ptr_->insert(op_id_string_, creator);

    backend_id_ = backend_graph_shared_ptr_->backend_id;
    backend_ = backend_graph_shared_ptr_->backend;
    graph_ = backend_graph_shared_ptr_->graph;
    input_shape_info_ = backend_graph_shared_ptr_->weight_shape_info;

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

  void extractOutputBatchSizes();

  // If needed, adjust output tensor shape based on the real input batch size.
  // If the output shape is conditioned on first dim (batch size), we have a
  // fast path to shrink the tensor shape by just manipulating the meta data.
  // Otherwise, we have to slice it in the middle of the dimension with copy
  // invoked. This is a slow path and we don't expect it to happen very often.
  // We can already omit this step by setting "adjust_output_batch_" to false
  void maybeAdjustOutputBatchSizes();

  std::vector<onnxTensorDescriptorV1> buildInitializationList(
      Workspace* ws,
      const std::vector<std::string>& initializers,
      std::vector<std::string>* weight_names,
      std::vector<std::vector<uint64_t>>* weight_shapes,
      std::vector<std::vector<float>>* all_scales,
      std::vector<std::vector<int32_t>>* all_offsets) const;

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

  // Output reshape info
  OutputReshapeInfo output_reshape_info_;

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

  // ONNX model or not
  bool use_onnx_{false};

  // max batch size
  int max_batch_size_;

  // max sequence lookup size
  int max_seq_size_;

  // index of the input whose first dimension represents the batch size
  int nominal_batch_idx_{0};

  // We bind the op input/output by position while ONNXIFI binds input/output by
  // names. In addition, op input/output names can be writtten by, for example,
  // memonger. We cache the original input/output name of ONNX object here and
  // bind them by position.
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  // NetDef of the onnxifi subgraph for shape inference
  NetDef netdef_;

  std::vector<c10::SmallVector<uint64_t, 4>> input_shapes_;
  std::vector<c10::SmallVector<uint64_t, 4>> output_shapes_;

  // A cache vector to avoid repeated reallocation. The existence of this is not
  // ideal, which is purely due to the factor that we use int64_t for c2::tensor
  // dim but uint64_t for onnxDesciptor dim. Maybe we should just use int64_t
  c10::SmallVector<int64_t, 4> tensor_dims_int64_;

  // This is for multi group quantization info
  std::vector<std::vector<float>> all_scales_;
  std::vector<std::vector<int32_t>> all_offsets_;

  // output shape hints
  std::unordered_map<int, TensorInfo> output_shape_hints_;

  // input shape info. Used by shape inference when inputs are not at
  // max_batch_size
  std::unordered_map<std::string, ShapeInfo> input_shape_info_;

  // Whether we need to resize outputs or not
  bool adjust_output_batch_{false};

  // Whether we enable tracing in one run of inference
  bool enable_tracing_{false};
};

} // namespace caffe2
