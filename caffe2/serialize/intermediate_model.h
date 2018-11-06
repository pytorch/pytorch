#pragma once

#include <unordered_map>
#include <stack>
#include <string>

#include <ATen/core/Allocator.h>
#include <c10/util/Optional.h>

#include "caffe2/core/common.h"
#include "caffe2/serialize/inline_container.h"

namespace caffe2 {
  class TensorProto;
  class NetDef;
}

namespace torch {
  class ParameterDef;
  class MethodDef;
  class ModuleDef;
  class ModelDef;
}

namespace at {
namespace serialize {

enum class DeserializeMode {
  // In LOADER_TENSOR_DATA mode, we load the file from the beginning, and eagarly load the content of tensors
  LOADER_TENSOR_DATA = 1,
  // In HEADER_ONLY mode, we only load the last record of the file, which is the model metadata
  // And the data of the tensor will be loaded later
  HEADER_ONLY = 2,
};

// multiple tensor may share the same content
// SharedData contains:
//    1) record id (i.e., the offset in the inline container)
//    2) size, the size of the content
//    3) data, in serialize, IntermediateModel does NOT own the data,
//       in deserialize, the data pointer is returned by PyTorchFileReader,
//       and IntermediateModel owns the data. The ownership later will be
//       transferred to Tensor
struct SharedData {
  // constructor
  explicit SharedData(uint64_t record_id, uint64_t size)
    : recordId(record_id), size(size), dataPtr() {}
  explicit SharedData(uint64_t record_id, uint64_t size, at::DataPtr&& data_ptr)
    : recordId(record_id), size(size), dataPtr(std::move(data_ptr)) {}

  c10::optional<uint64_t> recordId;
  uint64_t size = 0; // the size of the record
  at::DataPtr dataPtr;
};

// IntermediateDeviceOption stores device related information
struct IntermediateDeviceOption {
  int32_t deviceType = 0;
  c10::optional<int32_t> deviceId;
};

// IntermediateTensor contains
//   1) element type information
//   2) shape information
//   3) pointer to the data (including offset and strides)
class CAFFE2_API IntermediateTensor final {
 public:
  // constructor
  IntermediateTensor() = default;

  explicit IntermediateTensor(int64_t dataType, const std::vector<int64_t>& dims,
      int64_t offset): dataType_(dataType), dims_(dims), offset_(offset) {}

  // extract data from TensorProto, called in deserialize
  // assume record id to data mapping is complete
  void update(caffe2::TensorProto* tensor_proto,
      std::unordered_map<uint64_t, std::shared_ptr<SharedData>>* id_data,
      DeserializeMode mode);

  // update the data pointer, invoked in serialize
  void updateData(std::shared_ptr<SharedData> data) {
    data_ = data;
  }

  // dump data to TensorProto, called in serialize
  // assume the data is already saved
  void dump(caffe2::TensorProto* tensor_proto);

  // getters/setters
  int64_t dataType() const {
    return dataType_;
  }

  const std::vector<int64_t>& dims() const {
    return dims_;
  }

  std::shared_ptr<SharedData> data() const {
    return data_;
  }

  int64_t offset() const {
    return offset_;
  }

  const IntermediateDeviceOption& deviceDetail() const {
    return deviceDetail_;
  }

  const std::vector<int64_t>& strides() const {
    return strides_;
  }

  bool noContent() const {
    return noContent_;
  }


  void setData(std::shared_ptr<SharedData> data) {
    AT_ASSERTM(!noContent_, "noContent_ is true, but set content!");
    data_ = data;
  }

  void setStrides(const std::vector<int64_t>& strides) {
    strides_ = strides;
  }

  void setDeviceDetail(const IntermediateDeviceOption& device_detail) {
    deviceDetail_ = device_detail;
  }

 private:
  std::string name_;
  int64_t dataType_;
  std::vector<int64_t> dims_;
  int64_t offset_ = 0;
  std::vector<int64_t> strides_;
  // TODO: since we still have 2 different Tensor classes in Caffe2 and PyTorch
  // right now, let's just store the data pointer, and create Tensors
  // while converting IntermediateModel to the JitScriptModule/Predictor/etc.
  std::shared_ptr<SharedData> data_;
  IntermediateDeviceOption deviceDetail_;
  bool noContent_ = false;
};

class CAFFE2_API IntermediateParameter final {
 public:
  // constructors
  IntermediateParameter() = default;

  explicit IntermediateParameter(const std::string& name, bool is_buffer, bool require_gradient) :
    name_(name), isBuffer_(is_buffer), requireGradient_(require_gradient) {}

  explicit IntermediateParameter(torch::ParameterDef* param_def,
      std::unordered_map<uint64_t, std::shared_ptr<SharedData>>* id_data,
      DeserializeMode mode);

  // dump data to ParameterDef, invoked in serialize
  void dump(torch::ParameterDef* param_def);

  // getters/setters
  const std::string& name() const {
    return name_;
  }

  bool isBuffer() const {
    return isBuffer_;
  }

  bool requireGradient() const {
    return requireGradient_;
  }

  const IntermediateTensor& tensor() const {
    return tensor_;
  }

  IntermediateTensor* mutableTensor() {
    return &tensor_;
  }

  void setName(const std::string& name) {
    name_ = name;
  }

  void setIsBuffer(bool is_buffer) {
    isBuffer_ = is_buffer;
  }

  void setRequireGradient(bool require_gradient) {
    requireGradient_ = require_gradient;
  }

 private:
  std::string name_;
  bool isBuffer_ = false;
  bool requireGradient_ = false;
  IntermediateTensor tensor_;
};

class CAFFE2_API IntermediateMethod final {
 public:
  // constructors
  IntermediateMethod() = default;

  explicit IntermediateMethod(torch::MethodDef* method_def);
  explicit IntermediateMethod(IntermediateMethod&& method) noexcept;
  explicit IntermediateMethod(const IntermediateMethod& method) = delete;

  IntermediateMethod& operator =(IntermediateMethod&& method) noexcept;
  IntermediateMethod& operator =(const IntermediateMethod& method) = delete;

  // dump data to MethodDef, called in serialize
  void dump(torch::MethodDef* method_def);

  // getters
  const std::string& name() const {
    return name_;
  }

  const std::string& torchScript() const {
    return torchScript_;
  }

  // setters
  void setName(const std::string& name) {
    name_ = name;
  }

  void setTorchScript(const std::string& torch_script) {
    torchScript_ = torch_script;
  }

 private:
  std::string name_;
  std::unique_ptr<caffe2::NetDef> graph_;
  std::string torchScript_;
};

class CAFFE2_API IntermediateModule final {
 public:
  // constructors
  IntermediateModule() = default;

  explicit IntermediateModule(torch::ModuleDef* module_def,
      std::unordered_map<uint64_t, std::shared_ptr<SharedData>>* id_data,
      DeserializeMode mode);

  // extract data from ModuleDef, invoked in deserialize
  // assume record id to data mapping is complete
  void update(torch::ModuleDef* module_def,
      std::unordered_map<uint64_t, std::shared_ptr<SharedData>>* id_data,
      DeserializeMode mode);

  // dump data to ModuleDef, called in serialize
  void dump(torch::ModuleDef* module_def);

  // getters/setters
  const std::string& name() const {
    return name_;
  }

  std::vector<IntermediateParameter>* mutableParameters() {
    return &parameters_;
  }

  std::vector<IntermediateModule>* mutableSubmodules() {
    return &submodules_;
  }

  std::vector<IntermediateMethod>* mutableMethods() {
    return &methods_;
  }

  const std::vector<IntermediateParameter>& parameters() const {
    return parameters_;
  }

  const std::vector<IntermediateModule>& submodules() const {
    return submodules_;
  }

  const std::vector<IntermediateMethod>& methods() const {
    return methods_;
  }

  void setName(const std::string& name) {
    name_ = name;
  }

 private:
  std::string name_;
  std::vector<IntermediateParameter> parameters_;
  std::vector<IntermediateModule> submodules_;
  std::vector<IntermediateMethod> methods_;
  // TODO handle cpp_arena
  // TODO handle pickle_arena
};

class CAFFE2_API IntermediateModel final {
 public:
  // constructor
  IntermediateModel() = default;

  // extract data from ModelDef, invoked in deserialize
  // assume record id to data mapping is complete
  void update(torch::ModelDef* model_def,
      std::unordered_map<uint64_t, std::shared_ptr<SharedData>>* id_data,
      DeserializeMode mode);

  // dump data to ModelDef, called in serialize
  void dump(torch::ModelDef* model_def);

  // getters
  const std::string& name() const {
    return name_;
  }

  const std::string& producerName() const {
    return producerName_;
  }

  const std::string& producerVersion() const {
    return producerVersion_;
  }

  const IntermediateModule& mainModule() const {
    return mainModule_;
  }

  int64_t protoVersion() const {
    return protoVersion_;
  }

  // setters, most for test purposes
  void setName(const std::string& name) {
    name_ = name;
  }

  void setProducerName(const std::string& producer_name) {
    producerName_ = producer_name;
  }

  void setProducerVersion(const std::string& producer_version) {
    producerVersion_ = producer_version;
  }

  void setProtoVersion(int64_t proto_version) {
    protoVersion_ = proto_version;
  }

  IntermediateModule* mutableMainModule() {
    return &mainModule_;
  }

 private:
  std::string name_;
  std::string producerName_;
  std::string producerVersion_;
  int64_t protoVersion_;
  IntermediateModule mainModule_;

};

// serialize an IntermediateModel through a PyTorchFileWriter
// we always put the model data at the end, so when serializing
// model, the we assume the record_id in imodel is already updated
CAFFE2_API void serializeIntermediateModel(IntermediateModel* imodel,
    torch::jit::PyTorchFileWriter* writer);

// serialize an IntermediateModel to a given file
CAFFE2_API void serializeIntermediateModel(IntermediateModel* imodel, const std::string& filename);

// deserialize an IntermediateModel through a reader,
// serialize tensors' data first, and maintain the mappint from
// record id to tensor data
CAFFE2_API void deserializeIntermediateModel(IntermediateModel* imodel,
    torch::jit::PyTorchFileReader* reader, DeserializeMode mode=DeserializeMode::LOADER_TENSOR_DATA);

// deserialize an IntermediateModel from a given file
CAFFE2_API void deserializeIntermediateModel(IntermediateModel* imodel, const std::string& filename,
    DeserializeMode mode=DeserializeMode::LOADER_TENSOR_DATA);

}  // namespace serialize
}  // namespace at
