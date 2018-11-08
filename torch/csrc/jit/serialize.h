#pragma once

#include <fstream>

#include "c10/util/Exception.h"
#include "onnx/onnx_pb.h"
#include "torch/csrc/onnx/onnx.h"
#include "caffe2/serialize/inline_container.h"
#include "caffe2/serialize/intermediate_model.h"
#include "torch/csrc/jit/import.h"
#include "torch/csrc/jit/export.h"

namespace torch {
namespace jit {
namespace serialize {

namespace onnx_torch = ::torch::onnx;
namespace onnx = ::ONNX_NAMESPACE;

class EncoderBase {
 public:
  EncoderBase(onnx_torch::OperatorExportTypes operator_export_type, bool strip_doc);

  onnx::ModelProto get_model_proto() {
    return model_proto_;
  }

 protected:
  void EncodeGraph(onnx::GraphProto *graph_proto,
                   const std::shared_ptr<Graph> &graph,
                   const std::vector<at::Tensor> &initializers = {});

  void EncodeBlock(onnx::GraphProto *graph_proto,
                   const Block *block,
                   const std::vector<at::Tensor> &initializers = {});

  virtual void EncodeTensor(
      onnx::TensorProto* tensor_proto,
      const at::Tensor& tensor,
      const c10::optional<std::string> external_ref = {}) = 0;

  virtual void EncodeIntermediateValueInfo(onnx::GraphProto *graph_proto,
                                           const Value* n) {};

  virtual void EncodeValueInfo(onnx::GraphProto *graph_proto,
                               onnx::ValueInfoProto* v,
                               const Value* n);

  void AddAttribute(onnx::NodeProto *node_proto, const jit::Node *node, const jit::Symbol name);

  onnx::ModelProto model_proto_;
  size_t num_blocks_;
  onnx_torch::OperatorExportTypes operator_export_type_;
  bool strip_doc_;
};

class ModuleEncoder: public EncoderBase {
 public:
  ModuleEncoder(const script::Module &module,
                std::ostream& out);
  ModuleEncoder(const script::Method& method, onnx::GraphProto* graph_proto);

 private:
  void EncodeModule(onnx::GraphProto *graph_proto, const script::Module &module);

  void EncodeParameters(onnx::GraphProto *graph_proto,
                        const script::Module &module,
                        const std::string prefix);

  void EncodeParameter(onnx::TensorProto *tensor_proto,
                       const script::NamedParameter &parameter,
                       const std::string prefix);

  void EncodeMethods(onnx::GraphProto *graph_proto,
                     const script::Module &module,
                     const std::string prefix);

  void EncodeMethod(onnx::NodeProto *node_proto,
                    script::Method &method,
                    const std::string prefix);

  virtual void EncodeTensor(
      onnx::TensorProto* tensor_proto,
      const at::Tensor& tensor,
      const c10::optional<std::string> external_ref = {}) override;

  virtual void EncodeIntermediateValueInfo(onnx::GraphProto *graph_proto,
                                           const Value* n) override;

  virtual void EncodeValueInfo(onnx::GraphProto *graph_proto,
                               onnx::ValueInfoProto* v,
                               const Value* n) override;

  void EncodeTypeInfo(onnx::GraphProto *graph_proto,
                      onnx::ValueInfoProto* v,
                      const TypePtr& type,
                      const std::string& name);

  PyTorchStreamWriter stream_writer_;
  // Used to deduplicate tensor storages
  std::unordered_map<const void*, uint64_t> storage_dedup_map_;

  // Used to keep track of Parameter names so Methods can refer to them
  std::unordered_map<at::Tensor*, std::string> parameter_map_;

  // Used to create sequential dummy names for node types
  size_t type_counter_ = 0;
};

class ModuleDecoder {
 public:
  ModuleDecoder(ModuleLookup module_lookup,
                std::istream& in);
  ModuleDecoder(const onnx::ModelProto& model_proto,
      const std::unordered_map<std::string, at::Tensor*>& param_map,
      script::Module* parent_module);

 private:
  std::shared_ptr<Graph> buildGraph(const onnx::GraphProto& graph_proto);

  void buildBlock(const onnx::GraphProto& graph_proto, Block* block,
                  std::unordered_map<std::string, Value*>& value_map);

  void buildBlocks(const std::vector<onnx::GraphProto>& graphs_, Node* node,
                   std::unordered_map<std::string, Value*>& value_map);

  void buildValue(Value* value, const onnx::ValueInfoProto& valueinfo_proto);

  void buildIntermediateValue(Value* value, const std::string& name);

  at::ScalarType onnxTypeToATenType(onnx::TensorProto_DataType tensor_proto);

  at::Tensor buildTensor(const onnx::TensorProto& tensor_proto);

  TypePtr buildType(const onnx::TypeProto& type_proto);

  at::Tensor buildParameter(const onnx::TensorProto& tensor_proto);

  at::Tensor buildTensorCommon(const onnx::TensorProto& tensor_proto,
                               const uint64_t record_number,
                               const int64_t storage_offset,
                               const std::vector<int64_t>& strides);

  std::pair<std::shared_ptr<script::Module>, std::string> parseFullName(
      ModuleLookup module_lookup,
      const std::string fullname);

  PyTorchStreamReader stream_reader_;
  std::unordered_map<uint64_t, std::shared_ptr<at::Storage>> storage_map_;
  std::unordered_map<std::string, const onnx::TypeProto*> value_type_map_;
};


class ScriptModuleSerializer final {
 public:
  ScriptModuleSerializer(const std::string& filename) :
    ofs_(filename, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary),
    writer_(&ofs_) {
    // TODO appropriate support for mmap, right now we still use stream writer
  }

  ScriptModuleSerializer(std::ostream* ofs) : ofs_(), writer_(ofs) {}

  void serialize(const script::Module& smodule) {
    at::serialize::IntermediateModel imodel;
    convertToModel(smodule, &imodel);
    at::serialize::serializeIntermediateModel(&imodel, &writer_);
  }

 private:
  void convertToModel(const script::Module& smodule, at::serialize::IntermediateModel* imodel) {
    imodel->setName("script-model");
    imodel->setProducerName("pytorch");
    imodel->setProducerVersion("1.0"); // TODO: set the producer version using appropriate function call
    // TODO set proto version
    convertModule(smodule, "main-module", imodel->mutableMainModule());
  }

  void convertModule(const script::Module& smodule, const std::string& name,
      at::serialize::IntermediateModule* imodule) {
    imodule->setName(name);

    auto* iparams = imodule->mutableParameters();
    for (const auto& elem: smodule.get_parameters()) {
      iparams->emplace_back();
      auto* iparam = &(iparams->back());
      convertParameter(elem.value, iparam);
    }

    auto* imethods = imodule->mutableMethods();
    for (const auto& elem: smodule.get_methods()) {
      imethods->emplace_back();
      auto* imethod = &(imethods->back());
      convertMethod(*(elem.value), imethod);
    }

    auto* isubs = imodule->mutableSubmodules();
    for (const auto& elem: smodule.get_modules()) {
      isubs->emplace_back();
      auto* isub = &(isubs->back());
      convertModule(*(elem.value.module), elem.value.name, isub);
    }
  }

  void convertParameter(const script::NamedParameter& sparam, at::serialize::IntermediateParameter* iparam) {
    iparam->setName(sparam.name);
    iparam->setIsBuffer(sparam.is_buffer);
    iparam->setRequireGradient(sparam.slot()->requires_grad());
    convertTensor(*(sparam.slot()), iparam->mutableTensor());
  }

  void convertTensor(const at::Tensor& stensor,
      at::serialize::IntermediateTensor* itensor) {
    itensor->setDataType(at::serialize::atenTypeToIModelType(stensor.type().scalarType()));
    std::vector<int64_t> dims;
    for (auto d : stensor.sizes()) {
      dims.push_back(d);
    }
    itensor->setDims(dims);
    itensor->setOffset(stensor.storage_offset());
    std::vector<int64_t> strides;
    for (auto s : stensor.strides()) {
      strides.push_back(s);
    }
    itensor->setStrides(strides);
    // TODO handle CUDA case
    auto* key = stensor.storage().unsafeGetStorageImpl();
    auto it = storageMap_.find(key);
    if (it == storageMap_.end()) {
      // TODO support CUDA, i.e., as CUDA tensor
      at::DataPtr data_ptr(stensor.storage().data(), at::DeviceType::CPU); // model still owns the data
      std::shared_ptr<at::serialize::SharedData> shared_data = std::make_shared<at::serialize::SharedData>(
          0, // record_id is unknown
          data_ptr,
          stensor.storage().size());
      storageMap_[key] = shared_data;
      itensor->setData(shared_data);
    } else {
      itensor->setData(it->second);
    }
  }

  void convertMethod(const script::Method& smethod, at::serialize::IntermediateMethod* imethod) {
    // TODO encode the real torch script instead of ModelProto
    ::ONNX_NAMESPACE::ModelProto model_proto;
    ModuleEncoder encoder(smethod, model_proto.mutable_graph());
    std::string serialized_proto;
    model_proto.SerializeToString(&serialized_proto);
    // NB: it should be fine to use torch_script for now
    imethod->setTorchScript(serialized_proto);
  }

  std::unordered_map<void*, std::shared_ptr<at::serialize::SharedData>> storageMap_;
  std::ofstream ofs_;
  PyTorchStreamWriter writer_;

};

class ScriptModuleDeserializer final {
 public:
  ScriptModuleDeserializer(const std::string& filename) :
    ifs_(filename, std::ifstream::in | std::ifstream::binary), reader_(&ifs_) {
    // TODO appropriate support for mmap, right now still use stream reader
  }

  ScriptModuleDeserializer(std::istream* is) : ifs_(), reader_(is) {}

  void deserialize(ModuleLookup module_lookup) {
    at::serialize::IntermediateModel imodel;
    at::serialize::deserializeIntermediateModel(&imodel, &reader_);
    moduleLookup_ = module_lookup;
    std::shared_ptr<script::Module> module = moduleLookup_(moduleStack_);
    convertModule(imodel.mainModule(), module.get());
  }

 private:
  void convertModule(const at::serialize::IntermediateModule& imodule,
      script::Module* smodule) {
    std::unordered_map<std::string, at::Tensor*> param_map;
    for (const auto& iparam: imodule.parameters()) {
      at::Tensor tensor = createTensor(iparam);
      autograd::Variable variable = autograd::make_variable(tensor,
          iparam.requireGradient());
      smodule->register_parameter(iparam.name(), variable, iparam.isBuffer());
      AT_ASSERT(param_map.find(iparam.name()) == param_map.end());
      param_map[iparam.name()] = &tensor;
    }

    for (const auto& imethod: imodule.methods()) {
      // TODO read unhacked torch script, right now it's serialized onnx proto
      ::ONNX_NAMESPACE::ModelProto method_proto;
      AT_ASSERTM(method_proto.ParseFromString(imethod.torchScript()),
            "cannot parse method proto (i.e., hacked onnx proto)");
      ModuleDecoder decoder(method_proto, param_map, smodule);
    }

    for (const auto& isub: imodule.submodules()) {
      moduleStack_.push_back(isub.name());
      std::shared_ptr<script::Module> ssub = moduleLookup_(moduleStack_);
      convertModule(isub, ssub.get());
      moduleStack_.pop_back();
    }
  }

  at::Tensor createTensor(const at::serialize::IntermediateParameter& iparam) {
    const auto& itensor = iparam.tensor();
    auto data = itensor.data();
    auto type = at::serialize::iModelTypeToATenType(itensor.dataType());
    auto storage_it = storageMap_.find(data->recordId.value());
    if (storage_it != storageMap_.end()) {
      at::DataPtr storage_ptr = data->data;
      uint64_t size = data->size;
      AT_ASSERT(data->recordId.has_value());
      uint64_t record_id = data->recordId.value();
      if (storage_ptr->get() == nullptr) {
        // the tensor data is NOT loaded yet
        AT_ASSERT(reader_ != nullptr);
        std::tie(storage_ptr, size) = reader_.getRecordWithKey(record_id);
      }
      auto storage = std::make_shared<at::Storage>(
          at::CPU(type).typeMeta(),
          std::move(storage_ptr),
          size / at::CPU(type).typeMeta().itemsize(),
          nullptr); // NB: we didn't set any allocator for the tensor
      storageMap_.insert(std::make_pair(record_id, storage));
      return at::CPU(type)._th_tensor(*storage, itensor.offset(),
        itensor.dims(), itensor().strides());
    }
    return at::CPU(type)._th_tensor(*(storage_it->second.get()), itensor.offset(),
      itensor.dims(), itensor.stridems());
  }

  std::ifstream ifs_;
  PyTorchStreamReader reader_;
  ModuleLookup moduleLookup_;
  std::vector<std::string> moduleStack_;
  std::unordered_map<std::string, std::shared_ptr<at::Storage>> storageMap_;

};

} // namespace serialize
} // namespace jit
} // namespace torch
