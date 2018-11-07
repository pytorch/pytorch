#pragma once

#include <fstream>

#include "caffe2/serialize/inline_container.h"
#include "caffe2/serialize/intermediate_model.h"

#include "torch/csrc/jit/import.h"
#include "torch/csrc/jit/export.h"

namespace torch {
namespace jit {

class ScriptModuleSerializer final {
 public:
  ScriptModuleSerializer(const std::string& filename) :
    ofs_(filename, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary),
    writer_(&ofs_) {
    // TODO appropriate support for mmap, right now we still use stream writer
  }

  ScriptModuleSerializer(std::istream* is) ofs_(), writer_(is) {}

  void serialize(const script::Module& smodule) {
    at::serialize::IntermediateModel imodel;
    convertToModel(smodule, &imodel);
    at::serialize::serializeIntermediateModel(imodel, &writer_);
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
      convertParameter(elem->value, iparam);
    }

    auto* imethods = imodule->mutableMethods();
    for (const auto& elem: smodule.get_methods()) {
      imethods->emplace_back();
      auto* imethod = &(imethods->back());
      convertMethod(elem->value, imethod);
    }

    auto* isubs = imodule->mutableSubmodules();
    for (const auto& elem: smodule.get_modules()) {
      isubs->emplace_back();
      auto* isub = &(isubs->back());
      convertModule(*(elem->value.module), elem->value.name, isub);
    }
  }

  void convertParameter(const script::NamedParameter& sparam, at::serialize::IntermediateParameter* iparam) {
    iparam->setName(sparam.name);
    iparam->setIsBuffer(sparam.is_buffer);
    iparam->setRequireGradient(sparam.slot()->requires_grad());
    convertTensor(*(sparam.slot()), iparam->mutableTensor());
  }

  void convertTensor(const at::Tensor& stensor,
      at::serialize::IntermediateTensor* itensor,
      std::unordered_map<void*, std::shared_ptr<at::serialize::SharedData>>* storage_map) {
    itensor->setDataType(ATenTypeToCaffe2Type(stensor.type().scalarType())); // TODO appropriate helper
    std::vector<int64_t> dims;
    for (auto d : stensor.sizes()) {
      dims.push_back(d);
    }
    itensor->setDims(dims);
    itensor->setOffset(offset);
    std::vector<int64_t> strides;
    for (auto s : stensor.strides()) {
      strides.push_back(s);
    }
    itensor->setStrides(strides);
    // TODO handle CUDA case
    auto* key = stensor.storage().unsafeGetStorageImpl();
    auto it = storage_map->find(key);
    if (it == storage_map.end()) {
      // TODO support CUDA, i.e., as CUDA tensor
      at::DataPtr data_ptr(stensor.storage().data(), DeviceType::CPU); // model still owns the data
      std::shared_ptr shared_data = std::make_unique<at::serialize::SharedData>(
          0, // record_id is unknown
          data_ptr,
          tensor.storage().size());
      storage_map[key] = shared_data;
      itensor->setData(shared_data);
    } else {
      itensor->setData(it->second);
    }
  }

  void convertMethod(const script::Method& smethod, at::serialize::IntermediateMethod* imethod) {
    // TODO encode the real torch script instead of ModelProto
    ::ONNX_NAMESPAC::ModelProto model_proto;
    ModuleEncoder encoder(smethod, model_proto.mutable_graph());
    string serialized_proto;
    model_proto.SerializeToString(&serialized_proto);
    // NB: it should be fine to use torch_script for now
    imethod->setTorchScript(serialized_proto);
  }

  at::serialize::PyTorchStreamWriter writer_;
  std::ofstream ofs_;

};

class ScriptModuleDeserializer {
 public:
  ScriptModuleDeserializer(const std::string& filename) :
    ifs(filename, std::ifstream::in | std::ifstream::binary), reader_(&ifs) {
    // TODO appropriate support for mmap, right now still use stream reader
  }

  ScriptModuleSerializer(std::istream* is) : ifs(), reader_(is) {}

  void deserialize(ModuleLookup module_lookup) {
    IntermediateModel imodel;
    at::serialize::deserializeIntermediateModel(&imodel, reader_);
    moduleLookup_ = module_lookup;
    std::shared_ptr<script::Module> module = moduleLook_(moduleStack_);
    convertModule(module.get());
  }

 private:
  void convertModule(const at::serialize::IntermediateModule& imodule,
      script::Module* smodule,
      std::unordered_map<uint64_t, std::shared_ptr<at::Storage>>* id_storage) {
    std::unordered_map<std::string, at::Tensor*> param_map;
    for (const auto& iparam: imodule.parameters()) {
      at::Tensor tensor = createTensor(iparam, reader, id_storage);
      autograd::Variable variable = autograd::make_variable(tensor,
          iparam.requireGradient());
      smodule->register_parameter(iparam.name(), variable, iparam.isBuffer());
      AT_ASSERT(param_map.find(iparam.name()) == param_map.end());
      param_map[iparam.name()] = &tensor;
    }

    for (const auto& imethod: imodule.methods()) {
      // TODO read unhacked torch script, right now it's serialized onnx proto
      ::ONNX_NAMESPACE::ModelProto method_proto;
      AT_ASSERTM(method_proto.ParseFromString(imethod.torchScript(),
            "cannot parse method proto (i.e., hacked onnx proto)");
      ModuleDecoder decoder(smodule, method_proto, param_map);
    }

    for (const auto& isub: imodule.submodules()) {
      moduleStack_.push(isub.name());
      std::shared_ptr<script::Module> ssub = moduleLook_(moduleStack_);
      convertModule(isub, ssub.get());
      moduleStack_.pop();
    }
  }

  at::Tensor createTensor(const at::serialize::IntermediateParameter& iparam,
      std::unordered_map<uint64_t, std::shared_ptr<at::Storage>>* id_storage) {
    const auto& tensor = iparam.tensor();
    auto data = tensor.data();
    auto type = CONVERT_UNIMPLEMENTED(tensor.DataType);
    auto storage_it = id_storage->find(data->recordId.value());
    if (storage_it != id_storage.end()) {
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
      id_storage->insert(std::make_pair(record_id, storage));
      return at::CPU(type)._th_tensor(*storage, tensor.offset(),
        tensor.dims(), tensor().strides());
    }
    return at::CPU(type)._th_tensor(*(storage_it->second.get()), tensor.offset(),
      tensor.dims(), tensor.stridems());
  }

  std::ifstream ifs_;
  PyTorchStreamReader reader_;
  ModuleLookup moduleLook_;
  std::stack<std::string> moduleStack_;

};

} // namespace jit
} // namespace torch
