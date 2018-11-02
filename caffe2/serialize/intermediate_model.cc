#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/proto/torch_pb.h"

#include "caffe2/serialize/intermediate_model.h"

namespace at {
namespace serialize{

void IntermediateTensor::update(caffe2::TensorProto* tensor_proto,
    std::unordered_map<uint64_t, std::shared_ptr<SharedData>>* id_data,
    DeserializeMode mode) {
  AT_ASSERTM(tensor_proto->has_data_type(), "no data_type in TensorProto!");
  dataType_ = tensor_proto->data_type();
  for (int i = 0; i < tensor_proto->dims_size(); ++i) {
    dims_.push_back(tensor_proto->dims(i));
  }
  if (tensor_proto->has_name()) {
    // TODO: TensorProto's name is not used, we just keep it here for now
    // later we will deprecate it.
    name_ = tensor_proto->name();
  }
  if (tensor_proto->has_device_detail()) {
    const auto& device_detail = tensor_proto->device_detail();
    deviceDetail_.deviceType = device_detail.device_type();
    if (device_detail.has_device_id()) {
      deviceDetail_.deviceId = device_detail.device_id();
    }
    if (device_detail.has_random_seed()) {
      AT_ERROR("DeviceOption contains random seed, not supported!");
    }
    if (device_detail.has_node_name()) {
      AT_ERROR("DeviceOption contains node name, not supported!");
    }
    if (device_detail.extra_info_size() > 0) {
      AT_ERROR("DeviceOption contains extra info, not supported!");
    }
  }
  AT_ASSERTM(tensor_proto->has_storage_type(), "no storage_type in TensorProto!");
  int64_t storage_type = tensor_proto->storage_type();
  switch (storage_type) {
    case caffe2::TensorProto_StorageType_TYPED:
      // TODO
      AT_ERROR("Storing data in typed field is not suppored yet!");
    case caffe2::TensorProto_StorageType_RAW:
      // TODO
      AT_ERROR("Storing data in raw field is not supported yet!");
    case caffe2::TensorProto_StorageType_EXTERNAL:
      {
        AT_ASSERTM(tensor_proto->has_external_data(), "storage type is EXTERNAL, "
            "but no external_data in TensorProto!");
        auto& external_data = tensor_proto->external_data();
        offset_ = external_data.offset();
        for (int i = 0; i < external_data.strides_size(); ++i) {
          strides_.push_back(external_data.strides(i));
        }
        int64_t source_type = external_data.source_type();
        if (source_type == caffe2::ExternalDataProto_SourceType_INLINE_CONTAINER) {
          AT_ASSERTM(external_data.has_record_id(), "no record_id in ExternalDataProto and source_type is INLINE_CONTAINER!");
          // only load the data of the tensor in LOADER_TENSOR_DATA mode
          uint64_t record_id = caffe2::stoull(external_data.record_id());
          uint64_t size = external_data.size();
          auto it = id_data->find(record_id);
          if (mode == DeserializeMode::LOADER_TENSOR_DATA) {
            // tensor data is only loaded in LOADER_TENSOR_DATA mode
            if (it == id_data->end()) {
              AT_ERROR("Tensor's data is missing in id_data, tensor name is ",
                  name_, ", and record_id is ", caffe2::to_string(record_id));
            }
            data_ = it->second;
            AT_ASSERT(data_->recordId.value() == record_id);
          } else {
            AT_ASSERTM(mode == DeserializeMode::HEADER_ONLY, "unkonw deserialize mode.");
            if (it == id_data->end()) {
              data_ = std::make_shared<SharedData>(record_id, size);
              (*id_data)[record_id] = data_;
            } else {
              data_ = it->second;
            }
          }
        } else if (source_type == caffe2::ExternalDataProto_SourceType_SIMPLE_FILE) {
          // TODO
          AT_ERROR("Storing data in separate file is not supported yet!");
        } else {
          // TODO
          AT_ERROR("Unknown source_type: ", caffe2::to_string(source_type));
        }
        break;
      }
    case caffe2::TensorProto_StorageType_NO_CONTENT:
      {
        noContent_ = true;
        break;
      }
    default:
      AT_ERROR("Uknown storage_type: ", caffe2::to_string(storage_type));
  }

}

void IntermediateTensor::dump(caffe2::TensorProto* tensor_proto) {
  for (auto dim : dims_) {
    tensor_proto->add_dims(dim);
  }
  tensor_proto->set_data_type(static_cast<caffe2::TensorProto_DataType>(dataType_));
  // NB: maybe later we support RAW
  tensor_proto->set_storage_type(caffe2::TensorProto_StorageType::TensorProto_StorageType_EXTERNAL);
  caffe2::ExternalDataProto* data_proto = tensor_proto->mutable_external_data();
  // NB: maybe later we support SIMPLE_FILE
  data_proto->set_source_type(caffe2::ExternalDataProto_SourceType_INLINE_CONTAINER);
  AT_ASSERTM(data_->recordId.has_value(), "recordId is required for SharedData!");
  data_proto->set_record_id(caffe2::to_string(data_->recordId.value()));
  data_proto->set_size(data_->size);
  data_proto->set_offset(offset_);
  for (auto stride : strides_) {
    data_proto->add_strides(stride);
  }
  caffe2::DeviceOption* device_detail = tensor_proto->mutable_device_detail();
  device_detail->set_device_type(deviceDetail_.deviceType);
  if (deviceDetail_.deviceId.has_value()) {
    device_detail->set_device_id(deviceDetail_.deviceId.value());
  }
}

IntermediateParameter::IntermediateParameter(torch::ParameterDef* param_def,
    std::unordered_map<uint64_t, std::shared_ptr<SharedData>>* id_data,
    DeserializeMode mode) {
  AT_ASSERTM(param_def->has_name(), "ParameterDef has no name! ",
      param_def->DebugString());
  name_ = param_def->name();
  isBuffer_ = param_def->is_buffer();
  requireGradient_ = param_def->require_gradient();
  if (param_def->has_tensor()) {
    tensor_.update(param_def->mutable_tensor(), id_data, mode);
  } else {
    // TODO
    AT_ERROR("A ParameterDef does not contain any tensor!");
  }
}

void IntermediateParameter::dump(torch::ParameterDef* param_def) {
  param_def->set_name(name_);
  param_def->set_is_buffer(isBuffer_);
  param_def->set_require_gradient(requireGradient_);
  caffe2::TensorProto* tensor_def = param_def->mutable_tensor();
  tensor_.dump(tensor_def);
}

IntermediateMethod::IntermediateMethod(torch::MethodDef* method_def) {
  AT_ASSERTM(method_def->has_name(), "name is required for MethodDef!");
  name_ = method_def->name();
  if (method_def->has_torch_script()) {
    torchScript_ = method_def->torch_script();
  } else if (method_def->has_graph()) {
    graph_.reset(method_def->release_graph());
  } else {
    AT_ERROR("No method body is found!");
  }
}

IntermediateMethod::IntermediateMethod(const IntermediateMethod& method) {
  name_ = method.name_;
  graph_ = caffe2::make_unique<caffe2::NetDef>(*(method.graph_));
  torchScript_ = method.torchScript_;
}

IntermediateMethod& IntermediateMethod::operator =(const IntermediateMethod& method) {
  name_ = method.name_;
  graph_ = caffe2::make_unique<caffe2::NetDef>(*(method.graph_));
  torchScript_ = method.torchScript_;
  return *this;
}

void IntermediateMethod::dump(torch::MethodDef* method_def) {
  AT_ASSERTM(name_.size() > 0, "IntermediateMethod's name is invalid. name: ", name_);
  method_def->set_name(name_);
  if (graph_) {
    method_def->set_allocated_graph(graph_.release());
  } else {
    method_def->set_torch_script(torchScript_);
  }
}

IntermediateModule::IntermediateModule(torch::ModuleDef* module_def,
    std::unordered_map<uint64_t, std::shared_ptr<SharedData>>* id_data,
    DeserializeMode mode) {
  update(module_def, id_data, mode);
}

void IntermediateModule::update(torch::ModuleDef* module_def,
    std::unordered_map<uint64_t, std::shared_ptr<SharedData>>* id_data,
    DeserializeMode mode) {
  AT_ASSERTM(module_def->has_name(), "name is required for ModuleDef!");
  name_ = module_def->name();
  for (int i = 0; i < module_def->parameters_size(); ++i) {
    auto* param_def = module_def->mutable_parameters(i);
    parameters_.emplace_back(param_def, id_data, mode);
  }

  for (int i = 0; i < module_def->submodules_size(); ++i) {
    auto* sub_def = module_def->mutable_submodules(i);
    submodules_.emplace_back(sub_def, id_data, mode);
  }

  for (int i = 0; i < module_def->methods_size(); ++i) {
    auto* method_def = module_def->mutable_methods(i);
    methods_.emplace_back(method_def);
  }
}

void IntermediateModule::dump(torch::ModuleDef* module_def) {
  module_def->set_name(name_);

  for (int i = 0; i < parameters_.size(); ++i) {
    module_def->add_parameters();
    torch::ParameterDef* param_def = module_def->mutable_parameters(module_def->parameters_size() - 1);
    parameters_.at(i).dump(param_def);
  }

  for (int i = 0; i < submodules_.size(); ++i) {
    module_def->add_submodules();
    torch::ModuleDef* sub_def = module_def->mutable_submodules(module_def->submodules_size() - 1);
    submodules_.at(i).dump(sub_def);
  }

  for (int i = 0; i < methods_.size(); ++i) {
    module_def->add_methods();
    torch::MethodDef* method_def = module_def->mutable_methods(module_def->methods_size() - 1);
    methods_.at(i).dump(method_def);
  }
}

void IntermediateModel::update(torch::ModelDef* model_def,
    std::unordered_map<uint64_t, std::shared_ptr<SharedData>>* id_data,
    DeserializeMode mode) {
  AT_ASSERTM(model_def->has_name(), "name is required for ModelDef.");
  name_ = model_def->name();
  AT_ASSERTM(model_def->has_producer_name(), "producer_name is required for ModelDef.");
  producerName_ = model_def->producer_name();
  producerVersion_ = model_def->producer_version();
  AT_ASSERTM(model_def->has_proto_version(), "proto_version is required for ModelDef.");
  protoVersion_ = model_def->proto_version();
  AT_ASSERTM(model_def->has_main_module(), "main_module is required for ModelDef.");
  mainModule_.update(model_def->mutable_main_module(), id_data, mode);
}

void IntermediateModel::dump(torch::ModelDef* model_def) {
  model_def->set_name(name_);
  model_def->set_producer_name(producerName_);
  model_def->set_producer_version(producerVersion_);
  model_def->set_proto_version(protoVersion_);
  mainModule_.dump(model_def->mutable_main_module());
}

void serializeIntermediateModel(IntermediateModel* imodel,
    torch::jit::PyTorchFileWriter* writer) {
  std::unordered_map<void*, uint64_t> data_id;
  std::stack<IntermediateModule*> imodules;
  imodules.push(imodel->mutableMainModule());
  while (!imodules.empty()) {
    IntermediateModule* m = imodules.top();
    imodules.pop();
    for (auto& param : *(m->mutableParameters())) {
      std::shared_ptr<SharedData> dataptr = param.mutableTensor()->data();
      void* data = dataptr->dataPtr.get();
      size_t size = dataptr->size;
      auto it = data_id.find(data);
      if (it != data_id.end()) {
        dataptr->recordId = it->second;
      } else {
        uint64_t id = writer->writeRecord(data, size);
        dataptr->recordId = id;
        data_id[data] = id;
      }
    }

    for (auto& sub : *(m->mutableSubmodules())) {
      imodules.push(&sub);
    }
  }

  torch::ModelDef model_def;
  imodel->dump(&model_def);
  std::string buffer;
  model_def.SerializeToString(&buffer);
  writer->writeRecord(buffer.c_str(), buffer.size());
}

void serializeIntermediateModel(IntermediateModel* imodel, const std::string& filename) {
  torch::jit::PyTorchFileWriter writer(filename);
  serializeIntermediateModel(imodel, &writer);
  writer.writeEndOfFile();
}

void deserializeIntermediateModel(IntermediateModel* imodel,
    torch::jit::PyTorchFileReader* reader, DeserializeMode mode) {
  std::unordered_map<uint64_t, std::shared_ptr<SharedData>> id_data;
  if (mode == DeserializeMode::HEADER_ONLY) {
    // only load model meta data, no tensor data
    at::DataPtr data_ptr;
    size_t data_size;
    std::tie(data_ptr, data_size) = reader->getLastRecord();
    torch::ModelDef model_def = torch::ModelDef();
    AT_ASSERTM(model_def.ParseFromArray(data_ptr.get(), data_size), "parse metadata (i.e., ModelDef) failed.");;
    imodel->update(&model_def, &id_data, mode);
    return;
  }
  AT_ASSERT(mode == DeserializeMode::LOADER_TENSOR_DATA);
  while (reader->hasNextRecord()) {
    at::DataPtr data_ptr;
    size_t data_key;
    size_t data_size;
    std::tie(data_ptr, data_key, data_size) = reader->getNextRecord();
    if (!reader->hasNextRecord()) {
      // the last record is model data (ModelDef)
      torch::ModelDef model_def = torch::ModelDef();
      AT_ASSERTM(model_def.ParseFromArray(data_ptr.get(), data_size), "parse metadata (i.e., ModelDef) failed.");
      imodel->update(&model_def, &id_data, mode);
      break;
    }
    // first to the second last records are all tensor data
    auto it = id_data.find(data_key);
    AT_ASSERTM(it == id_data.end(), "record id should not be duplicated!");
    id_data[data_key] = std::make_shared<SharedData>(
        data_key, data_size, std::move(data_ptr));
  }
}

void deserializeIntermediateModel(IntermediateModel* imodel, const std::string& filename, DeserializeMode mode) {
  torch::jit::PyTorchFileReader reader(filename);
  deserializeIntermediateModel(imodel, &reader, mode);
}

} // namespace serialize
} // namespace at
