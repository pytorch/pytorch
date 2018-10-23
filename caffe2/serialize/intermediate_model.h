#pragma once

#include <unordered_map>

#include <ATen/core/Allocator.h>

#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/proto/torch_pb.h"
#include "caffe2/serialize/inline_container.h"

namespace at {

struct IntermediateDeviceOption {
  int32_t deviceType = 0;
  bool hasDeviceId = false;
  bool hasRandomSeed = false;
  bool hasNodeName = false;
  int32_t deviceId;
  uint32_t randomSeed;
  std::string nodeName;
  std::vector<std::string> extraInfo;
};

class IntermediateTensor final {
 public:
  // constructor
  IntermediateTensor() = default;

  // update functions
  void update(caffe2::TensorProto* tensor_proto,
      std::unordered_map<int64_t, at::DataPtr>* id_tensor) {
    CAFFE_ENFORCE(tensor_proto->has_data_type(), "no data_type in TensorProto!");
    dataType_ = tensor_proto->data_type();
    for (int i = 0; i < tensor_proto->dims_size()) {
      dims_.push_back(tensor_proto->dims(i));
    }
    CAFFE_ENFORCE(tensor_proto->has_name(), "no name in TensorProto!");
    name_ = tensor_proto->name();
    if (tensor_proto->has_device_detail()) {
      const auto& device_detail = tensor_proto->device_detail();
      deviceDetail_.deviceType = device_detail.device_type;
      if (device_detail.has_device_id()) {
        deviceDetail_.hasDeviceId = true;
        deviceDetail_.devicedId = device_detail.device_id();
      }
      if (device_detail.has_random_seed()) {
        deviceDetail_.hasRandomSeed = true;
        deviceDetail_.randomSeed = device_detail.random_seed();
      }
      if (device_detail.has_node_name()) {
        deviceDetail_.hasNodeName = true;
        deviceDetail_.nodeName = device_detail.nodeName();
      }
      for (int i = 0; i < device_detail.extra_info_size(); ++i) {
        deviceDetail_.extraInfo.emplace_back(device_detail.extra_info(i));
      }
    }
    CAFFE_ENFORCE(tensor_proto->has_storage_type(), "no storage_type in TensorProto!");
    int64_t storage_type = tensor_proto->storage_type();
    switch (storage_type) {
      case TensorProto_StorageType_TYPED:
        // TODO
        CAFFE_THROW("Not implemented!");
      case TensorProto_StorageType_RAW:
        // TODO create the data type
        CAFFE_THROW("Not implemented!");
      case TensorProto_StorageType_EXTERNAL:
        CAFFE_ENFORCE(tensor_proto->has_external_data(), "storage type is EXTERNAL, "
            "but no external_data in TensorProto!");
        const auto& external_data = tensor_proto->external_data();
        if (external_data->has_offset()) {
          offset_ = external_data->offset();
        }
        for (int i = 0; i < external_data.strides_size(); ++i) {
          strides_->push_back(external_data.strides(i));
        }
        CAFFE_ENFORCE(external_data->has_source_type(), "no source_type in TensorProto!");
        int64_t source_type = external_data->source_type();
        if (source_type == ExternalDataProto_SourceType::ExternalDataProto_SourceType_INLINE_CONTAINER) {
          CAFFE_ENFORCE(external_data->has_record_id(), "no record_id in ExternalDataProto!");
          int64_t record_id = external_data->record_id();
          auto it = id_tensor.find(record_id);
          if (it == id_tensor.end()) {
            CAFFE_THROW("Tensor's data is missing, tensor name is ", name_);
          }
          dataPtr_ = std::move(it->second);
        } else if (source_type == ExternalDataProto_SourceType::ExternalDataProto_SourceType_SIMPLE_FILE) {
          // TODO
          CAFFE_THROW("Not implemented!");
        } else {
          // TODO
          CAFFE_THROW("Not implemented!");
        }
        break;
      case TensorProto_StorageType_NO_CONTENT:
        noContent_ = true;
    }
  }

  void updateData(at::DataPtr&& dataPtr) {
    this->dataPtr_ = std::move(dataPtr);
  }

 private:
  std::string name_;
  int64_t dataType_;
  vector<int64_t> dims_;
  int64_t offset_;
  vector<int64_t> strides_;
  // TODO: since we still have 2 different Tensor classes in Caffe2 and PyTorch
  // right now, let's just store the data pointer, and create Tensors
  // while converting IntermediateModel to the JitScriptModule/Predictor/etc.
  at::DataPtr dataPtr_;
  IntermediateDeviceOption deviceDetail_;
  bool noContent_ = false;
};

class IntermediateParameter final {
 public:
  // constructor
  explicit IntermediateParameter(torch::ParameterDef* param_def,
      std::unordered_map<std::string, IntermediateTensor*>* id_tensor) {
    if (param_def->has_is_buffer()) {
      isBuffer_ = param_def->is_buffer();
    }
    if (param_def->has_require_gradient()) {
      requireGradient_ = param_def->require_gradient();
    }
    if (param_def->has_tensor()) {
      tensor_.update(param_def->mutable_tensor(), id_tensor);
    } else {
      // TODO throw exception
    }
  }

  // getters
  bool isBuffer() const {
    return isBuffer_;
  }

  bool requireGradient() const {
    return requireGradient_;
  }

  IntermediateTensor* mutableTensor() {
    return &tensor_;
  }

 private:
  bool isBuffer_ = false;
  bool requireGradient_ = false;
  IntermediateTensor tensor_;

};

class IntermediateMethod final {
 public:
  explicit IntermediateMethod(torch::MethodDef* method_def) {
    if (method_def->has_name()) {
      name_ = method_def->name();
    } else {
      // TODO throw exception
    }
    if (method_def->has_torch_script()) {
      torchScript_ = method_def->torch_script();
    } else if (method_def->has_graph()) {
      graph_ = make_unique(release_graph());
    } else {
      // TODO throw exception
    }
  }

 private:
  std::string name_;
  std::unique_ptr<caffe2::NetDef> graph_;
  std::string torchScript_;
};

class IntermediateModule final {
 public:
  // constructor
  IntermediateModule() = default;

  // update functions
  void update(torch::ModuleDef* module_def,
      std::unordered_map<std::string, IntermediateTensor*>* id_tensor) {
    if (module_def->has_name()) {
      name_ = module_def->name();
    } else {
      // TODO throw exception
    }
    for (int i = 0; i < module_def->parameters_size(); ++i) {
      auto* param_def = module_def->mutable_parameters(i);
      parameters_.emplace_back(param_def, id_tensor);
    }

    for (int i = 0; i < module_def->submodules_size(); ++i) {
      auto* sub_def = module_def->mutable_submodules(i);
      submodules_.emplace_back(sub_def, id_tensor);
    }

    for (int i = 0; i < module_def->methods_size(); ++i) {
      auto* method_def = module_def->mutable_methods(i);
      methods_.emplace_back(method_def);
    }
  }

 private:
  std::string name_;
  std::vector<IntermediateParameter> parameters_;
  std::vector<IntermediateModule> submodules_;
  std::vector<IntermediateMethod> methods_;
};

class IntermediateModel final {
 public:
  // constructor
  IntermediateModel() = default;

  // update functions
  void update(torch::ModelDef* model_def) {
    if (model_def->has_name()) {
      name_ = model_def->name();
    } else {
        // TODO throw exception
    }
    if (model_def->has_producer_name()) {
      producerName_ = model_def->producer_name();
    } else {
        // TODO throw exception
    }
    if (model_def->has_producer_version()) {
      producerVersion_ = model_def->producer_version();
    } else {
        // TODO throw exception
    }
    if (model_def->has_proto_version()) {
      protoVersion_ = model_def->proto_version();
    } else {
        // TODO throw exception
    }
    if (model_def->has_main_module()) {
      mainModule.update(model_def->mutable_main_module(), id2Data_);
    } else {
        // TODO throw exception
    }
  }

  void update(torch::jit::PyTorchFileReader* reader) {
    at::DataPtr proto_ptr;
    size_t proto_size;
    while (reader->hasNextRecord()) {
      at::DataPtr data_ptr;
      int64_t data
      size_t data_size;
      std::tie(data_ptr, data_key, data_size) = reader->getNextRecord();
      if (!reader->hasNextRecord()) {
        auto it = id2Data_.find(data_key);
        if (it != id2Data_.end()) {
          id2Data_->second->updateData(std::move(data_ptr));
        } else {
          id2Data_[data_key] = std::move(data_ptr);
        }
        continue;
      }
      torch::ModelDef model_def = torch::ModelDef();
      model_def.ParsePartialFromArray(proto_ptr.get(), proto_size);
      update(&model_def);
    }

    for (std::pair<int64_t, at::DataPtr> element : id2Data_) {
      if (element->second->get() == nullptr) {
        // TODO throw exception
      }
    }
  }

  void update(const std::string& filename) {
    PyTorchFileReader reader{filename};
    update(&reader);
  }

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

  int64_t producerVersion() const {
    return protoVersion_;
  }

  const IntermediateModule& mainModule() const {
    return mainModule_;
  }

  std::unordered_map<int64_t, at::DataPtr>* mutableId2Data() {
    return &id2Data_;
  }

 private:
  std::string name_;
  std::string producerName_;
  std::string producerVersion_;
  int64_t protoVersion_;
  IntermediateModule mainModule_;
  std::unordered_map<int64_t, at::DataPtr> id2Data_;

};

}  // namespace at
