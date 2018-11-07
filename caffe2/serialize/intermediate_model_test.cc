#include <cstdio>
#include <string>

#include <gtest/gtest.h>

#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/proto/torch_pb.h"
#include "caffe2/serialize/intermediate_model.h"

namespace at {
namespace {

TEST(IntermediateModel, SerializeAndDeserialize) {
  // TODO: split the test cases
  // TODO test different type of device

  // prepare model
  std::string model_name("Test-Model-Name");
  std::string producer_name("Test-Producer-Name");
  std::string producer_version("Test-Producer-Version");
  int64_t proto_version = 2; // invalid, only for testing
  serialize::IntermediateModel imodel;
  imodel.setName(model_name);
  imodel.setProducerName(producer_name);
  imodel.setProducerVersion(producer_version);
  imodel.setProtoVersion(proto_version);

  // prepare main module
  serialize::IntermediateModule* main_module = imodel.mutableMainModule();
  std::string module_name("Test-Module-Name");
  main_module->setName(module_name);

  // prepare method for main module
  std::vector<serialize::IntermediateMethod>* methods =
      main_module->mutableMethods();
  methods->resize(1);
  serialize::IntermediateMethod& method = methods->at(0);
  std::string method_name("Test-Method-Name");
  method.setName(method_name);
  std::string torch_script("Test-Method-Torch-Script");
  method.setTorchScript(torch_script);

  // prepare submodule for main module
  std::vector<serialize::IntermediateModule>* subs =
      main_module->mutableSubmodules();
  subs->resize(1);
  serialize::IntermediateModule& sub_module = subs->at(0);
  std::string sub_name("Test-Submodule-Name");
  sub_module.setName(sub_name);

  // prepare parameters for main module
  std::vector<serialize::IntermediateParameter>* params =
      main_module->mutableParameters();
  std::string param1_name("Test-Parameter-Name");
  bool is_buffer = true;
  bool require_gradient = true;
  // prepare first parameter
  params->emplace_back(param1_name, is_buffer, require_gradient);
  serialize::IntermediateTensor* tensor1 = params->at(0).mutableTensor();
  size_t raw_size = sizeof(float);
  std::vector<int64_t> dims1 = {2, 3, 4};
  for (auto dim : dims1) {
    raw_size *= dim;
  }
  std::vector<int64_t> strides1 = {12, 4, 1};
  *tensor1 = serialize::IntermediateTensor(
      caffe2::TensorProto_DataType_FLOAT, dims1, 0);
  std::vector<char> data_vector;
  data_vector.resize(raw_size);
  for (size_t i = 0; i < data_vector.size(); ++i) {
    data_vector[i] = data_vector.size() - i;
  }
  at::DataPtr data_ptr(data_vector.data(), at::kCPU);
  std::shared_ptr<serialize::SharedData> data =
      std::make_shared<serialize::SharedData>(0, raw_size, std::move(data_ptr));
  tensor1->setData(data);
  tensor1->setStrides(strides1);
  // prepare second parameter, share the data with first parameter
  std::string param2_name = "Test-Parameter-2-Name";
  std::vector<int64_t> dims2 = {3, 4};
  std::vector<int64_t> strides2 = {12, 4, 1};
  int64_t offset2 = 12;
  params->emplace_back(param2_name, is_buffer, require_gradient);
  serialize::IntermediateTensor* tensor2 = params->at(1).mutableTensor();
  *tensor2 = serialize::IntermediateTensor(
      caffe2::TensorProto_DataType_FLOAT, dims2, offset2);
  tensor2->setData(data);
  tensor2->setStrides(strides1);

  // serialize the prepared model
  std::string tmp_name = std::tmpnam(nullptr);
  serialize::serializeIntermediateModel(&imodel, tmp_name);

  // load the serialized model
  serialize::IntermediateModel loaded_model;
  serialize::deserializeIntermediateModel(&loaded_model, tmp_name);

  // verify the loaded model
  ASSERT_EQ(loaded_model.name(), model_name);
  ASSERT_EQ(loaded_model.producerName(), producer_name);
  ASSERT_EQ(loaded_model.producerVersion(), producer_version);
  ASSERT_EQ(loaded_model.protoVersion(), proto_version);

  // verify the main module
  const auto& loaded_main_module = loaded_model.mainModule();
  ASSERT_EQ(loaded_main_module.name(), module_name);

  // verify the method
  ASSERT_EQ(loaded_main_module.methods().size(), 1);
  ASSERT_EQ(loaded_main_module.methods().at(0).name(), method_name);
  ASSERT_EQ(loaded_main_module.methods().at(0).torchScript(), torch_script);

  // verify the submodule
  ASSERT_EQ(loaded_main_module.submodules().size(), 1);
  ASSERT_EQ(loaded_main_module.submodules().at(0).name(), sub_name);
  ASSERT_EQ(loaded_main_module.parameters().size(), 2);

  // verify the parameter
  const auto& loaded_param1 = loaded_main_module.parameters().at(0);
  ASSERT_EQ(loaded_param1.name(), param1_name);
  ASSERT_EQ(loaded_param1.isBuffer(), is_buffer);
  ASSERT_EQ(loaded_param1.requireGradient(), require_gradient);
  const auto& loaded_tensor1 = loaded_param1.tensor();
  ASSERT_EQ(loaded_tensor1.dims(), dims1);
  ASSERT_EQ(loaded_tensor1.strides(), strides1);
  ASSERT_EQ(loaded_tensor1.offset(), 0);
  ASSERT_EQ(loaded_tensor1.deviceDetail().deviceType, 0);
  ASSERT_EQ(loaded_tensor1.noContent(), false);
  ASSERT_EQ(loaded_tensor1.dataType(), caffe2::TensorProto_DataType_FLOAT);
  ASSERT_EQ(loaded_tensor1.data()->size, raw_size);
  ASSERT_EQ(
      std::memcmp(
          loaded_tensor1.data()->dataPtr.get(), data_vector.data(), raw_size),
      0);
  ASSERT_EQ(loaded_tensor1.data()->recordId.value(), 64);
  const auto& loaded_param2 = loaded_main_module.parameters().at(1);
  ASSERT_EQ(loaded_param2.name(), param2_name);
  ASSERT_EQ(loaded_param2.isBuffer(), is_buffer);
  ASSERT_EQ(loaded_param2.requireGradient(), require_gradient);
  const auto& loaded_tensor2 = loaded_param2.tensor();
  ASSERT_EQ(loaded_tensor2.dims(), dims2);
  ASSERT_EQ(loaded_tensor2.strides(), strides2);
  ASSERT_EQ(loaded_tensor2.offset(), offset2);
  ASSERT_EQ(loaded_tensor2.deviceDetail().deviceType, 0);
  ASSERT_EQ(loaded_tensor2.noContent(), false);
  ASSERT_EQ(loaded_tensor2.dataType(), caffe2::TensorProto_DataType_FLOAT);
  ASSERT_EQ(loaded_tensor2.data()->size, raw_size);
  ASSERT_EQ(loaded_tensor2.data()->recordId.value(), 64);
  ASSERT_EQ(
      loaded_tensor2.data()->dataPtr.get(),
      loaded_tensor1.data()->dataPtr.get());
  // TODO test shared data between tensors

  // load the serialized model in HEADER_ONLY mode
  serialize::IntermediateModel lazy_model;
  serialize::deserializeIntermediateModel(
      &lazy_model, tmp_name, serialize::DeserializeMode::HEADER_ONLY);
  ASSERT_EQ(lazy_model.name(), model_name);
  ASSERT_EQ(lazy_model.mainModule().name(), module_name);
  const auto& lazy_params = lazy_model.mainModule().parameters();
  ASSERT_EQ(lazy_params.size(), 2);
  const auto& lazy_param1 = lazy_params.at(0);
  ASSERT_EQ(lazy_param1.name(), loaded_param1.name());
  const auto& lazy_tensor1 = lazy_param1.tensor();
  ASSERT_EQ(lazy_tensor1.offset(), 0);
  ASSERT_EQ(
      lazy_tensor1.data()->recordId.value(),
      loaded_tensor1.data()->recordId.value());
  ASSERT_EQ(lazy_tensor1.data()->dataPtr.get(), nullptr);
  ASSERT_EQ(lazy_tensor1.data()->size, loaded_tensor1.data()->size);
  const auto& lazy_param2 = lazy_params.at(1);
  ASSERT_EQ(lazy_param2.name(), loaded_param2.name());
  const auto& lazy_tensor2 = lazy_param2.tensor();
  ASSERT_EQ(lazy_tensor2.offset(), offset2);
  ASSERT_EQ(
      lazy_tensor2.data()->recordId.value(),
      loaded_tensor2.data()->recordId.value());
  ASSERT_EQ(lazy_tensor2.data()->dataPtr.get(), nullptr);
  ASSERT_EQ(lazy_tensor2.data()->size, loaded_tensor2.data()->size);
  ASSERT_EQ(lazy_tensor2.data()->recordId, lazy_tensor1.data()->recordId);

  std::remove(tmp_name.c_str());
}

} // namespace
} // namespace at
