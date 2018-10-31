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
  std::vector<serialize::IntermediateMethod>* methods = main_module->mutableMethods();
  methods->resize(1);
  serialize::IntermediateMethod& method = methods->at(0);
  std::string method_name("Test-Method-Name");
  method.setName(method_name);
  std::string torch_script("Test-Method-Torch-Script");
  method.setTorchScript(torch_script);

  // prepare submodule for main module
  std::vector<serialize::IntermediateModule>* subs = main_module->mutableSubmodules();
  subs->resize(1);
  serialize::IntermediateModule& sub_module = subs->at(0);
  std::string sub_name("Test-Submodule-Name");
  sub_module.setName(sub_name);

  // prepare parameters for main module
  std::vector<serialize::IntermediateParameter>* params = main_module->mutableParameters();
  std::string param_name("Test-Parameter-Name");
  bool is_buffer = true;
  bool require_gradient = true;
  params->emplace_back(param_name, is_buffer, require_gradient);
  serialize::IntermediateTensor* tensor = params->at(0).mutableTensor();
  size_t raw_size = sizeof(float);
  std::vector<int64_t> dims = {2, 3, 4};
  for (auto dim : dims) {
    raw_size *= dim;
  }
  std::vector<int64_t> strides = {12, 4, 1};
  *tensor = serialize::IntermediateTensor(caffe2::TensorProto_DataType_FLOAT, dims, 0);
  std::vector<char> data_vector;
  data_vector.resize(raw_size);
  for (size_t i = 0; i < data_vector.size(); ++i) {
    data_vector[i] = data_vector.size() - i;
  }
  at::DataPtr data_ptr(data_vector.data(), at::kCPU);
  std::shared_ptr<serialize::SharedData> data =
    std::make_shared<serialize::SharedData>(0, std::move(data_ptr), raw_size);
  tensor->setData(data);
  tensor->setStrides(strides);

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
  ASSERT_EQ(loaded_main_module.parameters().size(), 1);

  // verify the parameter
  const auto& loaded_param = loaded_main_module.parameters().at(0);
  ASSERT_EQ(loaded_param.name(), param_name);
  ASSERT_EQ(loaded_param.isBuffer(), is_buffer);
  ASSERT_EQ(loaded_param.requireGradient(), require_gradient);
  const auto& loaded_tensor = loaded_param.tensor();
  ASSERT_EQ(loaded_tensor.dims(), dims);
  ASSERT_EQ(loaded_tensor.strides(), strides);
  ASSERT_EQ(loaded_tensor.deviceDetail().deviceType, 0);
  ASSERT_EQ(loaded_tensor.noContent(), false);
  ASSERT_EQ(loaded_tensor.dataType(), caffe2::TensorProto_DataType_FLOAT);
  ASSERT_EQ(loaded_tensor.data()->size, raw_size);
  ASSERT_EQ(std::memcmp(loaded_tensor.data()->dataPtr.get(),
        data_vector.data(), raw_size), 0);
  ASSERT_EQ(loaded_tensor.data()->recordId.value(), 64);

  // TODO test shared data between tensors

  // load the serialized model in LAZY mode
  serialize::IntermediateModel lazy_model;
  serialize::deserializeIntermediateModel(&lazy_model, tmp_name, serialize::DeserializeMode::LAZY);
  const auto& lazy_params = lazy_model.mainModule().parameters();
  ASSERT_EQ(lazy_params.size(), 1);
  const auto& lazy_param = lazy_params.at(0);
  ASSERT_EQ(lazy_param.name(), loaded_param.name());
  const auto& lazy_tensor = lazy_param.tensor();
  ASSERT_EQ(lazy_tensor.data()->recordId.value(), loaded_tensor.data()->recordId.value());
  ASSERT_EQ(lazy_tensor.data()->dataPtr.get(), nullptr);
  ASSERT_EQ(lazy_tensor.data()->size, 0);

  std::remove(tmp_name.c_str());

}

}  // namespace
}  // namespace at
