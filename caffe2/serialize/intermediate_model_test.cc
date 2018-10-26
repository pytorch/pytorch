#include <cstdio>
#include <iostream>
#include <string>

#include <gtest/gtest.h>

#include "caffe2/serialize/inline_container.h"

namespace at {
namespace {

TEST(IntermediateModel, SerializeAndDeserialize) {

  // initialize the imodel for test
  std::string model_name("Test-Model-Name");
  std::string producer_name("Test-Producer-Name");
  std::string producer_version("Test-Producer-Version");
  int64_t proto_version = 1;

  IntermediateModel imodel;
  imodel.setName(model_name);
  imodel.setProducerName(producer_name);
  imodel.setProducerVersion(producer_version);
  imodel.setProtoVersion(proto_version);

  IntermediateModule* main_module = imodel.mutableMainModule();
  std::string module_name("Test-Module-Name");
  main_module->setName(module_name);
  std::vector<IntermediateModule>* subs = main_module->mutableSubmodules();
  subs->resize(1);
  IntermediateModule& sub_module = subs->at(0);
  std::string sub_name("Test-Submodule-Name");
  sub_module->setName(sub_name);

  std::vector<IntermediateParameter>* params = main_module->mutableParameters();
  params->resize(1);
  IntermediateParameter& param = params->at(0);
  std::string param_name("Test-Parameter-Name");
  param->setName(param_name);
  bool is_buffer = true;
  param->setIsBuffer(is_buffer);
  bool require_gradient = true;
  param->setRequireGradient(require_gradient);
  IntermediateParameter* tensor = param->mutableTensor();
  vector<int64_t>* dims = tensor->mutableDims();
  size_t raw_size = sizeof(float);
  for (size_t i = 2; i < 5; ++i) {
    raw_size *= i;
    dims->push_back(i);
  }
  tensor->dataType_ = caffe2::TensorProto_DataType::TensorProto_DataType_FLOAT;
  void* raw_data = malloc(raw_size);
  at::DataPtr data_ptr(raw_data, raw_data, free, at::kCPU);
  std::shared_ptr<SharedData> data = make_shared<SharedData>(0, std::move(data_ptr));
  tensor.setData(std::move(data));
}

}  // namespace
}  // namespace at
