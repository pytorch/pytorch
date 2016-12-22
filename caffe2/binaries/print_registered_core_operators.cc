#include <iostream>
#include <string>

#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/operator_schema.h"

CAFFE2_DEFINE_string(schema, "",
                     "Print doc and schema of a particular operator");

static bool HasSchema(const std::string& str) {
  return caffe2::OpSchemaRegistry::Schema(str);
}

static bool HasDoc(const std::string& str) {
  const auto* schema = caffe2::OpSchemaRegistry::Schema(str);
  return (schema != nullptr) && (schema->doc() != nullptr);
}

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);

  if (!caffe2::FLAGS_schema.empty()) {
    const auto* schema = caffe2::OpSchemaRegistry::Schema(
        caffe2::FLAGS_schema);
    if (!schema) {
      std::cerr << "Operator " << caffe2::FLAGS_schema
                << " doesn't have a schema" << std::endl;
      return 1;
    }
    std::cout << "Operator " << caffe2::FLAGS_schema << ": " << std::endl
              << *schema;
    return 0;
  }

  for (const auto& pair : *caffe2::gDeviceTypeRegistry()) {
    std::cout << "Device type " << pair.first
#ifndef CAFFE2_USE_LITE_PROTO
              << " (" << caffe2::DeviceType_Name(
                             static_cast<caffe2::DeviceType>(pair.first))
              << ")"
#endif
              << std::endl;
    for (const auto& key : pair.second->Keys()) {
      std::cout << "\t(schema: " << HasSchema(key) << ", doc: " << HasDoc(key)
                << ")\t" << key << std::endl;
    }
  }

  std::cout << "Operators that have gradients registered:" << std::endl;
  for (const auto& key : caffe2::GradientRegistry()->Keys()) {
    std::cout << "\t(schema: " << HasSchema(key) << ", doc: "
              << HasDoc(key) << ")\t"
              << key << std::endl;
  }
  return 0;
}
