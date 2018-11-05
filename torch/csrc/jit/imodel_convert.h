#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/script/module.h"


namespace torch {
namespace jit {

// IntermediateModel (for short, imodel) serves as a mid layer between serialization format and
// in-memory representation, this can helps us hide protobuf dependencies.

class ScriptModuleToIntermediateModel {
 public:
  static void convertToModel(const script::Module& smodule, at::serialize::IntermediateModel* imodel);

 private:
  static void convertModule(const script::Module& smodule, const std::string& name,
      at::serialize::IntermediateModule* imodule);
  static void convertParameter(const script::NamedParameter& sparam, at::serialize::IntermediateParameter* iparam);
  static void convertTensor(const at::Tensor& stensor, at::serialize::IntermediateTensor* itensor);
  static void convertMethod(const script::Method& smethod, at::serialize::IntermediateMethod* imethod);
};

class IntermediateModelToScriptModule {
 public:
  // load using file name, imodel was loaded using LAZY mode
  static void convertToModule(const at::serialize::IntermediateModel& imodel, const std::string& filename, script::Module* module);
  // load using istream, imodel was loaded using LAZY mode
  static void convertToModule(const at::serialize::IntermediateModel& imodel, std::istream* is, script::Module* module);
  // imodel was loaded using EAGER mode, which means tensor data is already ready for use
  static void convertModule(const at::serialize::IntermediateModule& imodule, script::Module* smodule);

 private:
  static void convertModule(const at::serialize::IntermediateModule& imodule, const std::string& filename,
      script::Module* smodule, std::unordered_map<uint64_t, std::shared_ptr<at::Storage>> id_storage);
  static void convertModule(const at::serialize::IntermediateModule& imodule, PyTorchStreamReader* reader,
      script::Module* smodule, std::unordered_map<uint64_t, std::shared_ptr<at::Storage>> id_storage);
  static void convertModule(const at::serialize::IntermediateModule& imodule, script::Module* smodule);
  static void convertModule(const at::serialize::IntermediateModule& imodule, script::Module* smodule);
  static void convertParameter(const at::serialize::IntermediateParameter& iparam, script::NamedParameter* sparam);
  static void convertTensor(const at::serialize::IntermediateTensor& itensor, at::Tensor* stensor);
  static void convertMethod(const at::serialize::IntermediateMethod& imethod, script::Method* method);
};

} // namespace jit
} // namespace torch
