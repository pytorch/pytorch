// in memory description of all ATen Ops similar to Caffe2 schema
// once C10 exists this can be removed, or stubbed out, but we need
// it now to implement correct semantic checking for script
#pragma once
#include "ATen/ATen.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/function_schema.h"

namespace torch { namespace jit {

const std::vector<FunctionSchema>& getOperatorSchema(const std::string& name);
std::vector<FunctionSchema> & getOperatorSchemas();

}}
