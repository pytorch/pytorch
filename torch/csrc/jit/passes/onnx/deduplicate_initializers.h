#pragma once

#include <memory>

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void DeduplicateInitializers(
    std::shared_ptr<Graph>& g,
    std::map<std::string, IValue>& paramsDict,
    bool is_train);

} // namespace jit

} // namespace torch
