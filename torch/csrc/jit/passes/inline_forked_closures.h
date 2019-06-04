#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace script {

void inlineForkedClosures(std::shared_ptr<Graph>& to_clean);

}
} // namespace jit
} // namespace torch
