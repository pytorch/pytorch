#pragma once

#include "torch/csrc/jit/ir.h"

#include <ATen/ATen.h>
#include <vector>

namespace torch { namespace jit {

// This function mutates the given graph (which should only contain a single stage)
// by appending nodes of a next stage, which computes the Jacobian-vector product
// (aka backward) of inputs to the first stage w.r.t. the outputs of the first stage.
void differentiate(std::shared_ptr<Graph>& graph);

}}
