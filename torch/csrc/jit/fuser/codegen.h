#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fuser/arg_spec.h"
#include "torch/csrc/jit/fuser/partition_desc.h"
#include "torch/csrc/jit/fuser/tensor_desc.h"

#include <tuple>
#include <vector>
#include <iostream>
#include <string>

namespace torch { namespace jit { namespace fuser {

std::tuple<
  std::string
, std::vector<PartitionDesc>
, std::vector<PartitionDesc>
, bool> generateKernel(
  const std::string& name
, const Graph& graph
, const int device
, const std::vector<TensorDesc>& input_desc
, const std::vector<TensorDesc>& output_desc
, const bool use_cuda);

} // namespace fuser
} // namespace jit
} // namespace torch
