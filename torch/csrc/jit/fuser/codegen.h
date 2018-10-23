#pragma once

#include "torch/csrc/jit/fuser/arg_spec.h"
#include "torch/csrc/jit/fuser/common/annotated_graph.h"
#include "torch/csrc/jit/fuser/common/partition_desc.h"

#include <tuple>
#include <vector>
#include <iostream>
#include <string>

namespace torch { namespace jit { namespace fuser {

std::tuple<
  std::vector<PartitionDesc>
, std::vector<PartitionDesc>
, bool> generateKernel(
  std::ostream& out
, const std::string& name
, AnnotatedGraph& agraph
, const bool is_cuda);

} // namespace fuser
} // namespace jit
} // namespace torch
