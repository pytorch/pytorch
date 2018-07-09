#pragma once

#include "torch/csrc/jit/pybind.h"
#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

std::shared_ptr<Graph> to_batch_graph(std::shared_ptr<Graph>& graph, int64_t batch_size=1);
static std::unordered_map<std::string, std::shared_ptr<Graph>> batch_operator_table;
void initRegisterBatchOpsBindings(PyObject* module);
}}
