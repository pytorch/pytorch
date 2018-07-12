#pragma once

#include "torch/csrc/jit/pybind.h"
#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

class ToBatch {
private:
  // mapping from tensor in original graph to {data, mask, dims} in new graph
  std::unordered_map<Value*, std::vector<Value*>> batch_map;
  // mapping from input in original graph to new input in new graph - used in createClone
  std::unordered_map<Value*, Value*> rn_env;
  std::function<Value*(Value*)> rn_fn = [this](Value* v) { return rn_env.at(v); };

private:
  std::vector<std::string> get_name(std::string name);
  void visitAten(Node* n, Block* block, Block* res_block, std::unordered_map<std::string, Value*>& var_map);
  void visitConstant(Node* n, Block* block, Block* res_block);
  void visitNumToTensor(Node* n, Block* block, Block* res_block);
  void visitIf(Node* n, Block* block, Block* res_block, std::unordered_map<std::string, Value*>& var_map);
  void visitLoop(Node* n, Block* block, Block* res_block, std::unordered_map<std::string, Value*>& var_map);

public:
  static std::unordered_map<std::string, std::shared_ptr<Graph>> batch_operator_table;
  void toBatch(Block* block, Block* res_block, std::unordered_map<std::string, Value*>& upper_var_map);
};

std::shared_ptr<Graph> to_batch_graph(std::shared_ptr<Graph>& graph);
void initRegisterBatchOpsBindings(PyObject* module);
}}
