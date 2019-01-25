#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/pybind.h>

#include <ATen/ATen.h>

namespace torch {
namespace jit {

class ToBatch {
 private:
  // number of tensors to represent a expanded BatchTensor. {data, mask, dims}
  // for now.
  const size_t EXP_BTENSOR_SIZE = 3;
  const std::vector<std::string> EXP_BTENSOR_NAME = {"data", "mask", "dims"};
  // mapping from tensor in original graph to {data, mask, dims} in new graph
  std::unordered_map<Value*, std::vector<Value*>> batch_map;
  // mapping from input in original graph to new input in new graph - used in
  // createClone
  std::unordered_map<Value*, Value*> rn_env;
  std::function<Value*(Value*)> rn_fn = [this](Value* v) {
    return rn_env.at(v);
  };

 private:
  std::shared_ptr<Graph> getBatchOperator(
      const std::string& name,
      int64_t input_num = -1);
  void visitAten(Node* n, Block* block, Block* res_block);
  void visitConstant(Node* n, Block* block, Block* res_block);
  void visitNumToTensor(Node* n, Block* block, Block* res_block);
  void visitTensorToNum(Node* n, Block* block, Block* res_block);
  void visitListConstruct(Node* n, Block* block, Block* res_block);
  void visitIf(Node* n, Block* block, Block* res_block);
  void visitLoop(Node* n, Block* block, Block* res_block);

 public:
  static std::unordered_map<std::string, std::vector<std::shared_ptr<Graph>>>
      batch_operator_table;
  TORCH_API void toBatch(Block* block, Block* res_block);
};

TORCH_API std::shared_ptr<Graph> to_batch_graph(std::shared_ptr<Graph> graph);
TORCH_API void initRegisterBatchOpsBindings(PyObject* module);
} // namespace jit
} // namespace torch
