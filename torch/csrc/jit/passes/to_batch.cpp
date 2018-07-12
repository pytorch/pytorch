#include "torch/csrc/jit/passes/to_batch.h"
#include "torch/csrc/jit/script/compiler.h"

namespace torch { namespace jit {

std::unordered_map<std::string, std::shared_ptr<Graph>> ToBatch::batch_operator_table;

void ToBatch::toBatch(Block* block, Block* res_block) {
  // change inputs of a graph - expand tensor to {data, mask, dims}
  auto size = block->inputs().size();
  for(size_t i = 0; i < size; i++){
    auto input = block->inputs()[i];
    auto name = input->uniqueName();
    res_block->addInput(name + "_data");
    res_block->addInput(name + "_mask");
    res_block->addInput(name + "_dims");
    batch_map[input] = std::vector<Value*>(res_block->inputs().slice(i * 3, 3));
  }

  for (auto it = block->nodes().begin(); it != block->nodes().end(); it++) {
    auto n = *it;
    // replace tensor operator to BatchTensor operator
    if(n->kind().is_aten()){
      auto batch_graph = batch_operator_table.at(n->kind().toUnqualString());
      WithInsertPoint guard(res_block);
      std::vector<Value*> new_inputs;
      for(Value *input : n->inputs()){
        if(batch_map.find(input) != batch_map.end()){
          auto new_input = batch_map.at(input);
          new_inputs.insert(new_inputs.end(), new_input.begin(), new_input.end());
        }
        else{
          throw std::runtime_error("NYI: non-tensor input for aten operator is not supported yet");
        }
      }
      auto outputs = script::inlineCallTo(*res_block->owningGraph(), *batch_graph, new_inputs);
      // Assume all outputs from inlined operator implementation are in the triple form.
      for(size_t i = 0; i < n->outputs().size(); i++){
        auto output = n->outputs()[i];
        batch_map[output] = std::vector<Value*>(outputs.begin() + i * 3, outputs.begin() + i * 3 + 3);
      }
    }
    else if(n->kind().is_prim()){
      throw std::runtime_error("NYI: node of prim kind is not supported to transform to batch graph yet");
    }
  }
  // change outputs of a graph - expand tensor to {data, mask, dims}
  for(Value* output : block->outputs()){
    auto r_output = batch_map.at(output);
    res_block->registerOutput(r_output[0]);
    res_block->registerOutput(r_output[1]);
    res_block->registerOutput(r_output[2]);
  }
}

std::shared_ptr<Graph> to_batch_graph(std::shared_ptr<Graph>& graph){
  // std::cout<<graph->toString()<<std::endl;
  auto res_graph = std::make_shared<Graph>(graph->scope_root());
  ToBatch to_batch;
  to_batch.toBatch(graph->block(), res_graph->block());
  // std::cout<<res_graph->toString()<<std::endl;
  return res_graph;
}

void initRegisterBatchOpsBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("to_batch_graph", &to_batch_graph);
  m.def("register_batch_operator", [](std::string name, std::shared_ptr<Graph> graph){
    ToBatch::batch_operator_table[name] = graph;
  });
}

}} // namespace torch.jit
