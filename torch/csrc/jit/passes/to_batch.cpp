#include "torch/csrc/jit/passes/to_batch.h"
#include "torch/csrc/jit/script/compiler.h"

namespace torch { namespace jit {

// map from batchTensor to {data, mask, dims}
static std::unordered_map<Value*, std::vector<Value*>> batch_map;

static void ToBatch(Block* block, Block* res_block) {
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
          new_inputs.push_back(input);
        }
      }
      auto outputs = script::inlineCallTo(*res_block->owningGraph(), *batch_graph, new_inputs);
      // Assume all outputs from inlined operator implementation are in the triple form.
      for(size_t i = 0; i < n->outputs().size(); i++){
        auto output = n->outputs()[i];
        batch_map[output] = std::vector<Value*>(outputs.begin() + i * 3, outputs.begin() + i * 3 + 3);
      }
    }
    // control flow: not supported yet, will be added further
    else if(n->kind().is_prim()){
      if(n->kind() == prim::Loop){
        // TODO
      }
      else if(n->kind() == prim::If){
        // TODO
      }
    }
  }
}

std::shared_ptr<Graph> to_batch_graph(std::shared_ptr<Graph>& graph, int64_t batch_size){
  // batch_size: not used yet, will be used to deal with scalarType
  // std::cout<<graph->toString()<<std::endl;
  auto res_graph = std::make_shared<Graph>(graph->scope_root());
  auto size = graph->inputs().size();
  for(size_t i = 0; i < size; i++){
    auto input = graph->inputs()[i];
    auto name = input->uniqueName();
    res_graph->addInput(name + "_data");
    res_graph->addInput(name + "_mask");
    res_graph->addInput(name + "_dims");
    batch_map[input] = std::vector<Value*>(res_graph->inputs().slice(i * 3, 3));
  }
  ToBatch(graph->block(), res_graph->block());

  for(Value* output : graph->outputs()){
    auto r_output = batch_map.at(output);
    res_graph->registerOutput(r_output[0]);
    res_graph->registerOutput(r_output[1]);
    res_graph->registerOutput(r_output[2]);
  }
  // std::cout<<res_graph->toString()<<std::endl;
  return res_graph;
}

void initRegisterBatchOpsBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("to_batch_graph", &to_batch_graph);
  m.def("register_batch_operator", [](std::string name, std::shared_ptr<Graph> graph){
    batch_operator_table[name] = graph;
  });
}

}} // namespace torch.jit
