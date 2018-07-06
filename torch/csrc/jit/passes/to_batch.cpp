#include "torch/csrc/jit/passes/to_batch.h"
#include "torch/csrc/jit/script/compiler.h"

namespace torch { namespace jit {

// map from batchTensor to {data, mask, dims}
static std::unordered_map<Value*, std::vector<Value*>> batch_map;

static void ToBatch(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end(); it++) {
    auto n = *it;
    // std::cout << *n << std::endl;
    // replace tensor operator to BatchTensor operator
    if(n->kind().is_aten()){
      auto batch_graph = batch_operator_table.at(n->kind().toUnqualString());
      // std::cout << "batch_graph:" << *batch_graph << std::endl;
      WithInsertPoint guard(n);
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
      auto outputs = script::inlineCallTo(*n->owningGraph(), *batch_graph, new_inputs);
      // Assume all outputs from inlined operator implementation are in the triple form.
      for(size_t i = 0; i < n->outputs().size(); i++){
        auto output = n->outputs()[i];
        output->replaceAllUsesWith(outputs[i * 3]);
        batch_map[outputs[i * 3]] = std::vector<Value*>(outputs.begin() + i * 3, outputs.begin() + i * 3 + 3);
      }
      it.destroyCurrent();
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
    for (Block *subblock : n->blocks()) {
      ToBatch(subblock);
    }
  }
}

void to_batch_graph(std::shared_ptr<Graph>& graph, int64_t batch_size){
  // batch_size: not used yet, will be used to deal with scalarType
  // std::cout<<graph->toString()<<std::endl;
  auto size = graph->inputs().size();
  for(size_t i = 0; i < size; i++){
    auto input = graph->inputs()[i * 3];
    auto name = input->uniqueName();
    graph->insertInput(i * 3, name + "_data");
    graph->insertInput(i * 3 + 1, name + "_mask");
    graph->insertInput(i * 3 + 2, name + "_dims");
    batch_map[graph->inputs()[i * 3]] = std::vector<Value*>(graph->inputs().slice(i * 3, 3));
    input->replaceAllUsesWith(graph->inputs()[i * 3]);
    graph->eraseInput(i * 3 + 3);
  }
  ToBatch(graph->block());
  size = graph->outputs().size();
  for(size_t i = 0; i < size; i++){
    auto output = graph->outputs()[i * 3];
    auto new_output = batch_map.at(output);
    graph->registerOutput(new_output[1]);
    graph->registerOutput(new_output[2]);
  }
  // std::cout<<graph->toString()<<std::endl;
}

void initRegisterBatchOpsBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("to_batch_graph", &to_batch_graph);
  m.def("register_batch_operator", [](std::string name, std::shared_ptr<Graph> graph){
    batch_operator_table[name] = graph;
  });
}

}} // namespace torch.jit
