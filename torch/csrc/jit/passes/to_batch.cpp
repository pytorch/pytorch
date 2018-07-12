#include "torch/csrc/jit/passes/to_batch.h"
#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/tensor_conversions.h"

namespace torch { namespace jit {

std::unordered_map<std::string, std::shared_ptr<Graph>> ToBatch::batch_operator_table;

// eg: "a.1" -> {"a", "1"}; "a" -> {"a"}
std::vector<std::string> ToBatch::get_name(std::string name) {
  auto last_dot_pos = name.find_last_of('.');
  if (last_dot_pos != std::string::npos && last_dot_pos + 1 != name.size()) {
    if (name.find_first_not_of("0123456789", last_dot_pos + 1) == std::string::npos) {
      auto suffix = name.substr(last_dot_pos + 1);
      std::string name_base = name.substr(0, last_dot_pos);
      return {name_base, suffix};
    }
  }
  return {name};
}

// replace aten operator node with BatchTensor operator graph
void ToBatch::visitAten(Node* n, Block* block, Block* res_block, std::unordered_map<std::string, Value*>& var_map){
  if(n->outputs().size() > 1){
    throw std::runtime_error("Cannot process multiple assignment");
  }
  auto batch_graph = batch_operator_table.at(n->kind().toUnqualString());
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

  // do update on assignment
  auto name_base = get_name(n->output()->uniqueName())[0];
  if(var_map.find(name_base) != var_map.end()){
    std::vector<Value*> inputs(batch_map.at(var_map.at(name_base)));
    inputs.insert(inputs.end(), outputs.begin(), outputs.end());
    outputs = script::inlineCallTo(*res_block->owningGraph(), *batch_operator_table.at("update"), inputs);
  }
  // Assume all outputs from inlined operator implementation are in the triple form.
  for(size_t i = 0; i < n->outputs().size(); i++){
    auto output = n->outputs()[i];
    batch_map[output] = std::vector<Value*>(outputs.begin() + i * 3, outputs.begin() + i * 3 + 3);
    if(output->hasUniqueName()){
      var_map[get_name(output->uniqueName())[0]] = output;
    }
  }
}

// clone prim::Constant to new graph
void ToBatch::visitConstant(Node* n, Block* block, Block* res_block){
  auto res_graph = res_block->owningGraph();
  auto* r_node = res_graph->createClone(n, rn_fn);
  r_node->setStage(n->stage());
  res_graph->appendNode(r_node);
  rn_env[n->output()] = r_node->output();
}

// change return tensor to {data, mask, dims}
void ToBatch::visitNumToTensor(Node* n, Block* block, Block* res_block){
  auto res_graph = res_block->owningGraph();
  auto* r_node = res_graph->createClone(n, rn_fn);
  r_node->setStage(n->stage());
  res_graph->appendNode(r_node);
  auto outputs = script::inlineCallTo(*res_block->owningGraph(), *batch_operator_table.at("batch_from_scalar_tensor"), r_node->outputs());
  batch_map[n->output()] = outputs;
}

// elif is not supported
// assume every variable assigned in an if statement is already defined before
void ToBatch::visitIf(Node* n, Block* block, Block* res_block, std::unordered_map<std::string, Value*>& var_map){
  auto res_graph = res_block->owningGraph();

  // create prim::If node for res_block
  auto add_if_node = [&](Block* block, std::shared_ptr<Graph> cond_graph, std::vector<Value*> cond, std::vector<Value*> unchanged_outputs){
    auto outputs = script::inlineCallTo(*res_block->owningGraph(), *cond_graph, cond); // if condition graph: any/else_any
    rn_env[n->input()] = outputs[0];
    auto* r_node = res_graph->createClone(n, rn_fn, /*copy_blocks=*/false);
    r_node->setStage(n->stage());
    res_graph->appendNode(r_node);
    auto then_block = r_node->addBlock();
    toBatch(block, then_block, var_map);
    // variables assigned in then_block will remain the previous value in else_block
    auto else_block = r_node->addBlock();
    for(Value* output : unchanged_outputs){
      else_block->registerOutput(output);
    }
    // change outputs of prim::If
    auto size = r_node->outputs().size();
    for(size_t i = 0; i < size; i++){
      auto output = r_node->outputs()[i * 3];
      r_node->insertOutput(i * 3 + 1)->setType(output->type());
      r_node->insertOutput(i * 3 + 2)->setType(output->type());
    }
    return r_node;
  };

  auto cond = batch_map.at(n->input());
  std::vector<Value*> unchanged_outputs; // used to register outputs in else_block
  for(Value* output : n->outputs()){
    output = var_map.at(get_name(output->uniqueName())[0]);
    for(Value* else_output : batch_map.at(output)){
      unchanged_outputs.push_back(else_output);
    }
  }
  auto if_node = add_if_node(n->blocks()[0], batch_operator_table.at("any"), cond, unchanged_outputs);
  auto else_node = add_if_node(n->blocks()[1], batch_operator_table.at("any_false"), cond, unchanged_outputs);

  // combine results from two if nodes
  for(size_t i = 0; i < n->outputs().size(); i++){
    std::vector<Value*> inputs(cond);
    for(size_t j = 0; j < 3; j++){
      inputs.push_back(if_node->outputs()[i * 3 + j]);
    }
    for(size_t j = 0; j < 3; j++){
      inputs.push_back(else_node->outputs()[i * 3 + j]);
    }
    auto outputs = script::inlineCallTo(*res_block->owningGraph(), *batch_operator_table.at("where"), inputs);
    batch_map[n->outputs()[i]] = outputs;
  }
}

void ToBatch::visitLoop(Node* n, Block* block, Block* res_block, std::unordered_map<std::string, Value*>& var_map){
  auto res_graph = res_block->owningGraph();

  if(n->inputs()[1]->type()->str() != "Tensor"){
    throw std::runtime_error("NYI: Loop with non-tensor condition is not supported yet");
  }
  // create prim::Loop node for res_block
  auto cond = batch_map.at(n->inputs()[1]);
  auto cond_any = script::inlineCallTo(*res_block->owningGraph(), *batch_operator_table.at("any"), cond);
  rn_env[n->inputs()[1]] = cond_any[0];
  for(size_t i = 2; i < n->inputs().size(); i++){
    auto input = n->inputs()[i];
    rn_env[input] = batch_map.at(input)[0];
  }
  auto* r_node = res_graph->createClone(n, rn_fn, /*copy_blocks=*/false);

  // change inputs of prim::Loop
  for(size_t i = 0; i < 3; i++){
    r_node->insertInput(i + 2, cond[i]);
  }
  for(size_t i = 2; i < n->inputs().size(); i++){
    r_node->insertInput((i - 2) * 3 + 5 + 1, batch_map.at(n->inputs()[i])[1]);
    r_node->insertInput((i - 2) * 3 + 5 + 2, batch_map.at(n->inputs()[i])[2]);
  }
  r_node->setStage(n->stage());
  res_graph->appendNode(r_node);

  // create block for Loop node in res_block
  // first 4 inputs of block: first cond_any, cond_data, cond_mask, cond_dims
  auto loop_block = r_node->addBlock();
  toBatch(n->blocks()[0], loop_block, var_map);

  // change inputs and outputs of block[0] in prim::Loop
  loop_block->eraseInput(2);
  loop_block->eraseInput(1);
  loop_block->insertInput(1, "cond_data");
  loop_block->insertInput(2, "cond_mask");
  loop_block->insertInput(3, "cond_dims");

  WithInsertPoint guard(loop_block);

  // use where operator to update variables
  for(size_t i = 0; i < n->outputs().size(); i++){
    std::vector<Value*> inputs;
    for(size_t j = 0; j < 3; j++){
      inputs.push_back(loop_block->inputs()[j + 1]);
    }
    auto data = batch_map.at(n->blocks()[0]->outputs()[i + 1]);
    inputs.insert(inputs.end(), data.begin(), data.end());
    for(size_t j = 0; j < 3; j++){
      inputs.push_back(loop_block->inputs()[i * 3 + j + 4]);
    }
    auto outputs = script::inlineCallTo(*res_block->owningGraph(), *batch_operator_table.at("where"), inputs);
    batch_map[n->outputs()[i]] = outputs;
    for(size_t j = 0; j < 3; j++){
      loop_block->registerOutput(outputs[j]);
    }
  }

  // update loop conditions
  cond = batch_map.at(n->blocks()[0]->outputs()[0]);
  cond_any = script::inlineCallTo(*res_block->owningGraph(), *batch_operator_table.at("any"), cond);
  loop_block->insertOutput(0, cond_any[0]);
  for(size_t i = 0; i < 3; i++){
    loop_block->insertOutput(i + 1, cond[i]);
  }

  // change outputs of prim::Loop
  auto size = r_node->outputs().size();
  for(size_t i = 0; i < size; i++){
    r_node->insertOutput(i * 3 + 1);
    r_node->insertOutput(i * 3 + 2);
    batch_map[n->outputs()[i]] = r_node->outputs().slice(i * 3, 3);
  }
  for(size_t i = 0; i < 3; i++){
    r_node->insertOutput(i);
  }
}

void ToBatch::toBatch(Block* block, Block* res_block, std::unordered_map<std::string, Value*>& upper_var_map) {
  // mapping from name to last updated Value*
  // eg: a.1 = aten::add(b, c) -> var_map["a"] = a.1
  //     a.2 = aten::sub(c, d) -> var_map["a"] = a.2
  // every block has its own var_map initialized with upper level var_map, destroyed after leaving the block
  // used in 1. prim::If when there exists a variable that is not defined along all paths
  //         2. assignment in aten operators (update func)
  std::unordered_map<std::string, Value*> var_map(upper_var_map);

  WithInsertPoint guard(res_block);

  // change inputs of block - expand tensor to batchtensor(data, mask, dims)
  // eg: a -> a_data, a_mask, a_dims
  // eg: a.1 -> a_data.1, a_mask.1, a_dims.1
  auto size = block->inputs().size();
  for(size_t i = 0; i < size; i++){
    auto input = block->inputs()[i];
    auto name = input->uniqueName();
    auto names = get_name(name);
    if(names.size() == 1){
      res_block->addInput(name + "_data");
      res_block->addInput(name + "_mask");
      res_block->addInput(name + "_dims");
    }
    else{
      auto base_name = names[0];
      auto suffix = names[1];
      res_block->addInput(base_name + "_data." + suffix);
      res_block->addInput(base_name + "_mask." + suffix);
      res_block->addInput(base_name + "_dims." + suffix);
    }
    batch_map[input] = std::vector<Value*>(res_block->inputs().slice(i * 3, 3));
    var_map[names[0]] = input;
  }

  for (auto it = block->nodes().begin(); it != block->nodes().end(); it++) {
    auto n = *it;
    if(n->kind().is_aten()){
      visitAten(n, block, res_block, var_map);
    }
    else if(n->kind().is_prim()){
      if(n->kind() == prim::Constant){
        visitConstant(n, block, res_block);
      }
      else if(n->kind() == prim::NumToTensor){
        visitNumToTensor(n, block, res_block);
      }
      else if(n->kind() == prim::If){
        visitIf(n, block, res_block, var_map);
      }
      else if(n->kind() == prim::Loop){
        visitLoop(n, block, res_block, var_map);
      }
      else{
        throw std::runtime_error("NYI: node of prim kind other than [Constant, NumToTensor, If, Loop] is not supported yet");
      }
    }
    else{
      throw std::runtime_error("NYI: node that is not aten or prim kind is not supported yet");
    }
  }
  // change outputs of block - expand tensor to batchtensor(data, mask, dims)
  // for block in prim::Loop, register outputs separately
  if(!block->owningNode() || block->owningNode()->kind() != prim::Loop) {
    for(Value* output : block->outputs()){
      auto r_output = batch_map.at(output);
      res_block->registerOutput(r_output[0]);
      res_block->registerOutput(r_output[1]);
      res_block->registerOutput(r_output[2]);
    }
  }
}

std::shared_ptr<Graph> to_batch_graph(std::shared_ptr<Graph>& graph){
  // std::cout<<graph->toString()<<std::endl;
  auto res_graph = std::make_shared<Graph>(graph->scope_root());
  ToBatch to_batch;
  std::unordered_map<std::string, Value*> var_map;
  to_batch.toBatch(graph->block(), res_graph->block(), var_map);
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
