#include "torch/csrc/jit/passes/to_batch.h"
#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/tensor_conversions.h"

namespace torch { namespace jit {

std::unordered_map<std::string, std::shared_ptr<Graph>> ToBatch::batch_operator_table;

// replace aten operator node with BatchTensor operator graph
void ToBatch::visitAten(Node* n, Block* block, Block* res_block, std::unordered_map<std::string, Value*>& var_map){
  if(n->outputs().size() > 1){
    throw std::runtime_error("NYI: multiple assignment is not supported yet");
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
  auto name_base = n->output()->getNameBaseSuffix(n->output()->uniqueName())[0];
  if(var_map.find(name_base) != var_map.end()){
    std::vector<Value*> inputs = batch_map.at(var_map.at(name_base));
    inputs.insert(inputs.end(), outputs.begin(), outputs.end());
    outputs = script::inlineCallTo(*res_block->owningGraph(), *batch_operator_table.at("update"), inputs);
  }
  // Assume all outputs from inlined operator implementation are in the triple form.
  for(size_t i = 0; i < n->outputs().size(); i++){
    auto output = n->outputs()[i];
    batch_map[output] = std::vector<Value*>(outputs.begin() + i * EXP_BTENSOR_SIZE, outputs.begin() + i * EXP_BTENSOR_SIZE + EXP_BTENSOR_SIZE);
    if(output->hasUniqueName()){
      var_map[output->getNameBaseSuffix(output->uniqueName())[0]] = output;
    }
  }
}

// clone prim::Constant to new graph
// batching transformation is applied to the output of prim::NumToTensor.
// If there is a prim::NumToTensor following prim::Constant, it will be finally transformed to BatchTensor.
void ToBatch::visitConstant(Node* n, Block* block, Block* res_block){
  auto res_graph = res_block->owningGraph();
  auto* r_node = res_graph->createClone(n, rn_fn);
  r_node->setStage(n->stage());
  res_graph->appendNode(r_node);
  rn_env[n->output()] = r_node->output();
}

// change return tensor to expanded batched tensor, eg: {data, mask, dims}
void ToBatch::visitNumToTensor(Node* n, Block* block, Block* res_block){
  auto res_graph = res_block->owningGraph();
  auto* r_node = res_graph->createClone(n, rn_fn);
  r_node->setStage(n->stage());
  res_graph->appendNode(r_node);
  auto outputs = script::inlineCallTo(*res_block->owningGraph(), *batch_operator_table.at("batch_from_scalar_tensor"), r_node->outputs());
  batch_map[n->output()] = outputs;
}

// prim::If transformation:
// elif is not supported
// assume every variable assigned in an if statement is already defined before
//
// transformation example:
// @torch.jit.batch(batch_size=4)
// def batch_if(a, b):
//     if a > b:
//         a += b
//     else:
//         a -= b
//     return a
//
// original graph:
// graph(%a.1 : Dynamic
//       %b : Dynamic) {
//   %2 : Dynamic = aten::gt(%a.1, %b)
//   %a : Dynamic = prim::If(%2)
//     block0() {
//       %a.2 : Dynamic = aten::add[alpha={1}](%a.1, %b)
//       -> (%a.2)
//     }
//     block1() {
//       %a.3 : Dynamic = aten::sub[alpha={1}](%a.1, %b)
//       -> (%a.3)
//     }
//   return (%a);
// }
//
// transformed graph:
// graph(%a_data.1 : Dynamic
//       %a_mask.1 : Dynamic
//       %a_dims.1 : Dynamic
//       %b_data : Dynamic
//       %b_mask : Dynamic
//       %b_dims : Dynamic) {
//   %6 : Dynamic = aten::gt(%a_data.1, %b_data)
//   %7 : Dynamic = aten::mul(%a_mask.1, %b_mask)
//   %8 : Dynamic = aten::__or__(%a_dims.1, %b_dims)
//   %9 : Dynamic = aten::mul(%6, %7)
//   %10 : Dynamic = aten::sum(%9)
//   %11 : Dynamic = aten::gt[other={0}](%10)  // cond_any
//   %a.1 : Dynamic, %17 : Dynamic, %18 : Dynamic = prim::If(%11)
//     block0() {
//       %data.1 : Dynamic = aten::add[alpha={1}](%a_data.1, %b_data)
//       %mask.1 : Dynamic = aten::mul(%a_mask.1, %b_mask)
//       %dims.1 : Dynamic = aten::__or__(%a_dims.1, %b_dims)
//       %data.2 : Dynamic = aten::where(%mask.1, %data.1, %a_data.1)
//       -> (%data.2, %mask.1, %dims.1)
//     }
//     block1() {
//       -> (%a_data.1, %a_mask.1, %a_dims.1)
//     }
//   %19 : Dynamic = aten::zeros_like(%6)
//   %data.3 : Dynamic = aten::eq(%6, %19)
//   %21 : Dynamic = aten::mul(%data.3, %7)
//   %22 : Dynamic = aten::sum(%21)
//   %23 : Dynamic = aten::gt[other={0}](%22)  // else_cond_any
//   %a : Dynamic, %29 : Dynamic, %30 : Dynamic = prim::If(%23)
//     block0() {
//       %data.4 : Dynamic = aten::sub[alpha={1}](%a_data.1, %b_data)
//       %mask : Dynamic = aten::mul(%a_mask.1, %b_mask)
//       %dims : Dynamic = aten::__or__(%a_dims.1, %b_dims)
//       %data : Dynamic = aten::where(%mask, %data.4, %a_data.1)
//       -> (%data, %mask, %dims)
//     }
//     block1() {
//       -> (%a_data.1, %a_mask.1, %a_dims.1)
//     }
//   %res_data : Dynamic = aten::where(%6, %a.1, %a)  // combine results from two if nodes
//   %res_mask : Dynamic = aten::where(%6, %17, %29)
//   %res_dims : Dynamic = aten::__or__(%18, %30)
//   return (%res_data, %res_mask, %res_dims);
// }
void ToBatch::visitIf(Node* n, Block* block, Block* res_block, std::unordered_map<std::string, Value*>& var_map){
  auto res_graph = res_block->owningGraph();

  // create prim::If node for res_block
  auto add_if_node = [this, &res_block, &res_graph, &n, &var_map](Block* block, std::shared_ptr<Graph> cond_graph, std::vector<Value*> cond, std::vector<Value*> unchanged_outputs){
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
      auto output = r_node->outputs()[i * EXP_BTENSOR_SIZE];
      for(size_t j = 1; j < EXP_BTENSOR_SIZE; j++){
        r_node->insertOutput(i * EXP_BTENSOR_SIZE + j)->setType(output->type());
      }
    }
    return r_node;
  };

  auto cond = batch_map.at(n->input());
  std::vector<Value*> unchanged_outputs; // used to register outputs in else_block
  for(Value* output : n->outputs()){
    output = var_map.at(output->getNameBaseSuffix(output->uniqueName())[0]);
    for(Value* else_output : batch_map.at(output)){
      unchanged_outputs.push_back(else_output);
    }
  }
  auto if_node = add_if_node(n->blocks()[0], batch_operator_table.at("any"), cond, unchanged_outputs);
  auto else_node = add_if_node(n->blocks()[1], batch_operator_table.at("any_false"), cond, unchanged_outputs);

  // combine results from two if nodes
  for(size_t i = 0; i < n->outputs().size(); i++){
    std::vector<Value*> inputs(cond);
    for(size_t j = 0; j < EXP_BTENSOR_SIZE; j++){
      inputs.push_back(if_node->outputs()[i * EXP_BTENSOR_SIZE + j]);
    }
    for(size_t j = 0; j < EXP_BTENSOR_SIZE; j++){
      inputs.push_back(else_node->outputs()[i * EXP_BTENSOR_SIZE + j]);
    }
    auto outputs = script::inlineCallTo(*res_block->owningGraph(), *batch_operator_table.at("where"), inputs);
    batch_map[n->outputs()[i]] = outputs;
  }
}

// prim::Loop transformation:
//
// transformation example:
// @torch.jit.batch(batch_size=4)
// def batch_while(a, b):
//     while a > b:
//         a -= b
//     return a
//
// original graph:
// graph(%a.1 : Dynamic
//       %b : Dynamic) {
//   %2 : int = prim::Constant[value={2147483647}]()
//   %3 : Dynamic = aten::gt(%a.1, %b)
//   %a : Dynamic = prim::Loop(%2, %3, %a.1)
//     block0(%4 : Dynamic, %5 : Dynamic) {
//       %a.2 : Dynamic = aten::sub[alpha={1}](%5, %b)
//       %9 : Dynamic = aten::gt(%a.2, %b)
//       -> (%9, %a.2)
//     }
//   return (%a);
// }
//
// transformed graph:
// graph(%a_data.1 : Dynamic
//       %a_mask.1 : Dynamic
//       %a_dims.1 : Dynamic
//       %b_data : Dynamic
//       %b_mask : Dynamic
//       %b_dims : Dynamic) {
//   %6 : int = prim::Constant[value={2147483647}]()
//   %7 : Dynamic = aten::gt(%a_data.1, %b_data)
//   %8 : Dynamic = aten::mul(%a_mask.1, %b_mask)
//   %9 : Dynamic = aten::__or__(%a_dims.1, %b_dims)
//   %10 : Dynamic = aten::mul(%7, %8)
//   %11 : Dynamic = aten::sum(%10)
//   %12 : Dynamic = aten::gt[other={0}](%11)  // cond_any
//   %38 : Dynamic, %39 : Dynamic, %40 : Dynamic, %a : Dynamic, %36 : Dynamic, %37 : Dynamic = prim::Loop(%6, %12, %7, %8, %9, %a_data.1, %a_mask.1, %a_dims.1)
//     block0(%4_data : Dynamic, %cond_data : Dynamic, %cond_mask : Dynamic, %cond_dims : Dynamic, %5_data : Dynamic, %5_mask : Dynamic, %5_dims : Dynamic) {
//       %data.1 : Dynamic = aten::sub[alpha={1}](%5_data, %b_data)
//       %mask : Dynamic = aten::mul(%5_mask, %b_mask)
//       %dims : Dynamic = aten::__or__(%5_dims, %b_dims)
//       %data : Dynamic = aten::where(%mask, %data.1, %a_data.1)
//       %24 : Dynamic = aten::gt(%data, %b_data)  // new cond
//       %25 : Dynamic = aten::mul(%mask, %b_mask)
//       %26 : Dynamic = aten::__or__(%dims, %b_dims)
//       %res_data : Dynamic = aten::where(%cond_data, %data, %5_data) // update outputs
//       %res_mask : Dynamic = aten::where(%cond_data, %mask, %5_mask)
//       %res_dims : Dynamic = aten::__or__(%dims, %5_dims)
//       %33 : Dynamic = aten::mul(%24, %25)
//       %34 : Dynamic = aten::sum(%33)
//       %35 : Dynamic = aten::gt[other={0}](%34)  // new cond_any
//       -> (%35, %24, %25, %26, %res_data, %res_mask, %res_dims)
//     }
//   return (%a, %36, %37);
// }
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
  for(size_t i = 0; i < EXP_BTENSOR_SIZE; i++){
    r_node->insertInput(i + 2, cond[i]);
  }
  for(size_t i = 2; i < n->inputs().size(); i++){
    for(size_t j = 1; j < EXP_BTENSOR_SIZE; j++){
      r_node->insertInput((i - 2) * EXP_BTENSOR_SIZE + EXP_BTENSOR_SIZE + 2 + j, batch_map.at(n->inputs()[i])[j]);
    }
  }
  r_node->setStage(n->stage());
  res_graph->appendNode(r_node);

  // create block for Loop node in res_block
  // first 4 inputs of block: first cond_any, cond_data, cond_mask, cond_dims
  auto loop_block = r_node->addBlock();
  toBatch(n->blocks()[0], loop_block, var_map);

  // change inputs and outputs of block[0] in prim::Loop
  for(size_t i = EXP_BTENSOR_SIZE - 1; i > 0; i--){
    loop_block->eraseInput(i);
  }
  for(size_t i = 0; i < EXP_BTENSOR_SIZE; i++){
    loop_block->insertInput(1, "cond_" + EXP_BTENSOR_NAME[i]);
  }

  WithInsertPoint guard(loop_block);

  // use where operator to update variables
  for(size_t i = 0; i < n->outputs().size(); i++){
    std::vector<Value*> inputs;
    for(size_t j = 0; j < EXP_BTENSOR_SIZE; j++){
      inputs.push_back(loop_block->inputs()[j + 1]);
    }
    auto data = batch_map.at(n->blocks()[0]->outputs()[i + 1]);
    inputs.insert(inputs.end(), data.begin(), data.end());
    for(size_t j = 0; j < EXP_BTENSOR_SIZE; j++){
      inputs.push_back(loop_block->inputs()[i * EXP_BTENSOR_SIZE + j + EXP_BTENSOR_SIZE + 1]);
    }
    auto outputs = script::inlineCallTo(*res_block->owningGraph(), *batch_operator_table.at("where"), inputs);
    batch_map[n->outputs()[i]] = outputs;
    for(size_t j = 0; j < EXP_BTENSOR_SIZE; j++){
      loop_block->registerOutput(outputs[j]);
    }
  }

  // update loop conditions
  cond = batch_map.at(n->blocks()[0]->outputs()[0]);
  cond_any = script::inlineCallTo(*res_block->owningGraph(), *batch_operator_table.at("any"), cond);
  loop_block->insertOutput(0, cond_any[0]);
  for(size_t i = 0; i < EXP_BTENSOR_SIZE; i++){
    loop_block->insertOutput(i + 1, cond[i]);
  }

  // change outputs of prim::Loop
  auto size = r_node->outputs().size();
  for(size_t i = 0; i < size; i++){
    for(size_t j = 1; j < EXP_BTENSOR_SIZE; j++){
      r_node->insertOutput(i * EXP_BTENSOR_SIZE + j);
    }
    batch_map[n->outputs()[i]] = r_node->outputs().slice(i * EXP_BTENSOR_SIZE, EXP_BTENSOR_SIZE);
  }
  for(size_t i = 0; i < EXP_BTENSOR_SIZE; i++){
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

  // change inputs of block - expand tensor to batchtensor eg: (data, mask, dims)
  // eg: a -> a_data, a_mask, a_dims
  // eg: a.1 -> a_data.1, a_mask.1, a_dims.1
  auto size = block->inputs().size();
  for(size_t i = 0; i < size; i++){
    auto input = block->inputs()[i];
    auto name = input->uniqueName();
    auto names = input->getNameBaseSuffix(name);
    if(names.size() == 1){
      for(size_t j = 0; j < EXP_BTENSOR_SIZE; j++){
        res_block->addInput(name + "_" + EXP_BTENSOR_NAME[j]);
      }
    }
    else{
      auto base_name = names[0];
      auto suffix = names[1];
      for(size_t j = 0; j < EXP_BTENSOR_SIZE; j++){
        res_block->addInput(base_name + "_" + EXP_BTENSOR_NAME[j] + "." + suffix);
      }
    }
    batch_map[input] = std::vector<Value*>(res_block->inputs().slice(i * EXP_BTENSOR_SIZE, EXP_BTENSOR_SIZE));
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
      for(size_t i = 0; i < EXP_BTENSOR_SIZE; i++){
        res_block->registerOutput(r_output[i]);
      }
    }
  }
}

std::shared_ptr<Graph> to_batch_graph(std::shared_ptr<Graph>& graph){
  // std::cout<<graph->toString()<<std::endl;
  std::shared_ptr<Graph> res_graph = std::make_shared<Graph>(graph->scope_root());
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
