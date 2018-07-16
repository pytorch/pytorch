#include "torch/csrc/jit/passes/to_batch.h"
#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/tensor_conversions.h"

namespace torch { namespace jit {

std::unordered_map<std::string, std::shared_ptr<Graph>> ToBatch::batch_operator_table;

// replace aten operator node with BatchTensor operator graph
void ToBatch::visitAten(Node* n, Block* block, Block* res_block){
  auto res_graph = res_block->owningGraph();
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

  // assume in batched operators, all attr inputs are in the end
  for(Symbol attr : n->attributeNames()){
    // std::cout << toString(n->kindOf(attr)) << std::endl;
    at::Tensor t;
    switch (n->kindOf(attr)) {
      case AttributeKind::f:
        t = at::tensor(n->f(attr), at::kFloat);
        break;
      case AttributeKind::i:
        t = at::tensor(n->i(attr), at::kInt);
        break;
      case AttributeKind::t:
        t = n->t(attr);
        break;
      default:
        throw std::runtime_error("NYI: Attribute kind except for f, i, t is not supported yet");
    }
    auto attr_node = res_graph->createConstant(t);
    attr_node->output()->inferTypeFrom(t);
    res_block->appendNode(attr_node);
    new_inputs.push_back(attr_node->output());
  }
  auto outputs = script::inlineCallTo(*res_block->owningGraph(), *batch_graph, new_inputs);

  // Assume all outputs from inlined operator implementation are in the triple form.
  for(size_t i = 0; i < n->outputs().size(); i++){
    auto output = n->outputs()[i];
    batch_map[output] = std::vector<Value*>(outputs.begin() + i * EXP_BTENSOR_SIZE, outputs.begin() + i * EXP_BTENSOR_SIZE + EXP_BTENSOR_SIZE);
  }
}

// clone prim::Constant to new graph
// batching transformation is applied to the output of prim::NumToTensor.
// If there is a prim::NumToTensor following prim::Constant, it will be finally transformed to BatchTensor.
void ToBatch::visitConstant(Node* n, Block* block, Block* res_block){
  auto res_graph = res_block->owningGraph();
  auto* r_node = res_graph->createClone(n, rn_fn);
  r_node->setStage(n->stage());
  res_block->appendNode(r_node);
  rn_env[n->output()] = r_node->output();
}

// change return tensor to expanded batched tensor, eg: {data, mask, dims}
void ToBatch::visitNumToTensor(Node* n, Block* block, Block* res_block){
  auto res_graph = res_block->owningGraph();
  auto* r_node = res_graph->createClone(n, rn_fn);
  r_node->setStage(n->stage());
  res_block->appendNode(r_node);
  auto outputs = script::inlineCallTo(*res_block->owningGraph(), *batch_operator_table.at("batch_from_scalar_tensor"), r_node->outputs());
  batch_map[n->output()] = outputs;
}

// prim::If transformation:
// elif is not supported
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
// graph(%a.1_data : Dynamic
//       %a.1_mask : Dynamic
//       %a.1_dims : Dynamic
//       %b_data : Dynamic
//       %b_mask : Dynamic
//       %b_dims : Dynamic) {
//   %6 : Dynamic = aten::gt(%a.1_data, %b_data)  // calculate condition
//   %7 : Dynamic = aten::mul(%a.1_mask, %b_mask)
//   %8 : Dynamic = aten::__or__(%a.1_dims, %b_dims)
//   %9 : Dynamic = aten::mul(%6, %7)
//   %10 : Dynamic = aten::sum(%9)
//   %11 : Dynamic = aten::gt[other={0}](%10)
//   %12 : Long() = prim::Constant[value={1}]()  // if_block
//   %13 : Number = prim::TensorToNum(%12)
//   %data.1 : Dynamic = aten::add(%a.1_data, %b_data, %13)
//   %mask.1 : Dynamic = aten::mul(%a.1_mask, %b_mask)
//   %dims.1 : Dynamic = aten::__or__(%a.1_dims, %b_dims)
//   %17 : Long() = prim::Constant[value={1}]()  // else_block
//   %18 : Number = prim::TensorToNum(%17)
//   %data : Dynamic = aten::sub(%a.1_data, %b_data, %18)
//   %mask : Dynamic = aten::mul(%a.1_mask, %b_mask)
//   %dims : Dynamic = aten::__or__(%a.1_dims, %b_dims)
//   %res_data : Dynamic = aten::where(%6, %data.1, %data)  // combine two outputs
//   %res_mask : Dynamic = aten::where(%6, %mask.1, %mask)
//   %res_dims : Dynamic = aten::__or__(%dims.1, %dims)
//   return (%res_data, %res_mask, %res_dims);
// }
void ToBatch::visitIf(Node* n, Block* block, Block* res_block){
  auto cond = batch_map.at(n->input());
  auto cond_any = script::inlineCallTo(*res_block->owningGraph(), *batch_operator_table.at("any"), cond);
  toBatch(n->blocks()[0], res_block);
  toBatch(n->blocks()[1], res_block);

  // combine results from two if paths
  for(size_t i = 0; i < n->outputs().size(); i++){
    std::vector<Value*> inputs;
    inputs.insert(inputs.end(), cond.begin(), cond.end());
    auto if_output = batch_map.at(n->blocks()[0]->outputs()[i]);
    inputs.insert(inputs.end(), if_output.begin(), if_output.end());
    auto else_output = batch_map.at(n->blocks()[1]->outputs()[i]);
    inputs.insert(inputs.end(), else_output.begin(), else_output.end());
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
// graph(%a.1_data : Dynamic
//       %a.1_mask : Dynamic
//       %a.1_dims : Dynamic
//       %b_data : Dynamic
//       %b_mask : Dynamic
//       %b_dims : Dynamic) {
//   %6 : int = prim::Constant[value={2147483647}]()
//   %7 : Dynamic = aten::gt(%a.1_data, %b_data)
//   %8 : Dynamic = aten::mul(%a.1_mask, %b_mask)
//   %9 : Dynamic = aten::__or__(%a.1_dims, %b_dims)
//   %10 : Dynamic = aten::mul(%7, %8)
//   %11 : Dynamic = aten::sum(%10)
//   %12 : Dynamic = aten::gt[other={0}](%11)  // cond_any
//   %39 : Dynamic, %40 : Dynamic, %41 : Dynamic, %a : Dynamic, %37 : Dynamic, %38 : Dynamic = prim::Loop(%6, %12, %7, %8, %9, %a.1_data, %a.1_mask, %a.1_dims)
//     block0(%4_data : Dynamic, %cond_data : Dynamic, %cond_mask : Dynamic, %cond_dims : Dynamic, %5_data : Dynamic, %5_mask : Dynamic, %5_dims : Dynamic) {
//       %20 : Long() = prim::Constant[value={1}]()
//       %21 : Number = prim::TensorToNum(%20)
//       %data : Dynamic = aten::sub(%5_data, %b_data, %21)
//       %mask : Dynamic = aten::mul(%5_mask, %b_mask)
//       %dims : Dynamic = aten::__or__(%5_dims, %b_dims)
//       %25 : Dynamic = aten::gt(%data, %b_data)  // new cond
//       %26 : Dynamic = aten::mul(%mask, %b_mask)
//       %27 : Dynamic = aten::__or__(%dims, %b_dims)
//       %res_data : Dynamic = aten::where(%cond_data, %data, %5_data) // update outputs
//       %res_mask : Dynamic = aten::where(%cond_data, %mask, %5_mask)
//       %res_dims : Dynamic = aten::__or__(%dims, %5_dims)
//       %34 : Dynamic = aten::mul(%25, %26)
//       %35 : Dynamic = aten::sum(%34)
//       %36 : Dynamic = aten::gt[other={0}](%35)
//       -> (%36, %25, %26, %27, %res_data, %res_mask, %res_dims)
//     }
//   return (%a, %37, %38);
// }
void ToBatch::visitLoop(Node* n, Block* block, Block* res_block){
  auto res_graph = res_block->owningGraph();
  // bool cond_is_tensor indicates whether cond is tensor
  // cond_is_tensor = false, eg: for loop, n->inputs()[1] = byte()
  // cond_is_tensor = true, eg: in some while loop, cond is a batched tensor,
  //                            we need to add expanded cond to the inputs of loop node and block,
  //                            and compute cond_any as cond for while loop
  bool cond_is_tensor = (batch_map.find(n->inputs()[1]) != batch_map.end());
  // create prim::Loop node for res_block
  if(cond_is_tensor){
    auto cond = batch_map.at(n->inputs()[1]);
    auto cond_any = script::inlineCallTo(*res_block->owningGraph(), *batch_operator_table.at("any"), cond);
    rn_env[n->inputs()[1]] = cond_any[0];
  }
  for(size_t i = 2; i < n->inputs().size(); i++){
    auto input = n->inputs()[i];
    rn_env[input] = batch_map.at(input)[0];
  }
  auto* r_node = res_graph->createClone(n, rn_fn, /*copy_blocks=*/false);

  // change inputs of prim::Loop
  if(cond_is_tensor){
    for(size_t i = 0; i < EXP_BTENSOR_SIZE; i++){
      auto cond = batch_map.at(n->inputs()[1]);
      r_node->insertInput(i + 2, cond[i]);
    }
  }
  for(size_t i = 2; i < n->inputs().size(); i++){
    for(size_t j = 1; j < EXP_BTENSOR_SIZE; j++){
      r_node->insertInput((i - 2) * EXP_BTENSOR_SIZE + EXP_BTENSOR_SIZE * cond_is_tensor + 2 + j, batch_map.at(n->inputs()[i])[j]);
    }
  }
  r_node->setStage(n->stage());
  res_block->appendNode(r_node);

  // create block for Loop node in res_block
  // if cond is tensor:    first 4 inputs of block: cond_any, cond_data, cond_mask, cond_dims
  // if cond is not tensor: first 1 input of block: cond
  auto loop_block = r_node->addBlock();
  toBatch(n->blocks()[0], loop_block);

  // change inputs and outputs of block[0] in prim::Loop
  for(size_t i = EXP_BTENSOR_SIZE - 1; i > 0; i--){
    loop_block->eraseInput(i);
  }
  if(cond_is_tensor){
    for(size_t i = 0; i < EXP_BTENSOR_SIZE; i++){
      loop_block->insertInput(i + 1, "cond_" + EXP_BTENSOR_NAME[i]);
    }
  }

  WithInsertPoint guard(loop_block);

  // use where operator to update variables and add to outputs
  if(cond_is_tensor){
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
  }
  // add outputs
  else{
    for(size_t i = 0; i < n->outputs().size(); i++){
      auto outputs = batch_map.at(n->blocks()[0]->outputs()[i + 1]);
      for(size_t j = 0; j < EXP_BTENSOR_SIZE; j++){
        loop_block->registerOutput(outputs[j]);
      }
    }
  }

  // update loop conditions
  if(cond_is_tensor){
    auto cond = batch_map.at(n->blocks()[0]->outputs()[0]);
    auto cond_any = script::inlineCallTo(*res_block->owningGraph(), *batch_operator_table.at("any"), cond);
    loop_block->insertOutput(0, cond_any[0]);
    for(size_t i = 0; i < EXP_BTENSOR_SIZE; i++){
      loop_block->insertOutput(i + 1, cond[i]);
    }
  }
  else{
    auto cond = rn_env.at(n->blocks()[0]->outputs()[0]);
    loop_block->insertOutput(0, cond);
  }

  // change outputs of prim::Loop
  auto size = r_node->outputs().size();
  for(size_t i = 0; i < size; i++){
    for(size_t j = 1; j < EXP_BTENSOR_SIZE; j++){
      r_node->insertOutput(i * EXP_BTENSOR_SIZE + j);
    }
    batch_map[n->outputs()[i]] = r_node->outputs().slice(i * EXP_BTENSOR_SIZE, EXP_BTENSOR_SIZE);
  }
  // add cond to outputs of loop node
  if(cond_is_tensor){
    for(size_t i = 0; i < EXP_BTENSOR_SIZE; i++){
      r_node->insertOutput(i);
    }
  }
}

void ToBatch::toBatch(Block* block, Block* res_block) {
  WithInsertPoint guard(res_block);

  // change inputs of block - expand tensor to batchtensor eg: (data, mask, dims)
  // eg: a -> a_data, a_mask, a_dims
  auto size = block->inputs().size();
  for(size_t i = 0; i < size; i++){
    auto input = block->inputs()[i];
    auto name = input->uniqueName();
    for(size_t j = 0; j < EXP_BTENSOR_SIZE; j++){
      res_block->addInput(name + "_" + EXP_BTENSOR_NAME[j]);
    }
    batch_map[input] = std::vector<Value*>(res_block->inputs().slice(i * EXP_BTENSOR_SIZE, EXP_BTENSOR_SIZE));
  }

  for (auto it = block->nodes().begin(); it != block->nodes().end(); it++) {
    auto n = *it;
    if(n->kind().is_aten()){
      visitAten(n, block, res_block);
    }
    else if(n->kind().is_prim()){
      switch(n->kind()){
        case prim::Constant:
          visitConstant(n, block, res_block);
          break;
        case prim::NumToTensor:
          visitNumToTensor(n, block, res_block);
          break;
        case prim::If:
          visitIf(n, block, res_block);
          break;
        case prim::Loop:
          visitLoop(n, block, res_block);
          break;
        default:
          throw std::runtime_error("NYI: node of prim kind other than [Constant, NumToTensor, If, Loop] is not supported yet");
      }
    }
    else{
      throw std::runtime_error("NYI: node that is not aten or prim kind is not supported yet");
    }
  }
  // change outputs of block - expand tensor to batchtensor(data, mask, dims)
  // for block in prim::Loop, register outputs separately to deal with cond and cond_any
  // for block in prim::If, register outputs separately by combining outputs from two paths and return
  if(!block->owningNode() || (block->owningNode()->kind() != prim::Loop && block->owningNode()->kind() != prim::If)) {
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
