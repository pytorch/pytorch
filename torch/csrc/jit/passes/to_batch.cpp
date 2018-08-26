#include "torch/csrc/jit/passes/to_batch.h"
#include "torch/csrc/jit/script/compiler.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch { namespace jit {

std::unordered_map<std::string, std::vector<std::shared_ptr<Graph>>> ToBatch::batch_operator_table;

std::shared_ptr<Graph> ToBatch::getBatchOperator(std::string name, int64_t num_inputs){
  if(batch_operator_table.find(name) == batch_operator_table.end()){
    throw std::runtime_error("function " + name + " is not supported in batched tensor yet");
  }
  auto ops = batch_operator_table.at(name);
  if(num_inputs == -1)  // default function
    return ops[0];
  for(auto op : ops){
    if(size_t(num_inputs) == op->inputs().size())
      return op;
  }
  throw std::runtime_error("function " + name + " with " + std::to_string(num_inputs) + " inputs is not supported in batched tensor yet");
}

// replace aten operator node with BatchTensor operator graph
void ToBatch::visitAten(Node* n, Block* block, Block* res_block){
  auto res_graph = res_block->owningGraph();
  auto func_name = std::string(n->kind().toUnqualString());
  std::vector<Value*> new_inputs;
  for(Value *input : n->inputs()){
    if(rn_env.find(input) == rn_env.end()){  // non-tensor input
      auto new_input = batch_map.at(input);
      new_inputs.insert(new_inputs.end(), new_input.begin(), new_input.end());
    }
    else{  // batched tensor input
      new_inputs.push_back(rn_env.at(input));
    }
  }

  // transform scalar to tensor before pass to batch operator script
  for(size_t i = 0; i < new_inputs.size(); i++){
    auto input = new_inputs[i];
    if(input->type() == IntType::get() || input->type() == FloatType::get()){
      auto to_tensor_node = res_graph->createNumToTensor(input);
      res_graph->insertNode(to_tensor_node);
      new_inputs[i] = to_tensor_node->output();
    }
  }

  auto batch_graph = getBatchOperator(func_name, new_inputs.size());
  auto outputs = script::inlineCallTo(*res_block->owningGraph(), *batch_graph, new_inputs);

  // Assume all outputs from inlined operator implementation are in the triple form batched tensor or just a single non-tensor.
  if(outputs.size() == 1){
    // if previous output is scalar, transform new output back to scalar from dynamic
    if(n->outputs()[0]->type() != outputs[0]->type()){
      Node* to_scalar_node;
      if(n->outputs()[0]->type() == IntType::get()){
        to_scalar_node = res_graph->createTensorToNum(IntType::get(), outputs[0]);
      }
      else if(n->outputs()[0]->type() == FloatType::get()){
        to_scalar_node = res_graph->createTensorToNum(FloatType::get(), outputs[0]);
      }
      else{
        throw std::runtime_error("NYI: scalar type other than int, float is not supported yet");
      }
      res_graph->insertNode(to_scalar_node);
      rn_env[n->outputs()[0]] = to_scalar_node->output();
    }
    else
      rn_env[n->outputs()[0]] = outputs[0];
  }
  else{
    for(size_t i = 0; i < n->outputs().size(); i++){
      auto output = n->outputs()[i];
      batch_map[output] = std::vector<Value*>(outputs.begin() + i * EXP_BTENSOR_SIZE, outputs.begin() + i * EXP_BTENSOR_SIZE + EXP_BTENSOR_SIZE);
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
  res_block->appendNode(r_node);
  rn_env[n->output()] = r_node->output();
}

// change return tensor to expanded batched tensor, eg: {data, mask, dims}
void ToBatch::visitNumToTensor(Node* n, Block* block, Block* res_block){
  auto res_graph = res_block->owningGraph();
  auto* r_node = res_graph->createClone(n, rn_fn);
  r_node->setStage(n->stage());
  res_block->appendNode(r_node);
  auto outputs = script::inlineCallTo(*res_block->owningGraph(), *getBatchOperator("batch_from_scalar_tensor"), r_node->outputs());
  batch_map[n->output()] = outputs;
}

// clone prim::TensorToNum to new graph
void ToBatch::visitTensorToNum(Node* n, Block* block, Block* res_block){
  auto res_graph = res_block->owningGraph();
  if(rn_env.find(n->input()) == rn_env.end()){
    rn_env[n->input()] = batch_map.at(n->input())[0];
  }
  auto* r_node = res_graph->createClone(n, rn_fn);
  r_node->setStage(n->stage());
  res_block->appendNode(r_node);
  rn_env[n->output()] = r_node->output();
  batch_map[n->output()] = batch_map.at(n->input());
}

// clone prim::ListConstruct to new graph
void ToBatch::visitListConstruct(Node* n, Block* block, Block* res_block){
  auto res_graph = res_block->owningGraph();
  if(n->inputs()[0]->type() == DynamicType::get()){  // TensorList: expand directly
    std::vector<Value*> inputs;
    for(Value* input: n->inputs()) {
      auto res = batch_map.at(input);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }
    batch_map[n->output()] = inputs;
  }
  else {  // ScalarList: transform to tensor, then transform back
    for(Value* input : n->inputs()) {
      if(rn_env.find(input) == rn_env.end()){
        rn_env[input] = batch_map.at(input)[0];
      }
    }
    auto* r_node = res_graph->createClone(n, rn_fn);
    r_node->setStage(n->stage());
    res_block->appendNode(r_node);
    // transform int[] to tensor
    auto to_tensor_node = res_graph->create(Symbol::fromQualString("aten::_list_to_tensor"));
    to_tensor_node->setStage(n->stage());
    to_tensor_node->addInput(r_node->output());
    res_block->appendNode(to_tensor_node);
    rn_env[n->output()] = to_tensor_node->output();
  }
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
//   %9 : int = prim::TensorToNum(%6)
//   %10 : Long() = prim::Constant[value={1}]()  // if_block
//   %alpha.1 : float = prim::TensorToNum(%10)
//   %data.1 : Dynamic = aten::add(%a.1_data, %b_data, %alpha.1)
//   %mask.1 : Dynamic = aten::mul(%a.1_mask, %b_mask)
//   %dims.1 : Dynamic = aten::__or__(%a.1_dims, %b_dims)
//   %15 : Long() = prim::Constant[value={1}]()  // else_block
//   %alpha : float = prim::TensorToNum(%15)
//   %data.4 : Dynamic = aten::sub(%a.1_data, %b_data, %alpha)
//   %mask : Dynamic = aten::mul(%a.1_mask, %b_mask)
//   %dims : Dynamic = aten::__or__(%a.1_dims, %b_dims)
//   %20 : Dynamic = aten::type_as(%7, %6)   // combine two outputs (batch_where)
//   %cond_mask.1 : Dynamic = aten::mul(%6, %20)
//   %22 : int = aten::dim(%cond_mask.1)
//   %23 : int = prim::Constant[value=1]()
//   %24 : int = aten::eq(%22, %23)
//   %cond_data : Dynamic, %cond_mask : Dynamic, %data : Dynamic = prim::If(%24)
//     block0() {
//       %28 : int = aten::dim(%data.1)
//       %29 : int = prim::Constant[value=1]()
//       %30 : int = aten::sub(%28, %29)
//       %31 : int = prim::Constant[value=1]()
//       %data.3 : Dynamic = prim::Loop(%30, %31, %cond_mask.1)
//         block0(%_ : int, %34 : Dynamic) {
//           %35 : int = prim::Constant[value=1]()
//           %36 : int = aten::neg(%35)
//           %data.2 : Dynamic = aten::unsqueeze(%34, %36)
//           %38 : int = prim::Constant[value=1]()
//           -> (%38, %data.2)
//         }
//       %cond_data.1 : Dynamic = aten::expand_as(%data.3, %data.1)
//       %cond_mask.2 : Dynamic = aten::expand_as(%data.3, %mask.1)
//       -> (%cond_data.1, %cond_mask.2, %data.3)
//     }
//     block1() {
//       -> (%cond_mask.1, %cond_mask.1, %cond_mask.1)
//     }
//   %res_data : Dynamic = aten::where(%cond_data, %data.1, %data.4)
//   %res_mask : Dynamic = aten::where(%cond_mask, %mask.1, %mask)
//   %res_dims : Dynamic = aten::__or__(%dims.1, %dims)
//   return (%res_data, %res_mask, %res_dims);
// }
void ToBatch::visitIf(Node* n, Block* block, Block* res_block){
  toBatch(n->blocks()[0], res_block);
  toBatch(n->blocks()[1], res_block);

  // combine results from two if paths
  for(size_t i = 0; i < n->outputs().size(); i++){
    std::vector<Value*> inputs;
    if(batch_map.find(n->input()) == batch_map.end()){  // cond is scalar
      inputs.push_back(rn_env.at(n->input()));
    }
    else{  // cond is tensor
      auto cond = batch_map.at(n->input());
      inputs.insert(inputs.end(), cond.begin(), cond.end());
    }
    auto if_output = batch_map.at(n->blocks()[0]->outputs()[i]);
    inputs.insert(inputs.end(), if_output.begin(), if_output.end());
    auto else_output = batch_map.at(n->blocks()[1]->outputs()[i]);
    inputs.insert(inputs.end(), else_output.begin(), else_output.end());
    auto outputs = script::inlineCallTo(*res_block->owningGraph(), *getBatchOperator("where", inputs.size()), inputs);
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
//   %6 : int = prim::Constant[value=2147483647]()
//   %7 : Dynamic = aten::gt(%a.1_data, %b_data)
//   %8 : Dynamic = aten::mul(%a.1_mask, %b_mask)
//   %9 : Dynamic = aten::__or__(%a.1_dims, %b_dims)
//   %10 : int = prim::TensorToNum(%7)
//   %11 : Dynamic = aten::mul(%7, %8)
//   %12 : Dynamic = aten::sum(%11)
//   %13 : Dynamic = aten::gt[other={0}](%12)  // cond_any
//   %14 : int = prim::TensorToNum(%13)
//   %62 : Dynamic, %63 : Dynamic, %64 : Dynamic, %a : Dynamic, %60 : Dynamic, %61 : Dynamic = prim::Loop(%6, %14, %7, %8, %9, %a.1_data, %a.1_mask, %a.1_dims)
//     block0(%loop_num : int, %cond_data.2 : Dynamic, %cond_mask.3 : Dynamic, %cond_dims : Dynamic, %6_data : Dynamic, %6_mask : Dynamic, %6_dims : Dynamic) {
//       %23 : Long() = prim::Constant[value={1}]()
//       %alpha : float = prim::TensorToNum(%23)
//       %data.1 : Dynamic = aten::sub(%6_data, %b_data, %alpha)
//       %mask : Dynamic = aten::mul(%6_mask, %b_mask)
//       %dims : Dynamic = aten::__or__(%6_dims, %b_dims)
//       %28 : Dynamic = aten::gt(%data.1, %b_data)
//       %29 : Dynamic = aten::mul(%mask, %b_mask)
//       %30 : Dynamic = aten::__or__(%dims, %b_dims)
//       %31 : int = prim::TensorToNum(%28)
//       %32 : Dynamic = aten::type_as(%cond_mask.3, %cond_data.2)  // update outputs (batch_where)
//       %cond_mask.1 : Dynamic = aten::mul(%cond_data.2, %32)
//       %34 : int = aten::dim(%cond_mask.1)
//       %35 : int = prim::Constant[value=1]()
//       %36 : int = aten::eq(%34, %35)
//       %cond_data : Dynamic, %cond_mask : Dynamic, %data : Dynamic = prim::If(%36)
//         block0() {
//           %40 : int = aten::dim(%data.1)
//           %41 : int = prim::Constant[value=1]()
//           %42 : int = aten::sub(%40, %41)
//           %43 : int = prim::Constant[value=1]()
//           %data.3 : Dynamic = prim::Loop(%42, %43, %cond_mask.1)
//             block0(%_ : int, %46 : Dynamic) {
//               %47 : int = prim::Constant[value=1]()
//               %48 : int = aten::neg(%47)
//               %data.2 : Dynamic = aten::unsqueeze(%46, %48)
//               %50 : int = prim::Constant[value=1]()
//               -> (%50, %data.2)
//             }
//           %cond_data.1 : Dynamic = aten::expand_as(%data.3, %data.1)
//           %cond_mask.2 : Dynamic = aten::expand_as(%data.3, %mask)
//           -> (%cond_data.1, %cond_mask.2, %data.3)
//         }
//         block1() {
//           -> (%cond_mask.1, %cond_mask.1, %cond_mask.1)
//         }
//       %res_data : Dynamic = aten::where(%cond_data, %data.1, %6_data)
//       %res_mask : Dynamic = aten::where(%cond_mask, %mask, %6_mask)
//       %res_dims : Dynamic = aten::__or__(%dims, %6_dims)
//       %56 : Dynamic = aten::mul(%28, %29)
//       %57 : Dynamic = aten::sum(%56)
//       %58 : Dynamic = aten::gt[other={0}](%57)
//       %59 : int = prim::TensorToNum(%58)
//       -> (%59, %28, %29, %30, %res_data, %res_mask, %res_dims)
//     }
//   return (%a, %60, %61);
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

  // type of cond in loop should be int type
  if(rn_env.at(n->inputs()[0])->type() != IntType::get()){
    auto to_int_node = res_graph->createTensorToNum(IntType::get(), rn_env.at(n->inputs()[0]));
    res_graph->insertNode(to_int_node);
    rn_env[n->inputs()[0]] = to_int_node->output();
  }
  if(cond_is_tensor){
    auto cond = batch_map.at(n->inputs()[1]);
    auto cond_any = script::inlineCallTo(*res_block->owningGraph(), *getBatchOperator("any"), cond);
    auto to_int_node = res_graph->createTensorToNum(IntType::get(), cond_any[0]);
    res_graph->insertNode(to_int_node);
    rn_env[n->inputs()[1]] = to_int_node->output();
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

  // add inputs
  loop_block->addInput("loop_num");
  loop_block->inputs()[0]->setType(IntType::get());
  rn_env[n->blocks()[0]->inputs()[0]] = loop_block->inputs()[0];
  if(cond_is_tensor){
    for(size_t i = 0; i < EXP_BTENSOR_SIZE; i++){
      loop_block->addInput("cond_" + EXP_BTENSOR_NAME[i]);
    }
  }
  for(size_t i = 1; i < n->blocks()[0]->inputs().size(); i++){
    auto input = n->blocks()[0]->inputs()[i];
    auto name = input->uniqueName();
    for(size_t j = 0; j < EXP_BTENSOR_SIZE; j++){
      loop_block->addInput(name + "_" + EXP_BTENSOR_NAME[j]);
    }
    batch_map[input] = std::vector<Value*>(loop_block->inputs().slice((i - 1) * EXP_BTENSOR_SIZE + 1 + EXP_BTENSOR_SIZE * cond_is_tensor, EXP_BTENSOR_SIZE).vec());
  }

  toBatch(n->blocks()[0], loop_block);

  WithInsertPoint guard(loop_block);

  // use where operator to update variables and add to outputs
  for(size_t i = 0; i < n->outputs().size(); i++){
    std::vector<Value*> inputs, outputs;
    if(cond_is_tensor){
      for(size_t j = 0; j < EXP_BTENSOR_SIZE; j++){
        inputs.push_back(loop_block->inputs()[j + 1]);
      }
      auto data = batch_map.at(n->blocks()[0]->outputs()[i + 1]);
      inputs.insert(inputs.end(), data.begin(), data.end());
      for(size_t j = 0; j < EXP_BTENSOR_SIZE; j++){
        inputs.push_back(loop_block->inputs()[i * EXP_BTENSOR_SIZE + j + EXP_BTENSOR_SIZE + 1]);
      }
      outputs = script::inlineCallTo(*res_block->owningGraph(), *getBatchOperator("where"), inputs);
    }
    else{
      for(size_t j = 0; j < EXP_BTENSOR_SIZE; j++){
        inputs.push_back(loop_block->inputs()[i * EXP_BTENSOR_SIZE + j + 1]);
      }
      auto data = batch_map.at(n->blocks()[0]->outputs()[i + 1]);
      inputs.insert(inputs.end(), data.begin(), data.end());
      outputs = script::inlineCallTo(*res_block->owningGraph(), *getBatchOperator("update"), inputs);
    }
    batch_map[n->outputs()[i]] = outputs;
    for(size_t j = 0; j < EXP_BTENSOR_SIZE; j++){
      loop_block->registerOutput(outputs[j]);
    }
  }

  // update loop conditions
  if(cond_is_tensor){
    auto cond = batch_map.at(n->blocks()[0]->outputs()[0]);
    auto cond_any = script::inlineCallTo(*res_block->owningGraph(), *getBatchOperator("any"), cond);
    auto to_int_node = res_graph->createTensorToNum(IntType::get(), cond_any[0]);
    res_graph->insertNode(to_int_node);
    loop_block->insertOutput(0, to_int_node->output());
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
    batch_map[n->outputs()[i]] = r_node->outputs().slice(i * EXP_BTENSOR_SIZE, EXP_BTENSOR_SIZE).vec();
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
  // for block in prim::Loop, register inputs separately to deal with cond
  if(!block->owningNode() || block->owningNode()->kind() != prim::Loop){
    auto size = block->inputs().size();
    for(size_t i = 0; i < size; i++){
      auto input = block->inputs()[i];
      auto name = input->uniqueName();
      for(size_t j = 0; j < EXP_BTENSOR_SIZE; j++){
        res_block->addInput(name + "_" + EXP_BTENSOR_NAME[j]);
      }
      batch_map[input] = std::vector<Value*>(res_block->inputs().slice(i * EXP_BTENSOR_SIZE, EXP_BTENSOR_SIZE).vec());
    }
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
        case prim::TensorToNum:
          visitTensorToNum(n, block, res_block);
          break;
        case prim::ListConstruct:
          visitListConstruct(n, block, res_block);
          break;
        case prim::If:
          visitIf(n, block, res_block);
          break;
        case prim::Loop:
          visitLoop(n, block, res_block);
          break;
        default:
          throw std::runtime_error("NYI: node of prim kind other than [Constant, NumToTensor, TensorToNum, If, Loop] is not supported yet");
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
  std::shared_ptr<Graph> res_graph = std::make_shared<Graph>(graph->scope_root());
  ToBatch to_batch;
  to_batch.toBatch(graph->block(), res_graph->block());

  EliminateDeadCode(res_graph);
  return res_graph;
}

void initRegisterBatchOpsBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("to_batch_graph", &to_batch_graph);
  m.def("register_batch_operator", [](std::string name, std::shared_ptr<Graph> graph){
    ToBatch::batch_operator_table[name].push_back(graph);
  });
}

}} // namespace torch.jit
