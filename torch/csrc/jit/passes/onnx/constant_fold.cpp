#include <torch/csrc/jit/passes/onnx/constant_fold.h>
#include <c10/util/Exception.h>

#include <c10/util/Optional.h>
#include <algorithm> 

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

static std::map<Value*, std::pair<std::string, at::Tensor>>
       buildValueToParamsMap(Block* b, const std::map<std::string, 
                             at::Tensor>& paramsDict) {
  std::map<Value*, std::pair<std::string, at::Tensor>> valsToParamsMap;
  for(auto& input : b->inputs()) {
      auto it = paramsDict.find(input->uniqueName());
      if (it != paramsDict.end()) {
          valsToParamsMap[input] = *it;
      }
  }
  return valsToParamsMap;
}

static void buildParamsMapFromValueToParamsMap(
    const std::map<Value*, std::pair<std::string, at::Tensor>>& valsToParamsMap,
    std::map<std::string, at::Tensor>& paramsDict) {
  paramsDict.clear();
  for(auto& nameTensorParamPair : valsToParamsMap) {
    paramsDict.insert(nameTensorParamPair.second);
  }
}

static void eraseUnusedNodeOutputs(Node* node) {
  std::vector<std::string> removedOutputNames;
  for (size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
      size_t i = i_1 - 1;
      if (!node->outputs().at(i)->hasUses()) {
        node->eraseOutput(i);
      }
  }
}

static at::Tensor runTorchBackendForOnnx(const Node* node, std::vector<at::Tensor>& inputTensorValues) {
  at::Tensor updated_val;
  auto nodeKind = node->kind().toDisplayString();

  if (node->kind() == onnx::Slice) {
    assert(inputTensorValues.size() == 1);
    if ( !(node->hasAttributeS("axes") && node->hasAttributeS("starts") && node->hasAttributeS("ends")) ) {
      throw std::runtime_error("Missing attribute(s) in onnx::Slice op.");
    }
    auto axesAttr = node->is(attr::axes);
    auto startsAttr = node->is(attr::starts);
    auto endsAttr = node->is(attr::ends);
    if (axesAttr.size() != startsAttr.size() || axesAttr.size() != endsAttr.size()) {
      throw std::runtime_error("onnx::Slice node attributues named, axes, starts, and ends, must be the same length.");
    }
    updated_val = inputTensorValues[0];
    for (size_t i = 0; i < axesAttr.size(); ++i) {
      updated_val = at::narrow(updated_val, axesAttr[i], startsAttr[i], endsAttr[i] - startsAttr[i]);
    }
  }  
  else if (node->kind() == onnx::Concat) {
    updated_val = at::cat(at::TensorList(inputTensorValues), node->i(attr::axis));
  }
  else if (node->kind() == onnx::Unsqueeze) {
    assert(inputTensorValues.size() == 1);
    if (!node->hasAttributeS("axes")) {
      throw std::runtime_error("Missing attribute 'axes' in onnx::Unsqueeze op.");
    }
    updated_val = inputTensorValues[0];
    for (auto axis: node->is(attr::axes)) {
      updated_val = at::unsqueeze(updated_val, axis);
    }
  }
  else if (node->kind() == onnx::Transpose) {
    assert(inputTensorValues.size() == 1);
    if (!node->hasAttributeS("perm")) {
      throw std::runtime_error("Missing attribute 'perm' in onnx::Transpose op.");
    }
    updated_val = inputTensorValues[0].permute(node->is(attr::perm));
  }
  else {
    updated_val = at::empty({0});
    auto qe = updated_val.size(0);
  }
  return updated_val;
}

// enum ConstantLeafNodeKind {
// // Currently only prim::Param and onnx::Constant nodes are supported.
// // More can be added if needed.    
//   PRIM_PARAM,
//   ONNX_CONSTANT
// };

static bool isConstant(Value* val, const std::map<Value*, std::pair<std::string, at::Tensor>>& valsToParamsMap) {
  auto parentNode = val->node();
  return (parentNode->kind() == prim::Param && 
          valsToParamsMap.find(val) != valsToParamsMap.end()) || // Checks val is a parameter and not a real input
         (parentNode->kind() == onnx::Constant && !parentNode->mustBeNone() &&
          parentNode->kindOf(attr::value) == AttributeKind::t); // Check other types?;
}

static std::vector<at::Tensor> getValues(Node* node, 
       const std::map<Value*, std::pair<std::string, at::Tensor>>& valsToParamsMap) {
  size_t numInputs = node->inputs().size();
  std::vector<at::Tensor> inputTensorValues;
  inputTensorValues.reserve(numInputs);
  for (auto val : node->inputs()) {
    if (val->node()->kind() == prim::Param) {
      auto itr = valsToParamsMap.find(val);
      if(itr == valsToParamsMap.end()) {
        throw std::runtime_error("getValues: Input value not found amongst constant parameters.");
      }
      inputTensorValues.push_back(itr->second.second);
    }
    else if (val->node()->kind() == onnx::Constant) {
      inputTensorValues.push_back(val->node()->t(attr::value));
    }
    else {
      throw std::runtime_error("getValues: Unsupported kind of constant node found.");
    }
  }
  AT_ASSERT(inputTensorValues.size() == numInputs);
  return inputTensorValues;
}

static void eraseUnusedValuesFromMap(std::map<Value*, std::pair<std::string, at::Tensor>>& valsToParamsMap) {
//   printf("---------------------------------\n");
//   for (auto& element : valsToParamsMap) {
//       printf("Value *: %p, Value name: %s, Param name: %s\n", 
//         (void *)element.first, element.first->uniqueName().c_str(), element.second.first.c_str());
//   }
//   printf("---------------------------------\n");
  auto it = valsToParamsMap.begin();
  while (it != valsToParamsMap.end()) {
    if (!it->first->hasUses()) {
      it = valsToParamsMap.erase(it);
    } 
    else {
      ++it;
    }
  }
//   printf("---------------------------------\n");
//   for (auto& element : valsToParamsMap) {
//       printf("Value *: %p, Value name: %s, Param name: %s\n", 
//         (void *)element.first, element.first->uniqueName().c_str(), element.second.first.c_str());
//   }
}

// This method updates the block in-place to fold all the one-time 
// constant-based computations/ops into an initializer node.
void ConstantFoldONNX(Block* b, std::map<std::string, at::Tensor>& paramsDict) {
  auto sourceNode = b->param_node();
  AT_ASSERT(sourceNode);
  auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
//   for (auto& element : valsToParamsMap) {
//       printf("Value *: %p, Value name: %s, Param name: %s\n", 
//         (void *)element.first, element.first->uniqueName().c_str(), element.second.first.c_str());
//   }
 
  // Only the root block is constant-folded. Folding nested blocks is
  // not supported for now.
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    auto node = *it;
    if (node->outputs().size() > 1) {
        // Constant folding for multiple-output nodes not supported. Skip it.
        continue;
      }
    if (!std::all_of(node->inputs().begin(), node->inputs().end(),
        [&valsToParamsMap](Value* v) { return isConstant(v, valsToParamsMap); })) {
        // If all the inputs to this node are not either parameter or
        // onnx::Constant, then skip this node.
        continue;
    }
    auto inputTensorValues = getValues(node, valsToParamsMap);
    if (inputTensorValues.empty()) {
        // This is a terminal node with no inputs, such as onnx::Constant. Skip it.
        continue;
    }
    auto updated_val = runTorchBackendForOnnx(node, inputTensorValues);
    if (updated_val.size(0) == 0) {
        // Constant folding is not supported for this op. Skip it.
        continue;
      }
    // Disconnect the folded node. Create a new initializer for the folded  
    // tensor and replace the output of the folded node with the 
    // initializer as input for all downstream nodes.
    auto newSourceNodeOutput = sourceNode->addOutput();
    valsToParamsMap.insert({newSourceNodeOutput, 
                            std::make_pair(newSourceNodeOutput->uniqueName(), updated_val)});
    newSourceNodeOutput->inferTypeFrom(updated_val);
    node->outputs().at(0)->replaceAllUsesWith(newSourceNodeOutput);

    node->removeAllInputs();
  }
  eraseUnusedNodeOutputs(sourceNode);
  eraseUnusedValuesFromMap(valsToParamsMap);
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);


//     // ---------------------------------------------
//     size_t numInputs = node->inputs().size();
//     // std::vector<at::Tensor> inputTensorValues;
//     // inputTensorValues.reserve(numInputs);
//     std::vector<ConstantLeafNodeKind> kindOfLeafNode;
//     kindOfLeafNode.reserve(numInputs);
//     for (auto val : node->inputs()) {
//       auto inputNode = val->node();
//       bool isParam = inputNode->kind() == prim::Param && paramsDict.count(val->uniqueName());
//       if (isParam) {
//         AT_ASSERT(sourceNode == inputNode); // One and only one prim::Param node in block.
//       }
//       bool isConstant = inputNode->kind() == onnx::Constant && !inputNode->mustBeNone()
//         && toString(inputNode->kindOfS("value")) == std::string("t"); // TODO: Check other types?
//       if (isParam) {
//         inputTensorValues.push_back(paramsDict[val->uniqueName()]);
//         kindOfLeafNode.push_back(ConstantLeafNodeKind::PRIM_PARAM);
//       }
//       else if (isConstant) {
//         inputTensorValues.push_back(inputNode->t(c10::Symbol::fromQualString("attr::value")));
//         kindOfLeafNode.push_back(ConstantLeafNodeKind::ONNX_CONSTANT);
//       }
//     }

//     // If there are inputs, and if they all can be folded, then fold them.
//     if (!inputTensorValues.empty() && inputTensorValues.size() == numInputs) {
//       auto updated_val = runTorchBackendForOnnx(node, inputTensorValues);
//       if (updated_val.size(0) == 0) {
//         // Constant folding not supported for this op. Skip it.
//         continue;
//       }
//       if (node->outputs().size() > 1) {
//         // Constant folding for multiple-output nodes not supported. Skip it.
//         continue;
//       }

//       // Disconnect the folded node. Create a new initializer for the folded  
//       // tensor and replace the output of the folded node with the 
//       // initializer as input for all downstream nodes.
//       auto newSourceNodeOutput = sourceNode->addOutput();
//       paramsDict[newSourceNodeOutput->uniqueName()] = updated_val;
//       newSourceNodeOutput->inferTypeFrom(updated_val);
//       node->outputs().at(0)->replaceAllUsesWith(newSourceNodeOutput);

//       // Find the indices to outputs of the source node that are
//       // feeding into the folded node. Used below for removing 
//       // corresponding entried in params_dict.
//       std::unordered_map<std::string, size_t> sourceOutputNames;
//       std::map<size_t, std::string> sourceOutputsToRemove;
//       for (size_t i = 0; i < sourceNode->outputs().size(); ++i) {
//         sourceOutputNames[sourceNode->outputs().at(i)->uniqueName()] = i;
//       }
//       for (size_t i = 0; i < numInputs; ++i) { 
//         if (kindOfLeafNode[i] == ConstantLeafNodeKind::PRIM_PARAM) {
//           auto matchIter = sourceOutputNames.find(node->inputs().at(i)->uniqueName());
//           if (matchIter != sourceOutputNames.end()) {
//             sourceOutputsToRemove[matchIter->second] = matchIter->first;
//           }
//         }
//       }
//       node->removeAllInputs();
//       for (const auto& elem : sourceOutputsToRemove) {
//         if (!sourceNode->outputs().at(elem.first)->hasUses()) {
//           paramsDict.erase(elem.second);
//         }
//       }
//       // // TODO: Should we delete the node, as in the line below?
//       // it.destroyCurrent();
//     }
//   }
//   if (sourceNode != nullptr) {
//     auto removedSourceOutputNames = eraseUnusedNodeOutputs(sourceNode);
//     for (const auto& removedName : removedSourceOutputNames) {
//       if (paramsDict.find(removedName) != paramsDict.end()) {
//         paramsDict.erase(removedName);
//       }
//     }
//   }
  return;
}

} // namespace jit
} // namespace torch