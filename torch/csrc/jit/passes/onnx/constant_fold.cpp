#include <torch/csrc/jit/passes/onnx/constant_fold.h>
#include <c10/util/Exception.h>

#include <c10/util/Optional.h>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

static Node* getSourceNode(Block& b) {
  // This method returns the first prim::Param node it encounters,
  // based on the assumption that there is always one and only one
  // prim::Param node in a block.
  for (auto it = b.nodes().begin(), end = b.nodes().end(); it != end; ++it) {
    auto n = *it;
    for (auto input : n->inputs()) {
      if(input->node()->kind() == prim::Param) {
        return input->node();
      }
    }
  }
  return nullptr;
}

static std::vector<std::string> eraseUnusedNodeOutputs(Node* node) {
  std::vector<std::string> removedOutputNames;
  for (size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
      size_t i = i_1 - 1;
      if (!node->outputs().at(i)->hasUses()) {
        removedOutputNames.push_back(node->outputs().at(i)->uniqueName());
        node->eraseOutput(i);
      }
  }
  return removedOutputNames;
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

enum ConstantLeafNodeKind {
// Currently only prim::Param and onnx::Constant nodes are supported.
// More can be added if needed.    
  PRIM_PARAM,
  ONNX_CONSTANT
};

// This method updates the block in-place to fold all the one-time 
// constant-based computations/ops into an initializer node.
void ConstantFoldONNX(Block* b, std::map<std::string, at::Tensor>& paramsDict) {
  auto sourceNode = getSourceNode(*b);
  if (sourceNode == nullptr) {
    return;
  }
  // Only the root block is constant-folded. Folding nested blocks is
  // not supported for now.
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    auto node = *it;
    auto nodeKind = node->kind().toDisplayString();
    size_t numInputs = node->inputs().size();
    std::vector<at::Tensor> inputTensorValues;
    inputTensorValues.reserve(numInputs);
    std::vector<ConstantLeafNodeKind> kindOfLeafNode;
    kindOfLeafNode.reserve(numInputs);
    for (auto val : node->inputs()) {
      auto inputNode = val->node();
      bool isParam = inputNode->kind() == prim::Param && paramsDict.count(val->uniqueName());
      if (isParam) {
        AT_ASSERT(sourceNode == inputNode); // One and only one prim::Param node in block.
      }
      bool isConstant = inputNode->kind() == onnx::Constant && !inputNode->mustBeNone()
        && toString(inputNode->kindOfS("value")) == std::string("t"); // TODO: Check other types?
      if (isParam) {
        inputTensorValues.push_back(paramsDict[val->uniqueName()]);
        kindOfLeafNode.push_back(ConstantLeafNodeKind::PRIM_PARAM);
      }
      else if (isConstant) {
        inputTensorValues.push_back(inputNode->t(c10::Symbol::fromQualString("attr::value")));
        kindOfLeafNode.push_back(ConstantLeafNodeKind::ONNX_CONSTANT);
      }
    }

    // If there are inputs, and if they all can be folded, then fold them.
    if (!inputTensorValues.empty() && inputTensorValues.size() == numInputs) {
      auto updated_val = runTorchBackendForOnnx(node, inputTensorValues);
      if (updated_val.size(0) == 0) {
        // Constant folding not supported for this op. Skip it.
        continue;
      }
      if (node->outputs().size() > 1) {
        // Constant folding for multiple-output nodes not supported. Skip it.
        continue;
      }

      // Disconnect the folded node. Create a new initializer for the folded  
      // tensor and replace the output of the folded node with the 
      // initializer as input for all downstream nodes.
      auto newSourceNodeOutput = sourceNode->addOutput();
      paramsDict[newSourceNodeOutput->uniqueName()] = updated_val;
      newSourceNodeOutput->inferTypeFrom(updated_val);
      node->outputs().at(0)->replaceAllUsesWith(newSourceNodeOutput);

      // Find the indices to outputs of the source node that are
      // feeding into the folded node. Used below for removing 
      // corresponding entried in params_dict.
      std::unordered_map<std::string, size_t> sourceOutputNames;
      std::map<size_t, std::string> sourceOutputsToRemove;
      for (size_t i = 0; i < sourceNode->outputs().size(); ++i) {
        sourceOutputNames[sourceNode->outputs().at(i)->uniqueName()] = i;
      }
      for (size_t i = 0; i < numInputs; ++i) { 
        if (kindOfLeafNode[i] == ConstantLeafNodeKind::PRIM_PARAM) {
          auto matchIter = sourceOutputNames.find(node->inputs().at(i)->uniqueName());
          if (matchIter != sourceOutputNames.end()) {
            sourceOutputsToRemove[matchIter->second] = matchIter->first;
          }
        }
      }
      node->removeAllInputs();
      for (const auto& elem : sourceOutputsToRemove) {
        if (!sourceNode->outputs().at(elem.first)->hasUses()) {
          paramsDict.erase(elem.second);
        }
      }
      // // TODO: Should we delete the node, as in the line below?
      // it.destroyCurrent();
    }
  }
  if (sourceNode != nullptr) {
    auto removedSourceOutputNames = eraseUnusedNodeOutputs(sourceNode);
    for (const auto& removedName : removedSourceOutputNames) {
      if (paramsDict.find(removedName) != paramsDict.end()) {
        paramsDict.erase(removedName);
      }
    }
  }
return;
}

} // namespace jit
} // namespace torch