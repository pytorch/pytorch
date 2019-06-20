#include <ATen/native/TensorFactories.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/jit/passes/onnx/constant_fold.h>
#include <algorithm>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

namespace {

using ParamMap = std::map<std::string, at::Tensor>;
using ValueToParamPairMap =
    std::map<Value*, std::pair<std::string, at::Tensor>>;

std::unordered_map<int, at::ScalarType> onnxTypeToScalarTypeMap = {
    // Only conversion of ONNX numeric types is included here.
    // Unsigned ONNX types are mapped to the next higher signed
    // ScalarType type.
    {1, at::kFloat},
    {2, at::kByte},
    {3, at::kChar},
    {4, at::kInt},
    {5, at::kShort},
    {6, at::kInt},
    {7, at::kLong},
    {10, at::kFloat},
    {11, at::kDouble},
    {12, at::kLong},
};

ValueToParamPairMap buildValueToParamsMap(
    Block* b,
    const ParamMap& paramsDict) {
  ValueToParamPairMap valsToParamsMap;
  for (auto& input : b->inputs()) {
    auto it = paramsDict.find(input->uniqueName());
    if (it != paramsDict.end()) {
      valsToParamsMap.emplace(input, *it);
    }
  }
  return valsToParamsMap;
}

void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict) {
  paramsDict.clear();
  for (const auto& nameTensorParamPair : valsToParamsMap) {
    paramsDict.insert(nameTensorParamPair.second);
  }
}

void eraseUnusedBlockInputs(Block* b) {
  for (size_t i_1 = b->inputs().size(); i_1 > 0; --i_1) {
    size_t i = i_1 - 1;
    if (!b->inputs().at(i)->hasUses()) {
      b->eraseInput(i);
    }
  }
}

c10::optional<at::Tensor> runTorchBackendForOnnx(
    const Node* node,
    std::vector<at::Tensor>& inputTensorValues) {
  at::Tensor updated_val;
  if (node->kind() == onnx::Slice) {
    assert(inputTensorValues.size() == 1);
    if (!(node->hasAttributeS("starts") && node->hasAttributeS("ends"))) {
      return c10::nullopt;
    }
    auto startsAttr = node->is(attr::starts);
    auto endsAttr = node->is(attr::ends);
    if (startsAttr.size() != endsAttr.size()) {
      return c10::nullopt;
    }
    std::vector<int64_t> axesAttr;
    if (node->hasAttributeS("axes")) {
      axesAttr = node->is(attr::axes);
    } else {
      axesAttr.resize(startsAttr.size());
      std::iota(axesAttr.begin(), axesAttr.end(), 0);
    }
    updated_val = inputTensorValues[0];
    for (size_t i = 0; i < axesAttr.size(); ++i) {
      updated_val = at::narrow(
          updated_val, axesAttr[i], startsAttr[i], endsAttr[i] - startsAttr[i]);
    }
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Concat) {
    if (!node->hasAttributeS("axis")) {
      return c10::nullopt;
    }
    updated_val =
        at::cat(at::TensorList(inputTensorValues), node->i(attr::axis));
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Unsqueeze) {
    assert(inputTensorValues.size() == 1);
    if (!node->hasAttributeS("axes")) {
      return c10::nullopt;
    }
    updated_val = inputTensorValues[0];
    for (auto axis : node->is(attr::axes)) {
      updated_val = at::unsqueeze(updated_val, axis);
    }
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Transpose) {
    assert(inputTensorValues.size() == 1);
    if (!node->hasAttributeS("perm")) {
      return c10::nullopt;
    }
    updated_val = inputTensorValues[0].permute(node->is(attr::perm));
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Cast) {
    assert(inputTensorValues.size() == 1);
    if (node->hasAttributeS("to") &&
        onnxTypeToScalarTypeMap.find(node->i(attr::to)) !=
            onnxTypeToScalarTypeMap.end()) {
      updated_val =
          inputTensorValues[0].to(onnxTypeToScalarTypeMap[node->i(attr::to)]);
      return c10::optional<at::Tensor>(updated_val);
    }
    return c10::nullopt;
  } else {
    return c10::nullopt;
  }
}

bool isConstant(Value* val, const ValueToParamPairMap& valsToParamsMap) {
  auto parentNode = val->node();
  return (parentNode->kind() == prim::Param &&
          valsToParamsMap.find(val) !=
              valsToParamsMap
                  .end()) || // Checks val is a parameter and not a real input
      (parentNode->kind() == onnx::Constant && !parentNode->mustBeNone() &&
       parentNode->kindOf(attr::value) ==
           AttributeKind::t); // Check other types?
}

std::vector<at::Tensor> getValues(
    Node* node,
    const ValueToParamPairMap& valsToParamsMap) {
  size_t numInputs = node->inputs().size();
  std::vector<at::Tensor> inputTensorValues;
  inputTensorValues.reserve(numInputs);
  for (auto val : node->inputs()) {
    if (val->node()->kind() == prim::Param) {
      auto itr = valsToParamsMap.find(val);
      if (itr == valsToParamsMap.end()) {
        throw std::runtime_error(
            "getValues: Input value not found amongst constant parameters.");
      }
      inputTensorValues.push_back(itr->second.second);
    } else if (val->node()->kind() == onnx::Constant) {
      inputTensorValues.push_back(val->node()->t(attr::value));
    } else {
      throw std::runtime_error(
          "getValues: Unsupported kind of constant node found.");
    }
  }
  AT_ASSERT(inputTensorValues.size() == numInputs);
  return inputTensorValues;
}

void eraseUnusedValuesFromMap(ValueToParamPairMap& valsToParamsMap) {
  auto it = valsToParamsMap.begin();
  while (it != valsToParamsMap.end()) {
    if (!it->first->hasUses()) {
      it = valsToParamsMap.erase(it);
    } else {
      ++it;
    }
  }
}

bool areNodeInputsConstant(
    Node* node,
    const ValueToParamPairMap& valsToParamsMap) {
  return std::all_of(
      node->inputs().begin(),
      node->inputs().end(),
      [&valsToParamsMap](Value* v) { return isConstant(v, valsToParamsMap); });
}

std::vector<Node*> getOnnxConstParentsToRemove(Node* node) {
  std::vector<Node*> parentNodes;
  for (auto val : node->inputs()) {
    // If the parent of 'node' is an onnx::Constant node,
    // and 'node' is the only downstream node it serves (this
    // is important), then push it in the list to remove.
    if (val->node()->kind() == onnx::Constant &&
        val->uses().size() == 1) {
          parentNodes.push_back(val->node());
    }
  }
  return parentNodes;
}

// Recursive tracker on input node dependency
bool isDynamic(const Node* node, const Graph* graph) {
  if (node == nullptr) {
    return false;
  }
  for (auto inp : node->inputs()) {
    auto ginps = graph->inputs();
    for (auto ginp : ginps) {
      if (inp == ginp) {
        return true;
      }
    }
    if (inp->type()->cast<DimensionedTensorType>()) {
      return false;
    }
    if (isDynamic(inp->node(), graph)) {
      return true;
    }
  }
  return false;
}

// Recursive collector of a tree having Concat as a root and Constants as leaves.
// With Unsqueeze, Gather and Shape only in between.
// Returns non-empty array if succeeded.
std::vector<int64_t> collectFoldables(int& axis, int level, Node* node,
    std::vector<std::vector<Node*>>& removeNodes) {
  std::vector<int64_t> ret;
  if (level > 4) {
    return ret; // not deeper
  }
  if (removeNodes.size() <= level) {
    removeNodes.emplace_back(std::vector<Node*>());
  }
  removeNodes[level].emplace_back(node);

  if (node->kind() == onnx::Constant) {
    if (node->hasAttribute(attr::value) &&
        node->kindOf(attr::value) == AttributeKind::t) {
      const at::Tensor& val = node->t(attr::value);
      if (val.numel() == 1) {
        ret.emplace_back(val.item().toLong());
      }
    }
  } else if (node->kind() == onnx::Shape && node->inputs().size() == 1) {
    if (isDynamic(node, node->owningGraph())) {
      return ret;
    }
    if (auto value = node->inputs()[0]->type()->cast<CompleteTensorType>()) {
      ret = value->sizes();
    }
  } else if (node->kind() == onnx::Unsqueeze && node->inputs().size() == 1) {
    auto inp = node->inputs()[0];
    return collectFoldables(axis, level + 1, inp->node(), removeNodes);
  } else if (node->kind() == onnx::Gather && node->inputs().size() == 2) {
    axis = 0LL;
    if (node->hasAttribute(attr::axis) && node->kindOf(attr::axis) == AttributeKind::i) {
      axis = node->i(attr::axis);
    }
    auto data = node->inputs()[0];
    auto indx = node->inputs()[1];
    auto dval = collectFoldables(axis, level + 1, data->node(), removeNodes);
    auto ival = collectFoldables(axis, level + 1, indx->node(), removeNodes);
    if(ival.size() == 1 && ival[axis] < dval.size()) {
      ret.emplace_back(dval[ival[axis]]);
    }
  } else if (node->kind() == onnx::Slice && node->inputs().size() == 1) {
    axis = 0LL;
    if (node->hasAttribute(attr::axes) && node->kindOf(attr::axes) == AttributeKind::is) {
      axis = node->is(attr::axes)[0];
    }
    if (axis == 0L) {
      auto data = node->inputs()[0];
      auto dval = collectFoldables(axis, level + 1, data->node(), removeNodes);
      if (!dval.empty() &&
          node->hasAttribute(attr::ends) && node->kindOf(attr::ends) == AttributeKind::is &&
          node->is(attr::ends).size() == 1) {
        int64_t end = node->is(attr::ends)[0];
        if (end <= 0L) {
          end += dval.size();
        }
        int64_t start = 0L;
        if (node->hasAttribute(attr::starts) && node->kindOf(attr::starts) == AttributeKind::is &&
            node->is(attr::starts).size() == 1) {
          start = node->is(attr::starts)[0];
          if (start < 0L) {
            start += dval.size();
          }
        }
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(dval[i]);
        }
      }
    }
  } else if (node->kind() == onnx::Concat && !node->inputs().empty()) {
    for (auto inp : node->inputs()) {
      auto inpVal = collectFoldables(axis, level + 1, inp->node(), removeNodes);
      if (inpVal.size() == 1) {
        ret.emplace_back(inpVal[0]);
      } else {
        break;
      }
    }
    if (ret.size() < 2) {
      ret.clear();
    }
  }
  return ret;
}

} // Anonymous namespace

// This method updates the block in-place to fold all the one-time
// constant-based computations/ops into an initializer node.
void ConstantFoldONNX(Block* b, ParamMap& paramsDict) {
  AT_ASSERT(b->param_node());
  /*
   * We can do better for *static* cases like the one below.
   * This use case is quite popular, but default implementation doesn't go deep.
   * This handler replaces 5 ops shown below with just one:
   *
...
%30 : Float(2, 256, 6, 6) = onnx::AveragePool[kernel_shape=[1, 1],strides=[1, 1]](%29), scope:
%31 : Long() = onnx::Constant[value={2}](), scope:
%32 : Long() = onnx::Constant[value={9216}](), scope:
%33 : Tensor = onnx::Unsqueeze[axes=[0]](%31)
%34 : Tensor = onnx::Unsqueeze[axes=[0]](%32)
%35 : Tensor = onnx::Concat[axis=0](%33, %34)
%36 : Float(2, 9216) = onnx::Reshape(%30, %35), scope:
...

   becomes:
...
%30 : Float(2, 256, 6, 6) = onnx::AveragePool[kernel_shape=[1, 1],strides=[1, 1]](%29), scope:
%44 : Long(2) = onnx::Constant[value=    2  9216 [ Variable[CPULongType]{2} ]]()
%36 : Float(2, 9216) = onnx::Reshape(%30, %44), scope:
...
  */

  int axis = 0;
  std::vector<int64_t> values;
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    std::vector<std::vector<Node*>> removeNodes;
    auto node = *it;
    if (node->kind() == onnx::Concat && node->hasUses()) {
      values = collectFoldables(axis, 0, node, removeNodes);
      if (!values.empty()) {
        at::Tensor updatedVal = at::tensor(values,
            at::TensorOptions().dtype(at::kLong).is_variable(true).layout(at::kStrided)
            .device(at::kCPU));
        Node* new_shape = b->owningGraph()->create(onnx::Constant, 1);
        new_shape->t_(attr::value, updatedVal);
        auto newSourceNodeOutput = new_shape->insertAfter(node)->output();
        newSourceNodeOutput->inferTypeFrom(updatedVal);
        node->outputs().at(0)->replaceAllUsesWith(newSourceNodeOutput);
        node->removeAllInputs();
        for (auto& nvec : removeNodes) {
          for (auto& n : nvec) {
            if (node != n) {
              n->destroy();
            }
          }
        }
        it.destroyCurrent();
      }
    }
  } // End of the special handler

  // Default implementation
  auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
  // Only the root block is constant-folded. Folding nested blocks is
  // not supported for now.
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    auto node = *it;
    if (node->outputs().size() > 1) {
      // Constant folding for multiple-output nodes not supported. Skip it.
      continue;
    }
    if (!areNodeInputsConstant(node, valsToParamsMap)) {
      // If all the inputs to this node are not either parameter or
      // onnx::Constant, then skip this node.
      continue;
    }
    auto inputTensorValues = getValues(node, valsToParamsMap);
    if (inputTensorValues.empty()) {
      // This is a terminal node with no inputs, such as onnx::Constant. Skip
      // it.
      continue;
    }
    auto updatedValWrapped = runTorchBackendForOnnx(node, inputTensorValues);
    if (updatedValWrapped == c10::nullopt) {
      // Constant folding is not supported for this op. Skip it.
      continue;
    }
    // Create a new input to the block (prim::Param node output). Add a
    // corresponding entryin valToParamMap. Replace the downstream inputs
    // with this value, and disconnect all the input values of the folded node.
    at::Tensor updatedVal = *updatedValWrapped;
    auto newSourceNodeOutput = b->addInput();
    valsToParamsMap.insert(
        {newSourceNodeOutput,
         std::make_pair(newSourceNodeOutput->uniqueName(), updatedVal)});
    newSourceNodeOutput->inferTypeFrom(updatedVal);
    node->outputs().at(0)->replaceAllUsesWith(newSourceNodeOutput);

    // Next we remove the current node that has been replaced by
    // an initializer. But before we start de-wiring this node,
    // we check if any parents of this nodes were onnx::Constant
    // and remove them first (following proper sequence as shown
    // below), and then remove the current node. If the parent was
    // an initializer (not onnx::Constant) then they are all removed
    // by eraseUnusedBlockInputs() call (below) outside the loop.
    auto onnxConstParents = getOnnxConstParentsToRemove(node);
    node->removeAllInputs();
    for (auto* n : onnxConstParents) {
      n->destroy();
    }
    it.destroyCurrent();
  }
  eraseUnusedValuesFromMap(valsToParamsMap);
  eraseUnusedBlockInputs(b);
  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
  return;
}

} // namespace jit
} // namespace torch
