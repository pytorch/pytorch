#include <torch/csrc/jit/passes/quantization.h>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/node_hashing.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/passes/alias_analysis.h>

#include <stack>

namespace torch {
namespace jit {
namespace {
// QuantizerUtils

bool checkIfNodeQuantizable(Node* n) {
  AT_ASSERT(n != nullptr);
  // This is map for quantizable nodes. It will be expanded in future to
  // support more ops and patterns.
  static const OperatorSet quantnodeLookup = {
      "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] \
stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
      "aten::relu(Tensor self) -> Tensor",
      "aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] \
stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, \
int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor"};
  return quantnodeLookup.find(n);
}

void insertQuantNodeParams(Node* quant, std::tuple<float, int> qparam) {
  WithInsertPoint ins(quant);
  Value* scale = quant->owningGraph()->insertConstant(std::get<0>(qparam));
  Value* zeropoint = quant->owningGraph()->insertConstant(std::get<1>(qparam));
  quant->addInput(scale);
  quant->addInput(zeropoint);
}

// Create Quant-Dequant node pair for quantizable Value
std::pair<Node*, Node*> createQuantDeQuantNodes(Value* v, Node* n) {
  Node* quant = n->owningGraph()->create(
      at::Symbol::fromQualString("aten::quantize_linear"));
  AT_ASSERTM(quant != nullptr, "Failed to create quant node");
  quant->output()->setUniqueName(v->uniqueName() + ".quant");

  Node* dequant =
      n->owningGraph()->create(at::Symbol::fromQualString("aten::dequantize"));
  AT_ASSERTM(dequant != nullptr, "Failed to create dequant node");
  dequant->output()->setUniqueName(v->uniqueName() + ".dequant");

  quant->setScope(n->scope());
  dequant->setScope(n->scope());

  return std::make_pair(quant, dequant);
}

// Insert Quant-Dequant node pair for quantizable node outputs
void addQuantDeQuantNodes(Value* v) {
  AT_ASSERT(v != nullptr);
  Node* n = v->node();
  auto qdq = createQuantDeQuantNodes(v, n);
  Node* quant = qdq.first;
  Node* dequant = qdq.second;

  // Add quant-dequant nodes and replace for all uses of Value
  quant->insertAfter(n);
  dequant->insertAfter(quant);
  v->replaceAllUsesWith(dequant->output());

  // Attach inputs to quant and dequant nodes
  quant->addInput(v);
  // Default Quant Params <Scale:1.0, ZeroPoint:0>
  insertQuantNodeParams(quant, std::make_tuple(1.0, 0));
  dequant->addInput(quant->output());
}

// Insert Quant-Dequant node pair for specific input to node n
void addQuantDeQuantNodesForInput(Value* v, Node* n) {
  AT_ASSERT(v != nullptr);
  AT_ASSERT(n != nullptr);
  auto qdq = createQuantDeQuantNodes(v, n);
  Node* quant = qdq.first;
  Node* dequant = qdq.second;

  // Insert the quant-dequant node for the V->N
  // pair which is identified as quantizable during
  // graph iteration
  dequant->insertBefore(n);
  quant->insertBefore(dequant);
  n->replaceInputWith(v, dequant->output());

  // Attach inputs to quant and dequant nodes
  quant->addInput(v);
  // Default Quant Params <Scale:1.0, ZeroPoint:0>
  insertQuantNodeParams(quant, std::make_tuple(1.0, 0));
  dequant->addInput(quant->output());
}

} // namespace

// PyBind APIs
void PropagateQuantInfo(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

static void addObserverFor(Value* v, Node* original_observer_node) {
  Node* def = v->node();
  WithInsertPoint ins(def);

  // We need to pass the value name to observer function - create a constant
  // holding this name.
  Value* vname = def->owningGraph()->insertConstant(v->uniqueName());

  // Create a new observer node. We just need to clone the original one.
  Node* observerNode =
      def->owningGraph()
          ->createClone(
              &*original_observer_node, [&](Value* v) { return v; }, false)
          ->insertAfter(def);

  // Set the type and the name of the output of the new observer node. It will
  // be used instead of the original value v.
  Value* observedValue = observerNode->addOutput();
  observedValue->setType(v->type());
  observedValue->setUniqueName(v->uniqueName() + ".observed");

  // Now we can add the inputs.
  observerNode->addInput(v);
  observerNode->addInput(vname);
}

static bool outputsNeedToBeObserved(Node* n) {
  return n->kind() != prim::Constant;
}

void InsertObserverNodes(std::shared_ptr<Graph>& graph, Node* observer_node) {
  // For storing all values that need to be instrumented with an observer call.
  std::vector<Value*> values_to_observe;

  // For traversing all blocks in the graph including subblocks.
  std::stack<Block*> blocks_to_visit;

  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      // Skip nodes that we don't need to observe, e.g. 'prim::Constant'.
      if (!outputsNeedToBeObserved(n)) {
        continue;
      }

      // Record all outputs in the values_to_observe - we'll later add observers
      // for all values from it.
      for (Value* v : n->outputs()) {
        values_to_observe.push_back(v);
      }

      // Schedule subblocks (if any) for visiting.
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }

  // Actually add observer nodes.
  for (Value* v : values_to_observe) {
    if (v->type()->isSubtypeOf(TensorType::get())) {
      addObserverFor(v, observer_node);
    }
  }
}

void InsertQuantDequantNodes(std::shared_ptr<Graph>& graph) {
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  // For storing quantizable values - node pairs that are external
  // or intermediate inputs to quantizable nodes
  std::vector<std::pair<Value*, Node*>> quantInputs;
  // For storing quantizable values that are output of quantizable nodes
  // Since same value can go to multiple nodes, we use set so that
  // we insert quant-dequant node pairs for value only once
  std::vector<Value*> quantOutputs;
  std::unordered_set<Value*> valueLookup;

  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();

    for (Node* n : b->nodes()) {
      // Schedule the sub blocks
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }

      // We iterate over node inputs to identify which Values
      // need to be quantized depending on node type
      for (auto& v : n->inputs()) {
        if (!v->type()->isSubtypeOf(TensorType::get())) {
          // Skip quantization for non tensors
          continue;
        }

        if (checkIfNodeQuantizable(v->node())) {
          // Goal of this iteration is to identify the parent node for V
          // that is quantizable and replace all uses of Value with
          // quant-dequant output. Usage of set helps adding single
          // q-dq nodes for all V->users
          // Example N1 -> (V1 -> (N2), V2 -> (N3))
          //         N1 is quantizable node. So we insert quant-dequant
          //         nodes for all outputs of N1 (V1, V2) once
          if (!valueLookup.count(v)) {
            valueLookup.emplace(v);
            quantOutputs.emplace_back(v);
          }
        } else if (checkIfNodeQuantizable(n)) {
          // Goal of this iteration is to identify nodes that are
          // quantizable but input value originate from non quantizable
          // node. This requires selectively inserting q-dq nodes for
          // inputs into node N(V, N pair) because parent node might
          // also have input into other non quantizable nodes
          // Example : N1(prim::Param) -> (V1 -> (N4, N5), V2 -> (N6, N7), V3)
          //           N1 is not quantizable node but N4 and N7 are
          //           quantizable nodes. So we add the (V1, N4) and
          //           (V2, N7) as insertion points for quant-dequant nodes
          quantInputs.emplace_back(v, n);
        }
      }
    } // End Loop for nodes within block

    // Since we are iterating node inputs only above, we need to iterate
    // over block outputs values and if they originate from quantizable
    // node, we push to quantOutputs set
    auto outputVals = b->outputs();
    for (auto& v : outputVals) {
      if (checkIfNodeQuantizable(v->node()) &&
          v->type()->isSubtypeOf(TensorType::get())) {
        quantOutputs.emplace_back(v);
      }
    } // end for
  } // end Block traversal

  // Insert the quant-dequant pair for values output from quantizable nodes
  for (auto& ele : quantOutputs) {
    addQuantDeQuantNodes(ele);
  }

  // Insert the quant-dequant pair for values inputs to quantizable nodes
  for (auto& ele : quantInputs) {
    addQuantDeQuantNodesForInput(ele.first, ele.second);
  }
}

void QuantLinting(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

void FoldQuantNodesIntoInputsOutputs(std::shared_ptr<Graph>& graph) {
  throw std::runtime_error("Pass not implemented yet!");
}

} // namespace jit
} // namespace torch
