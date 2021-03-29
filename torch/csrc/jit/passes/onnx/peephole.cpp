#include <torch/csrc/jit/passes/onnx/peephole.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/passes/onnx/helper.h>

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

bool isRNN(const Node* node) {
  auto k = node->kind();
  return k == onnx::RNN || k == onnx::LSTM || k == onnx::GRU;
}

bool isNopTranspose(const std::vector<int64_t>& perm) {
  for (int64_t i = 0, perm_size = perm.size(); i < perm_size; i++)
    if (perm[i] != i)
      return false;
  return true;
}

// returns a vector `ret` such that transposing by `ret` is equivalent
// to transposing by `t1` and then by `t2`
//
// This fires in the case that we have transpose ops T1 -> T2. We are
// fusing the transpose op T1 into T2 and discarding T1. We assume the elements
// of the permutation in `t1` are raw indices into its input, since a previous
// iteration would have folded all the transposes up to that point. Thus,
// `ret[i] = t1[t2[i]]` says "the output of t2 at position i takes the value of
// the input tensor index contained in t1 at position `t2[i]``".
std::vector<int64_t> composeTransposes(
    const std::vector<int64_t>& t1,
    const std::vector<int64_t>& t2) {
  AT_ASSERT(t1.size() == t2.size());
  std::vector<int64_t> ret;
  ret.reserve(t1.size());
  for (const auto& i : t2) {
    AT_ASSERT(i < int64_t(t1.size()));
    ret.push_back(t1[i]);
  }
  return ret;
}

std::vector<size_t> getBroadcastPositions(Node* node) {
  // Most of the element-wise ops in ONNX supports numpy broadcasting.
  // Only GEMM supports one-directional broadcasting, which broadcasts the bias
  // to the product.
  static std::unordered_map<NodeKind, std::vector<size_t>> broadcast_positions =
      {
          {onnx::Add, {0, 1}},
          {onnx::Div, {0, 1}},
          {onnx::Mul, {0, 1}},
          {onnx::Pow, {0, 1}},
          {onnx::Sub, {0, 1}},
          {onnx::Gemm, {2}},
          {onnx::Equal, {0, 1}},
          {onnx::Greater, {0, 1}},
          {onnx::Less, {0, 1}},
      };
  static std::vector<size_t> no_positions;
  std::vector<size_t> positions;

  auto iter = broadcast_positions.find(node->kind());
  if (iter != broadcast_positions.end()) {
    // skip optional input if not provided
    for (size_t position : iter->second) {
      if (position < node->inputs().size()) {
        positions.emplace_back(position);
      }
    }
    return positions;
  }
  return no_positions;
}

// Determine whether `from` can broadcast to `to`, and if so at which
// position. `from` must be a suffix of `to`, except that any
// occurrences of 1 in `from` are treated as wildcards.
c10::optional<size_t> fusibleExpandTo(
    at::IntArrayRef from,
    at::IntArrayRef to) {
  if (from.size() > to.size()) {
    return c10::nullopt;
  }

  for (size_t i = 0; i < from.size(); i++) {
    auto fdim = from[from.size() - 1 - i];
    auto tdim = to[to.size() - 1 - i];
    if (fdim != 1 && fdim != tdim) {
      return c10::nullopt;
    }
  }

  return to.size() - from.size();
}

void fuseBroadcast(Block* b) {
  for (auto n : b->nodes()) {
    for (auto* child_block : n->blocks()) {
      fuseBroadcast(child_block);
    }

    auto broadcast_positions = getBroadcastPositions(n);
    if (!broadcast_positions.empty()) {
      AT_ASSERT(!n->hasAttribute(attr::axis));
    }

    for (size_t position : broadcast_positions) {
      auto* expand_node = n->input(position)->node();

      // Confirm it is expand node.
      if (expand_node->kind() != aten::expand ||
          expand_node->input(1)->node()->kind() != onnx::Constant ||
          expand_node->input(2)->node()->kind() != onnx::Constant) {
        continue;
      }

      auto* unexpanded_input = expand_node->input(0);

      // We need to know what the type pre-expand is.  We should basically
      // always have this information (because expands are only ever traced,
      // not generated from symbolic), but if for some reason we don't
      // have it, we need to skip.
      if (!unexpanded_input->isCompleteTensor() ||
          !n->output()->isCompleteTensor())
        continue;

      // Not all broadcasts are supported by ONNX broadcast.
      c10::optional<size_t> axis = fusibleExpandTo(
          unexpanded_input->type()
              ->expectRef<TensorType>()
              .sizes()
              .concrete_sizes()
              .value(), // from
          n->output()
              ->type()
              ->expectRef<TensorType>()
              .sizes()
              .concrete_sizes()
              .value()); // to
      if (axis == c10::nullopt)
        continue;

      n->replaceInput(position, unexpanded_input);
      if (!expand_node->hasUses()) {
        expand_node->destroy();
      }
    }
  }
}

void fuseConsecutiveTransposes(Block* b) {
  for (auto n : b->nodes()) {
    for (auto* child_block : n->blocks()) {
      fuseConsecutiveTransposes(child_block);
    }
    if (n->kind() == onnx::Transpose &&
        n->input()->node()->kind() == onnx::Transpose &&
        n->owningBlock() == n->input()->node()->owningBlock()) {
      auto origInput = n->input();
      n->is_(
          attr::perm,
          composeTransposes(
              origInput->node()->is(attr::perm), n->is(attr::perm)));
      n->replaceInput(0, origInput->node()->input());
      if (origInput->uses().size() == 0) {
        origInput->node()->destroy();
      }
      continue;
    }
  }
}

void eliminateNopTranspose(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    auto n = *it;
    for (auto* child_block : n->blocks()) {
      eliminateNopTranspose(child_block);
    }
    if (n->kind() == onnx::Transpose) {
      if (isNopTranspose(n->is(attr::perm))) {
        n->output()->replaceAllUsesWith(n->input());
        it.destroyCurrent();
        continue;
      }
    }
  }
}

void fuseTransposeIntoGemm(Block* b) {
  static const std::vector<int64_t> simpleTransPerm({1, 0});

  for (auto n : b->nodes()) {
    for (auto* child_block : n->blocks()) {
      fuseTransposeIntoGemm(child_block);
    }
    if (n->kind() == onnx::Gemm) {
      for (size_t i : {0, 1}) {
        auto inp = n->inputs()[i];
        auto trans = i == 0 ? attr::transA : attr::transB;
        if (inp->node()->kind() == onnx::Transpose &&
            inp->node()->is(attr::perm) == simpleTransPerm) {
          n->replaceInput(i, inp->node()->input());
          n->i_(trans, n->hasAttribute(trans) ? !n->i(trans) : 1);
          if (inp->uses().size() == 0) {
            inp->node()->destroy();
          }
        }
      }
    }
  }
}

// Why this is here:
//
//   Pytorch has a "packed" representation of sequences, as well as a
//   "padded" representation. ONNX has only one representation,
//   corresponding to pytorch's "padded". Therefore, we need to remove
//   any use of packed sequences before exporting.
//
// What this does:
//
//   This code uses the observation that
//     RNN(PackPadded(x)) == PackPadded(RNN(x))
//   and converts the first form to the second whenever possible,
//   "pushing" the packing operation past the RNN operation. Then,
//   the removeNopPacking pass removes the packing operations
//   entirely by pairing them with their inverse PadPacked. If the
//   input graph does not pair the operations, export will fail.

void pushPackingPastRnn(Block* b) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it) {
    auto* n = *it;
    for (auto* child_block : n->blocks()) {
      pushPackingPastRnn(child_block);
    }

    if (n->kind() != prim::PackPadded) {
      continue;
    }
    if (n->outputs().at(0)->uses().size() != 1) {
      // For now, only handle the case where there is one consumer.
      continue;
    }
    Node* rnn = n->outputs()[0]->uses()[0].user;
    if (!isRNN(rnn)) {
      continue;
    }

    if (rnn->owningBlock() != n->owningBlock())
      continue;

    // Packing only has an effect on a network when its outputs are actually
    // used, so we can remove it here.
    if (rnn->outputs().at(0)->uses().empty() &&
        n->outputs().at(1)->uses().size() == 1) {
      n->outputs().at(0)->replaceAllUsesWith(n->inputs().at(0));
      n->outputs().at(1)->replaceFirstUseWith(n->inputs().at(1));
      it.destroyCurrent();
      continue;
    }

    // The rnn is followed by a transpose and a reshape (if
    // bidirectional), or by a squeeze (if unidirectional).
    Node* next = rnn->outputs().at(0)->uses().at(0).user;
    if (next->kind() == onnx::Transpose) {
      next = next->outputs().at(0)->uses().at(0).user;
      if (next->kind() != onnx::Reshape) {
        continue;
      }
    } else if (next->kind() != onnx::Squeeze) {
      continue;
    }

    // remove PackPadded from in front of the RNN
    n->outputs().at(0)->replaceAllUsesWith(n->inputs().at(0));

    // note there can be multiple uses of the length blob. If we are
    // translating a multi-level RNN it will be an input to each level.
    n->outputs().at(1)->replaceFirstUseWith(n->inputs().at(1));

    // and insert new PackPadded after the RNN
    Node* newPackPadded = b->owningGraph()->create(prim::PackPadded, 2);
    newPackPadded->insertAfter(next);

    // make things consume from the new PackPadded
    next->outputs().at(0)->replaceAllUsesWith(newPackPadded->outputs().at(0));
    n->outputs().at(1)->replaceAllUsesWith(newPackPadded->outputs().at(1));

    // setup the new PackPadded's inputs
    newPackPadded->addInput(next->outputs().at(0));
    newPackPadded->addInput(n->inputs().at(1));

    // See https://github.com/pytorch/pytorch/issues/9043 for a full
    // description.  Since PackPadded is for now treated in an
    // unhygenic way, Pytorch ends up propagating an incorrect type.
    // Until a long-term cleanup comes around, we can fix this by
    // resetting the size to the correct value.
    TensorTypePtr oldType = rnn->inputs().at(0)->type()->cast<TensorType>();
    if (oldType && oldType->isComplete()) {
      std::vector<int64_t> new_sizes;
      new_sizes.push_back(*oldType->sizes()[0]);
      new_sizes.push_back(*oldType->sizes()[1]);
      if (next->kind() == onnx::Reshape) {
        // bidirection
        new_sizes.push_back(rnn->i(attr::hidden_size) * 2);
      } else {
        // unidirection
        new_sizes.push_back(rnn->i(attr::hidden_size));
      }
      TensorTypePtr newType = TensorType::createContiguous(
          *oldType->scalarType(), *oldType->device(), new_sizes);
      next->outputs().at(0)->setType(newType);
    }

    it.destroyCurrent();
  }
}

void removeNopPacking(Block* graph) {
  for (auto it = graph->nodes().begin(); it != graph->nodes().end(); ++it) {
    auto* n = *it;
    for (auto* child_block : n->blocks()) {
      removeNopPacking(child_block);
    }

    if (n->kind() != prim::PadPacked) {
      continue;
    }
    Node* input = n->inputs()[0]->node();
    if (input->kind() != prim::PackPadded) {
      continue;
    }
    if (input->outputs()[0] != n->inputs()[0]) {
      continue;
    }
    if (input->outputs()[1] != n->inputs()[1]) {
      continue;
    }
    n->outputs()[0]->replaceAllUsesWith(input->inputs()[0]);
    n->outputs()[1]->replaceAllUsesWith(input->inputs()[1]);

    n->removeAllInputs();
    it.destroyCurrent();
  }
}

void hackFixupPadPackedShapes(Block* graph) {
  // FIXME: the shape of the input to the fictional PadPacked node has
  // incorrect shape. For now, just copy the shape of PadPacked to the shape
  // of its input.
  for (auto it = graph->nodes().begin(); it != graph->nodes().end(); ++it) {
    auto* n = *it;
    for (auto* child_block : n->blocks()) {
      removeNopPacking(child_block);
    }

    if (n->kind() != prim::PadPacked) {
      continue;
    }
    Node* input = n->inputs()[0]->node();
    input->outputs()[0]->setType(n->outputs()[0]->type());
  }
}

void fixDefaultRNNState(
    Graph* graph,
    Node* n,
    int input_index,
    int opset_version) {
  auto initial_state = n->inputs()[input_index];

  // The RNN code in pytorch accepts an optional hidden state.
  // 1- When it is provided as an input, everything works great.
  // 2- When it is not provided, it is default-initialized by constructing a new
  // Variable, which gets
  //    traced as a ConstantOfShape with the expected Shape.
  // 3- When the batch size is fixed, everything works great as well.
  // 4- When h0 and c0 are specified but are not inputs of the model (they are
  // Constants)
  //    and the batch size is variable, the model should be saved with a batch
  //    size of 1 (or an error will occur), and we save the value of h0 and c0
  //    with a batch size of 1. When the model is then called with a different
  //    batch size value, h0 and c0 are broadcasted to get the right shape.
  // Recognize that last pattern here (4) and fix the shape.
  // Note that for multi-layer RNNs there will be a Slice operation between the
  // Constant and the RNN.
  bool needsFixing = initial_state->node()->kind() == onnx::Constant ||
      (initial_state->node()->kind() == onnx::Slice &&
       initial_state->node()->inputs()[0]->node()->kind() == onnx::Constant);

  if (!needsFixing) {
    return;
  }

  Node* shape_of_input = graph->create(onnx::Shape, 1);
  shape_of_input->insertBefore(n);
  shape_of_input->addInput(n->inputs()[0]);

  Node* gather_indices = graph->create(onnx::Constant, 1);
  gather_indices->insertBefore(n);
  gather_indices->t_(attr::value, at::scalar_to_tensor(at::Scalar(1)));

  Node* batch_size = graph->create(onnx::Gather, 1);
  batch_size->insertBefore(n);
  batch_size->addInput(shape_of_input->outputs()[0]);
  batch_size->addInput(gather_indices->outputs()[0]);

  Node* unsqueezed_batch_size =
      createONNXUnsqueeze(graph, n, batch_size->outputs()[0], 0, opset_version);

  Node* hidden_size = graph->create(onnx::Constant, 1);
  hidden_size->insertBefore(n);
  hidden_size->t_(
      attr::value,
      at::full(
          {1},
          n->i(attr::hidden_size),
          at::kLong)); // at::Scalar(n->i(attr::hidden_size)).toTensor());

  Node* num_directions = graph->create(onnx::Constant, 1);
  num_directions->insertBefore(n);
  num_directions->t_(
      attr::value,
      scalar_to_tensor(at::Scalar(
          n->hasAttribute(attr::direction) &&
                  n->s(attr::direction) == "bidirectional"
              ? 2
              : 1)));

  Node* unsqueezed_num_directions = createONNXUnsqueeze(
      graph, n, num_directions->outputs()[0], 0, opset_version);

  Node* concated_dims = graph->create(onnx::Concat, 1);
  concated_dims->insertBefore(n);
  concated_dims->i_(attr::axis, 0);
  concated_dims->addInput(unsqueezed_num_directions->outputs()[0]);
  concated_dims->addInput(unsqueezed_batch_size->outputs()[0]);
  concated_dims->addInput(hidden_size->outputs()[0]);

  Node* fixed_init_state = graph->create(onnx::Expand, 1);
  fixed_init_state->insertBefore(n);
  fixed_init_state->addInput(initial_state);
  fixed_init_state->addInput(concated_dims->outputs()[0]);
  n->replaceInput(input_index, fixed_init_state->outputs()[0]);

  if (initial_state->uses().size() == 0) {
    initial_state->node()->destroy();
  }
}

void fixDefaultRnnHiddenState(Block* b, int opset_version) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it) {
    auto* n = *it;
    for (auto* child_block : n->blocks()) {
      fixDefaultRnnHiddenState(child_block, opset_version);
    }

    if (!isRNN(n)) {
      continue;
    }
    // Hidden state is the sixth input for RNN, LSTM, GRU.
    // See https://pytorch.org/docs/master/nn.html#torch.nn.RNN
    if (n->inputs().size() < 6) {
      continue;
    }
    fixDefaultRNNState(b->owningGraph(), n, 5, opset_version);
  }
}

void fixDefaultLstmCellState(Block* b, int opset_version) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it) {
    auto* n = *it;
    for (auto* child_block : n->blocks()) {
      fixDefaultLstmCellState(child_block, opset_version);
    }

    if (n->kind() != onnx::LSTM) {
      continue;
    }
    // Cell state is the seventh input for LSTM.
    // See https://pytorch.org/docs/master/nn.html#torch.nn.LSTM
    if (n->inputs().size() < 7) {
      continue;
    }
    fixDefaultRNNState(b->owningGraph(), n, 6, opset_version);
  }
}

static bool isSafeToSpeculate(Node* n) {
  return n->kind() == onnx::Transpose;
}

static void speculateOps(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* n = *it;
    ++it; // note: increment first so that it is safe to move the node if needed

    for (auto b : n->blocks()) {
      speculateOps(b);
    }
    if (!isSafeToSpeculate(n))
      continue;
    // XXX - only works for nodes with a single input
    // move node n outside of the control flow it is nested in
    auto node_input = n->input()->node();
    if (node_input->owningBlock() == n->owningBlock())
      continue;
    // Skip if output of this node is part of block output.
    bool is_block_output = false;
    for (auto node_output : n->outputs()) {
      for (auto node_output_use : node_output->uses()) {
        if (node_output_use.user == n->owningBlock()->return_node()) {
          is_block_output = true;
          break;
        }
      }
      if (is_block_output) {
        break;
      }
    }
    if (is_block_output) {
      continue;
    }
    // find the control flow node in the same block as node_input that contains
    // Node n
    auto control_flow_node = n->owningBlock()->owningNode();
    while (control_flow_node->owningBlock() != node_input->owningBlock())
      control_flow_node = control_flow_node->owningBlock()->owningNode();
    // put the node right before this flow node
    n->moveBefore(control_flow_node);
  }
}

static void replaceInputWithList(Node* node, size_t i, ArrayRef<Value*> to) {
  node->removeInput(i);
  for (auto* to_val : to) {
    AT_ASSERT(to_val->owningGraph() == node->owningGraph());
    node->insertInput(i++, to_val);
  }
}

static void eraseListConstruct(Block* block, int opset_version);

static void eraseListConstruct(Node* n, int opset_version) {
  for (auto b : n->blocks()) {
    eraseListConstruct(b, opset_version);
  }
  std::vector<std::tuple<size_t, std::vector<Value*>>> replacements;

  auto block = n->owningBlock();
  size_t i = 0;
  for (auto* input : n->inputs()) {
    if (input->node()->kind() == prim::ListConstruct) {
      auto* lc_node = input->node();
      TypePtr elem =
          lc_node->output()->type()->castRaw<ListType>()->getElementType();
      if (elem->cast<IntType>()) {
        // ListConstruct Int[] output case, we need to transform to ONNX
        // Concat to ensure the output is a single tensor(dynamic) type in
        // order to be consumed as inputs
        std::vector<Value*> unsqueezed;
        Graph* g = block->owningGraph();
        for (auto* input : lc_node->inputs()) {
          Node* unsqueezed_node =
              createONNXUnsqueeze(g, lc_node, input, 0, opset_version);
          unsqueezed.emplace_back(unsqueezed_node->output());
        }
        Node* concat_node = g->create(onnx::Concat, 1);
        concat_node->i_(attr::axis, 0);
        for (auto v : unsqueezed) {
          concat_node->addInput(v);
        }
        concat_node->insertBefore(lc_node);

        // make concat node output as new input, then ListConstruct should
        // become dead
        replacements.emplace_back(
            i, std::vector<Value*>({concat_node->output()}));

      } else {
        if (opset_version >= OPSET_VERSION_11) {
          c10::Symbol seq_node_kind = lc_node->inputs().size() > 0
              ? onnx::SequenceConstruct
              : onnx::SequenceEmpty;
          Node* seq_node = block->owningGraph()->create(
              seq_node_kind, {lc_node->inputs()}, 1);
          seq_node->insertBefore(lc_node);
          seq_node->output()->copyMetadata(lc_node->output());
          lc_node->replaceAllUsesWith(seq_node);
        }
      }
    }
    i++;
  }

  for (auto ritr = replacements.rbegin(); ritr != replacements.rend(); ++ritr) {
    replaceInputWithList(n, std::get<0>(*ritr), std::get<1>(*ritr));
  }
}

static void eraseListConstruct(Block* block, int opset_version) {
  // TODO: Fix this pass/maybe get rid of this part.
  // Tensor lists might be used for meshgrid and such ops as well.
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* n = *it;
    ++it;

    eraseListConstruct(n, opset_version);
  }
  eraseListConstruct(block->return_node(), opset_version);
}

// For ops such as meshgrid where output is a list of Tensors
// (returns prim::ListConstruct), we need to unpack the list
// before the pass which deletes ListConstruct.
static void fuseListConstructListUnpack(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      fuseListConstructListUnpack(child_block);
    }
    if (it->kind() == prim::ListUnpack &&
        it->input()->node()->kind() == prim::ListConstruct) {
      for (size_t i = 0; i < it->outputs().size(); i++) {
        auto output = it->outputs().at(i);
        output->replaceAllUsesWith(it->input()->node()->inputs().at(i));
      }
    }
  }
}

void removeMaxPoolUnusedOutput(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    auto n = *it;
    for (auto* child_block : n->blocks()) {
      removeMaxPoolUnusedOutput(child_block);
    }
    if (strcmp(n->kind().toQualString(), "onnx::MaxPool") == 0) {
      if (n->outputs().size() == 2 && n->outputs().at(1)->uses().empty()) {
        it->eraseOutput(1);
      }
    }
  }
}

// This optimization fuses LogSoftmax and NegativeLogLikelihoodLoss operators
// into one operator: SoftmaxCrossEntropyLoss, and depending on the dimensions
// of the input and different attributes there will be different subgraphs of
// LogSoftmax and NegativeLogLikelihoodLoss.
static void fuseLogSoftmaxNllLoss(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      fuseLogSoftmaxNllLoss(child_block);
    }
    if (it->kind() == onnx::NegativeLogLikelihoodLoss) {
      auto prev = it->input(0)->node();
      Node* origNllLossNode = *it;
      Node* origLogSoftmaxNode;

      // Check for patterns especially in cases with autocasting enabled
      // in which a cast node is inserted before the NegativeLogLikelihoodLoss
      // node and this causes the patterns below not to be recognizable by the
      // fuseLogSoftmaxNllLoss function
      // For example if the input is 2D
      // graph(%input : Half(3, 5),
      // %target : Long(3)):
      // %4 : Half(3, 5) = onnx::LogSoftmaxaxis=1
      // %8 : Float = onnx::Cast[to=1](%4)
      // %9 : Float(3) = onnx::NegativeLogLikelihoodLoss[reduction="none"]
      // return (%8)
      Node* castNode = nullptr;
      if (prev->kind() == onnx::Cast) {
        castNode = prev;
        prev = prev->input(0)->node();
      }

      if (prev->kind() == onnx::LogSoftmax) {
        // if the input is 2D
        // graph(%input : Float(3, 5),
        // %target : Long(3)):
        // %4 : Float(3, 5) = onnx::LogSoftmaxaxis=1
        // %8 : Float(3) = onnx::NegativeLogLikelihoodLoss[reduction="none"]
        // return (%8)
        origLogSoftmaxNode = prev;
      } else if (
          prev->kind() == onnx::Transpose &&
          prev->input(0)->node()->kind() == onnx::LogSoftmax) {
        // if the input is 4D
        // graph(%input : Float(3, 5, 2, 7),
        // %target : Long(3, 2, 7)):
        // %4 : Tensor = onnx::Transpose[perm=[0, 3, 2, 1]] (%input)
        // %5 : Tensor = onnx::LogSoftmax[axis=3] (%4)
        // %6 : Float(3, 5, 2, 7) = onnx::Transpose[perm=[0, 3, 2, 1]] (%5)
        // %10 : Float(3, 2, 7) =
        // onnx::NegativeLogLikelihoodLoss[reduction="none"](%6, %target) return
        // (%10)
        origLogSoftmaxNode = prev->input(0)->node();
        auto transpose = origLogSoftmaxNode->input(0)->node();
        if (transpose->inputs().size() > 0)
          origLogSoftmaxNode->replaceInput(0, transpose->inputs().at(0));
      } else if (
          prev->kind() == onnx::Reshape &&
          prev->input(0)->node()->kind() == onnx::Transpose &&
          prev->input(0)->node()->input(0)->node()->kind() ==
              onnx::LogSoftmax) {
        // if the input is 3D or > 4D
        // graph(%input : Float(3, 5, 2),
        // %target.1 : Long(3, 2)):
        // %4 : Tensor = onnx::Transpose[perm=[0, 2, 1]] (%input)
        // %5 : Tensor = onnx::LogSoftmax[axis=2] (%4)
        // %6 : Float(3, 5, 2) = onnx::Transpose[perm=[0, 2, 1]] (%5)
        // %8 : Tensor = onnx::Shape(%6)
        // %10 : Tensor = onnx::Constantvalue={0}
        // %11 : Long() = onnx::Gather[axis=0] (%8, %10)
        // %13 : Tensor = onnx::Shape(%6)
        // %15 Tensor = onnx::Constantvalue={1}
        // %16 : Long() = onnx::Gather[axis=0] (%13, %15)
        // ...
        // %22 : Float(3, 5, 1, 2) = onnx::Reshape(%6, %21)
        // ...
        // %26 : Long(3, 1, 2) = onnx::Reshape(%target.1, %25)
        // %30 : Float() = onnx::NegativeLogLikelihoodLoss[reduction="sum"](%22,
        // %26) return (%30)
        TORCH_INTERNAL_ASSERT(
            prev->input(1)->node()->input(0)->node()->kind() == onnx::Gather);
        TORCH_INTERNAL_ASSERT(
            prev->input(1)->node()->input(1)->node()->kind() == onnx::Gather);
        origLogSoftmaxNode = prev->input(0)->node()->input(0)->node();
        auto transpose = origLogSoftmaxNode->input(0)->node();
        TORCH_INTERNAL_ASSERT(transpose->kind() == onnx::Transpose);
        origLogSoftmaxNode->replaceInput(0, transpose->inputs().at(0));
        auto reshape = origNllLossNode->input(1)->node();
        TORCH_INTERNAL_ASSERT(reshape->kind() == onnx::Reshape);
        origNllLossNode->replaceInput(1, reshape->inputs().at(0));
        if (origNllLossNode->s(attr::reduction) == "none") {
          // when reduction=none a different graph is created and the graph
          // doesn't end with node NegativeLogLikelihoodLoss like in all other
          // cases.
          // graph(%input : Float(3, 5, 2), %target.1 : Long(3, 2)):
          // %4 : Tensor = onnx::Transposeperm=[0, 2, 1]
          // %5 : Tensor = onnx::LogSoftmaxaxis=2
          // %6 : Float(3, 5, 2) = onnx::Transposeperm=[0, 2, 1]
          // ...
          // %27 : Float(3, 5, 1, 2) = onnx::Reshape(%6, %26)
          // %31 : Long(3, 1, 2) = onnx::Reshape(%target.1, %30)
          // %35 : Float(3, 1, 2) =
          // onnx::NegativeLogLikelihoodLoss[reduction="none"](%27, %31) %36 :
          // int[] = prim::ListConstruct(%11, %21) %37 : Float(3, 2) =
          // onnx::Reshape(%35, %36) return (%37)
          auto nllloss_output = origNllLossNode->output(0)->uses()[0].user;
          TORCH_INTERNAL_ASSERT(nllloss_output->kind() == onnx::Reshape);
          // make output of reshape the output of nllloss
          nllloss_output->replaceAllUsesWith(origNllLossNode);
          origNllLossNode->output(0)->copyMetadata(nllloss_output->output(0));
        }
      } else {
        continue;
      }

      // If the pattern indeed consists of a cast node before the
      // NegativeLogLikelihoodLoss node, place a cast node in the beginning
      // of the pattern instead
      if (castNode != nullptr) {
        auto onnx_type = castNode->i(attr::to);
        Node* cast_node = b->owningGraph()->create(onnx::Cast, 1);
        cast_node->addInput(origLogSoftmaxNode->inputs().at(0));
        cast_node->i_(attr::to, onnx_type);
        cast_node->insertBefore(origLogSoftmaxNode);
        origLogSoftmaxNode->replaceInputWith(
            origLogSoftmaxNode->inputs().at(0), cast_node->output());
      }

      Node* softmaxCrossEntropyNode = b->owningGraph()->create(
          onnx::SoftmaxCrossEntropyLoss, it->outputs().size());
      for (size_t i = 0; i < softmaxCrossEntropyNode->outputs().size(); ++i) {
        softmaxCrossEntropyNode->outputs()[i]->copyMetadata(it->outputs()[i]);
      }
      softmaxCrossEntropyNode->copyAttributes(*origNllLossNode);
      softmaxCrossEntropyNode->insertBefore(origNllLossNode);
      softmaxCrossEntropyNode->addInput(origLogSoftmaxNode->inputs().at(0));
      softmaxCrossEntropyNode->addInput(origNllLossNode->inputs().at(1));
      // optional weight input is provided
      if (origNllLossNode->inputs().size() == 3) {
        softmaxCrossEntropyNode->addInput(origNllLossNode->inputs().at(2));
      }

      it->replaceAllUsesWith(softmaxCrossEntropyNode);
      it->removeAllInputs();
      it.destroyCurrent();
    }
  }
}

// This optimization removes consecutive SplitToSequence and ConcatFromSequence
// operators. The optimization only happens when
//  1. Output of SplitToSequence is not used by any other nodes.
//  2. The attribute keepdims and axis of SplitToSequence match
//     attribute new_axis and axis of ConcatFromSequence.
// In that case, the two ops combined are no-op, and can be safely removed.
static void removeSequenceSplitConcat(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      removeSequenceSplitConcat(child_block);
    }
    if (it->kind() == onnx::ConcatFromSequence &&
        it->input()->node()->kind() == onnx::SplitToSequence) {
      if (it->input()->uses().size() > 1) {
        continue;
      }

      auto split_node = it->input()->node();
      auto concat_node = *it;

      const auto split_axis =
          split_node->hasAttribute(attr::axis) ? split_node->i(attr::axis) : 0;
      const auto split_keepdims = split_node->hasAttribute(attr::keepdims)
          ? split_node->i(attr::keepdims)
          : 1;
      const auto concat_axis = concat_node->i(attr::axis);
      const auto concat_new_axis = concat_node->hasAttribute(attr::new_axis)
          ? concat_node->i(attr::new_axis)
          : 0;
      const bool has_input_split = split_node->inputs().size() == 2;

      if (has_input_split) {
        continue;
      }

      if (split_keepdims == concat_new_axis) {
        continue;
      }

      if (split_axis != concat_axis) {
        continue;
      }

      concat_node->output()->replaceAllUsesWith(split_node->input());
    }
  }
}

// This optimization does ONNX-specific peephole optimizations.
//
// At the moment, here are the optimizations it does:
//  - This optimization fuses expand calls into ONNX operators, because it is
//    easier for non-strided backends to more efficiently do broadcasts if this
//    is local information.  This optimization is not useful for PyTorch as
//    'expand' is free.
//  - Fusing of consecutive transposes
//  - Elimination of NOP transposes
//  - Fusing of transposes into Gemm
//  - Elimination of PaddedSequences
//
// Before you write an optimization here, ask yourself, "Could I do this
// optimization on ATen operators"?  If so, you should seriously consider
// writing your optimization in jit/passes/peephole.cpp rather than
// here, as it will be generally applicable to the JIT as well.  The
// optimizations here are ONLY applied on ONNX update
void PeepholeOptimizeONNX(
    std::shared_ptr<Graph>& graph,
    int opset_version,
    bool fixed_batch_size) {
  // TODO: decide on fixpoint strategy
  // TODO: make it easier not to do O(k) iterations over the graph, where
  // k is the number of distinct peephole optimizations
  hackFixupPadPackedShapes(graph->block());
  pushPackingPastRnn(graph->block());
  removeNopPacking(graph->block());
  // we only need to fix the size of hidden state and cell state if the batch
  // size is variable
  if (!fixed_batch_size) {
    fixDefaultRnnHiddenState(graph->block(), opset_version);
    fixDefaultLstmCellState(graph->block(), opset_version);
  }
  fuseBroadcast(graph->block());
  fuseConsecutiveTransposes(graph->block());
  eliminateNopTranspose(graph->block());
  fuseTransposeIntoGemm(graph->block());
  speculateOps(graph->block());
  fuseListConstructListUnpack(graph->block());
  fuseLogSoftmaxNllLoss(graph->block());
  eraseListConstruct(graph->block(), opset_version);
  removeMaxPoolUnusedOutput(graph->block());
  removeSequenceSplitConcat(graph->block());
}

} // namespace jit
} // namespace torch
