#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/autodiff.h"
#include "torch/csrc/jit/assertions.h"
#include "ATen/ExpandUtils.h"
#include <unordered_map>

#ifdef USE_CUDA
  #include "cuda.h" // for CUDA_VERSION
#endif

namespace torch { namespace jit {

namespace {

// What is a simple mappable operator?  It:
//    - Has a single tensor output
//    - Output and all tensor inputs have the same shape
//    - Output and all tensor inputs have the same scalar type
//    - Output and all tensor inputs should be on the same device
//    - Produces contiguous outputs
// Some of these restrictions may be relaxable, but you should
// carefully read the code first, as we rely on these assumptions.
bool isSimpleMap(Node *node) {
  static OperatorSet simple_mappable {{
    "aten::abs(Tensor self) -> Tensor",
    "aten::acos(Tensor self) -> Tensor",
    "aten::add(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
    "aten::asin(Tensor self) -> Tensor",
    "aten::atan(Tensor self) -> Tensor",
    "aten::atan2(Tensor self, Tensor other) -> Tensor",
    "aten::ceil(Tensor self) -> Tensor",
    "aten::clamp(Tensor self, Scalar? min, Scalar? max) -> Tensor",
    "aten::cos(Tensor self) -> Tensor",
    "aten::cosh(Tensor self) -> Tensor",
    "aten::div(Tensor self, Tensor other) -> Tensor",
    "aten::exp(Tensor self) -> Tensor",
    "aten::expm1(Tensor self) -> Tensor",
    "aten::floor(Tensor self) -> Tensor",
    "aten::fmod(Tensor self, Tensor other) -> Tensor",
    "aten::frac(Tensor self) -> Tensor",
    "aten::lgamma(Tensor self) -> Tensor",
    "aten::log(Tensor self) -> Tensor",
    "aten::log10(Tensor self) -> Tensor",
    "aten::log1p(Tensor self) -> Tensor",
    "aten::log2(Tensor self) -> Tensor",
    "aten::max(Tensor self, Tensor other) -> Tensor",
    "aten::min(Tensor self, Tensor other) -> Tensor",
    "aten::mul(Tensor self, Tensor other) -> Tensor",
    "aten::neg(Tensor self) -> Tensor",
    "aten::pow(Tensor self, Tensor exponent) -> Tensor",
    "aten::rand_like(Tensor self) -> Tensor",
    "aten::reciprocal(Tensor self) -> Tensor",
    "aten::relu(Tensor self) -> Tensor",
    "aten::remainder(Tensor self, Tensor other) -> Tensor",
    "aten::round(Tensor self) -> Tensor",
    "aten::rsqrt(Tensor self) -> Tensor",
    "aten::sigmoid(Tensor self) -> Tensor",
    "aten::sin(Tensor self) -> Tensor",
    "aten::sinh(Tensor self) -> Tensor",
    "aten::sqrt(Tensor self) -> Tensor",
    "aten::sub(Tensor self, Tensor other, *, Scalar alpha) -> Tensor",
    "aten::tan(Tensor self) -> Tensor",
    "aten::tanh(Tensor self) -> Tensor",
    "aten::trunc(Tensor self) -> Tensor",
    "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
    "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor",
    "aten::mul(Tensor self, Scalar other) -> Tensor",
    "aten::div(Tensor self, Scalar other) -> Tensor",
  }};
  if (!simple_mappable.find(node)) {
    return false;
  }
  // Check that all non-tensor inputs are constant
  for (Value * input : node->inputs()) {
    if (input->type()->isSubtypeOf(DynamicType::get())) {
      continue;
    }
    if (input->node()->kind() != prim::Constant) {
      return false;
    }
  }
  return true;
}

struct GraphFuser {
  Block * block;

  GraphFuser(Block * block)
    : block(block) {}

  value_list tensorInputs(Node * node) {
    return filter(node->inputs(), [](Value * v) {
      return v->type()->isSubtypeOf(DynamicType::get());
    });
  }

  bool isFusable(Node * node) {
    // We don't want to bother with cross-block node movements, as they
    // are not necessarily correct.
    if (node->owningBlock() != block) return false;
    return node->kind() == prim::FusionGroup || isSimpleMap(node);
  }

  bool isFusableCatNode(Node * node) {
    if (node->kind() != aten::cat)
      return false;
    if (!node->is_constant(attr::dim))
      return false;
    auto tensors_node = node->namedInput(attr::tensors)->node();
    if (tensors_node->kind() != prim::ListConstruct) return false;
    // NB: Note that technically other uses of the list aren't a big problem for us.
    // It would be enough to place the prim::FusedConcat before the prim::ListConstruct, and
    // allUsersAreThisConsumerOrOccurAfterIt would still be satisfied. However, I don't expect this
    // to be necessary any time soon, and so we're simply assuming that we don't have to deal with it.
    if (tensors_node->output()->uses().size() > 1) return false;
    return true;
  }

  // Can this node produce an _output_ of a fusion group?
  // all Fusable nodes can do this, but additionally Concat, which normally cannot be fused
  // because it is not a simple map, can be put in a fusion group
  // as long as no items in the group read the output of concat
  bool isFusableAsExitNode(Node * node) {
    return isFusable(node) || isFusableOnlyAsExitNode(node);
  }

  bool isFusableOnlyAsExitNode(Node * node) {
    return isFusableCatNode(node) || node->kind() == prim::FusedConcat;
  }

  bool allUsersAreThisConsumer(Node * consumer, Value * producer) {
    auto defining_node = producer->node();
    for(auto o : defining_node->outputs()) {
      for(auto u : o->uses()) {
        if(u.user != consumer)
          return false;
      }
    }
    return true;
  }

  bool mustRemainAsFusionGroupOutput(Value * producer) {
    if (producer->node()->kind() != prim::FusionGroup) {
      return false;
    }
    auto subgraph = producer->node()->g(attr::Subgraph);
    auto * node = subgraph->outputs().at(producer->offset())->node();
    return isFusableOnlyAsExitNode(node);
  }

  Graph & getSubgraph(Node * n) {
    JIT_ASSERT(n->kind() == prim::FusionGroup);
    return *n->g(attr::Subgraph);
  }

  void mergeFusionGroups(Node *consumer_group, Node *producer_group) {
    // Now we have two fusion groups!
    // Revert the fusion - place all inner nodes of producer back in the outer graph.
    std::vector<Node*> temporary_nodes;
    auto producer_subgraph = &getSubgraph(producer_group);

    // Initialize a map of inner graph values to outer graph values
    std::unordered_map<Value*, Value*> inner_to_outer;
    auto inner_inputs = producer_subgraph->inputs();
    auto outer_inputs = producer_group->inputs();
    for (size_t i = 0; i < inner_inputs.size(); ++i) {
      inner_to_outer[inner_inputs[i]] = outer_inputs[i];
    }

    // Clone all nodes
    for (auto inner : producer_subgraph->nodes()) {
      Node * outer = block->owningGraph()->createClone(inner, [&](Value * k) -> Value* {
        return inner_to_outer.at(k);
      });
      outer->insertBefore(producer_group);
      temporary_nodes.emplace_back(outer);
      auto inner_outputs = inner->outputs();
      auto outer_outputs = outer->outputs();
      for (size_t i = 0; i < inner_outputs.size(); ++i)
        inner_to_outer[inner_outputs[i]] = outer_outputs[i];
    }

    // Replace uses of producer_group outputs and destroy the producer
    auto subgraph_outputs = producer_subgraph->outputs();
    for (size_t i = 0; i < subgraph_outputs.size(); ++i) {
      auto outer_output = inner_to_outer.at(subgraph_outputs[i]);
      producer_group->outputs()[i]->replaceAllUsesWith(outer_output);
    }
    producer_group->destroy();
    producer_group = nullptr; // Just to get a clear error in case someone uses it

    // Inline the temporary nodes into the first group
    auto consumer_subgraph = &getSubgraph(consumer_group);
    for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend(); ++it) {
      Node *node = *it;
      Node *merged = mergeNodeIntoGroup(consumer_group, node);
      // If any of the outputs are still used then we need to add them
      auto outputs = node->outputs();
      for (size_t i = 0; i < outputs.size(); ++i) {
        auto output = outputs[i];
        if (output->uses().size() == 0) continue;
        consumer_subgraph->registerOutput(merged->outputs()[i]);
        auto new_output = consumer_group->addOutput();
        output->replaceAllUsesWith(new_output);
        new_output->setType(output->type());
      }
      node->destroy();
    }
  }

  // insert a producer node into a consuming fusion group.
  // DOES NOT WORK if n is a consumer of an output of the fusion group
  // returns the node _inside_ the group that represents the node
  Node * mergeNodeIntoGroup(Node* group, Node * n) {
    JIT_ASSERT(n->kind() != prim::FusionGroup);
    auto & subgraph = getSubgraph(group);
    // map from nodes in the surrounding graph to parameters in the fusion
    // group's subgraph that correspond to them
    std::unordered_map<Value*,Value*> inputs_map;
    size_t i = 0;
    JIT_ASSERT(group->inputs().size() == subgraph.inputs().size());
    for(auto input : group->inputs()) {
      inputs_map[input] = subgraph.inputs()[i++];
    }
    // add n's inputs to the fusion group's input list if we don't already have them
    WithInsertPoint guard(*subgraph.nodes().begin());
    for (auto input : n->inputs()) {
      if (inputs_map.count(input) == 0) {
        if (input->type()->isSubtypeOf(DynamicType::get())) {
          auto in_group = subgraph.addInput();
          in_group->setType(input->type());
          inputs_map[input] = in_group;
          group->addInput(input);
        } else {
          // We don't support passing in scalars as arguments to fused kernels, so we generally
          // don't allow fusing tensor-scalar operations unless the scalar is constant. In those
          // cases we inline the constants directly in the body of the fused group.
          JIT_ASSERT(input->node()->kind() == prim::Constant);
          Node * in_const = subgraph.createClone(input->node(), [](Value*) -> Value* { throw std::runtime_error("unexpected input"); });
          subgraph.insertNode(in_const);
          inputs_map[input] = in_const->output();
        }
      }
    }
    // copy n into the graph, remapping its inputs to internal nodes
    Node * in_graph = subgraph.createClone(n,[&](Value * k)-> Value* {
      return inputs_map[k];
    });
    // if n's outputs are already inputs to the fusion group,
    // we need to remove them because n is now inside the fusion group.
    //
    // i.e.,
    // x = f(w); group(x, y, z) becomes group(w, y, z).
    // x, y, z = f(w); group(x, y, z) becomes group(w).
    //
    // remapping nodes that used the input to the newly-merged node
    // n is not an input when the fusion group is empty
    auto inputs = group->inputs();
    for (size_t i = 0; i < n->outputs().size(); ++i) {
      auto it = std::find(inputs.begin(), inputs.end(), n->outputs()[i]);
      if(it != inputs.end()) {
        size_t p = it - inputs.begin();
        group->removeInput(p);
        subgraph.inputs()[p]->replaceAllUsesWith(in_graph->outputs()[i]);
        subgraph.eraseInput(p);
      }
    }
    return subgraph.insertNode(in_graph);
  }

  // turn consumer node n into a fusion group with just n inside
  // to prepare for fusion and replace uses of n with the new group
  Node * createSingletonFusionGroup(Node * n) {
    auto group = block->owningGraph()->createFusionGroup();
    // propogate position information for the new node so we can always
    // have a valid mapping
    group->insertBefore(n);
    Node * mergedNode = mergeNodeIntoGroup(group,n);
    getSubgraph(group).registerOutput(mergedNode->output());
    auto sel = group->addOutput();
    sel->copyMetadata(n->output());
    n->replaceAllUsesWith(group);
    n->destroy();
    return group;
  }

  // TODO: remove this and use WithInsertPoint instead
  void insertAt(Node ** insertion_point, Node * n) {
    n->insertAfter(*insertion_point);
    *insertion_point = n;
  }

  at::optional<Node*> tryFuse(Node * consumer, Value * producer) {
    // this handles cases where producer can be moved _into_ the fusion group of consumer.
    // TODO: extend to fusion of consumer into _producer's_ fusion blob
    // if the consumer allInputsAreThisProducer(consumer,producer)
    // we can move the consumer up into the producer.
    // but this requires better handling of merging fusion groups so it is not done now
    Node* real_consumer = consumer->kind() == aten::cat
        ? consumer->namedInput(attr::tensors)->node()
        : consumer;
    bool shouldFuse = isFusable(producer->node()) &&
        // Rearrange nodes such that all uses of producer are after the
        // consumer. Fusion will rewrite those later uses to use the version of
        // producer generated by the fused blob. In this case, producer becomes
        // an output of the fusion group.
        producer->node()->moveBeforeTopologicallyValid(real_consumer);

    if (!shouldFuse) {
      return at::nullopt;
    }

    auto group = consumer;
    if (consumer->kind() == aten::cat) {
      Graph * graph = consumer->owningGraph();
      Node * list_construct = consumer->namedInput(attr::tensors)->node();
      int64_t dim = consumer->get<int64_t>(attr::dim).value();

      Node * fused_cat = graph->create(prim::FusedConcat, list_construct->inputs())->i_(attr::dim, dim);
      fused_cat->insertBefore(list_construct);
      fused_cat->output()->copyMetadata(consumer->output());
      consumer->output()->replaceAllUsesWith(fused_cat->output());

      // NB: this deletes the fused_cat node from the original graph
      group = createSingletonFusionGroup(fused_cat);
      consumer->destroy();
      if (list_construct->output()->uses().empty()) {
        list_construct->destroy();
      }
    } else if (consumer->kind() != prim::FusionGroup) {
      group = createSingletonFusionGroup(consumer);
    }
    if (producer->node()->kind() == prim::FusionGroup) {
      mergeFusionGroups(group, producer->node());
      return group;
    }
    JIT_ASSERT(producer->node()->outputs().size() == 1);
    Node * merged = mergeNodeIntoGroup(group, producer->node());
    // remaining uses of this producer can occur because we allow
    // fusion in cases where uses remain after the consumer
    // if these exist, re-route them to the version of producer
    // created in FusionGroup
    if(producer->uses().size() != 0) {
      getSubgraph(group).registerOutput(merged->output());
      Value * new_producer = group->addOutput();
      new_producer->copyMetadata(producer);
      producer->replaceAllUsesWith(new_producer);
    }
    producer->node()->destroy();
    return group;
  }

  bool canFuseChunk(Node* consumer, Value* producer) {
    if (consumer->kind() != prim::FusionGroup) {
      return false;
    }
    // Does the chunk have constant chunks/dim?
    auto * chunk = producer->node();
    if (chunk->kind() != prim::ConstantChunk)
        return false;
    // And all uses of the chunk are in this consumer
    for (auto s : chunk->outputs()) {
      for (auto u : s->uses()) {
        if (u.user != consumer) {
          return false;
        }
      }
    }
    // And isn't a no-op chunk (chunks == 1). Have CSE clean this up.
    // We could fuse this but it's better to just delete the node.
    if (chunk->i(attr::chunks) == 1) {
      return false;
    }
    return true;
  }

  c10::optional<Node*> findFusedChunk(Node* group, Value* input) {
    JIT_ASSERT(group->kind() == prim::FusionGroup);
    auto it = std::find(group->inputs().begin(), group->inputs().end(), input);
    if (it == group->inputs().end()) {
      return c10::nullopt;
    }
    size_t input_index = it - group->inputs().begin();
    auto & subgraph = getSubgraph(group);
    auto * subgraph_input = subgraph.inputs().at(input_index);
    // If subgraph_input is an input to prim::ConstantChunk, it will have 1 use
    auto * node = subgraph_input->uses().at(0).user;
    if (node->kind() == prim::ConstantChunk) {
      JIT_ASSERT(subgraph_input->uses().size() == 1);
      return node;
    }
    return c10::nullopt;
  }

  void fuseChunkByReusingExistingFusedChunk(
      Node * group, Node * chunk, Node * existingFusedChunk) {
    if (chunk->outputs().size() != existingFusedChunk->outputs().size()) {
      return;
    }
    auto & subgraph = getSubgraph(group);
    for (size_t i = 0; i < chunk->outputs().size(); ++i) {
      // Find the input to the FusionGroup (group)
      auto * replacement_val = existingFusedChunk->outputs().at(i);
      auto * val = chunk->outputs().at(i);
      auto it = std::find(group->inputs().begin(), group->inputs().end(), val);
      auto input_index = it - group->inputs().begin();

      // Rewrite the graph to use replacement_val
      auto group_input = subgraph.inputs().at(input_index);
      group_input->replaceAllUsesWith(replacement_val);

      // Remove the input, it's no longer needed
      group->removeInput(input_index);
      subgraph.eraseInput(input_index);
    }
    chunk->destroy();
  }

  // There are two invariants for prim::ConstantChunk:
  // (1) the tensor input to prim::ConstantChunk must be an input to the fusion group
  // (2) no two ConstantChunks in the same FusionGroup can share a tensor input.
  graph_node_list::iterator fuseChunk(Node * consumer, Value * producer) {
    auto * chunk = producer->node();
    JIT_ASSERT(consumer->kind() == prim::FusionGroup);
    JIT_ASSERT(chunk->kind() == prim::ConstantChunk);

    // if producer's input is already an input to a prim::ConstantChunk node,
    // we cannot add a new prim::ConstantChunk node because of invariant (2).
    auto * chunked_tensor = producer->node()->input();
    if (auto existingFusedChunk = findFusedChunk(consumer, chunked_tensor)) {
      fuseChunkByReusingExistingFusedChunk(consumer, chunk, *existingFusedChunk);
      return consumer->reverseIterator();
    }

    // Move prim::ConstantChunk into the FusionGroup
    mergeNodeIntoGroup(consumer, chunk);
    chunk->destroy();
    return consumer->reverseIterator();
  }

  value_list sortReverseTopological(ArrayRef<Value*> inputs) {
    value_list result;
    for (auto i : inputs) {
      if (i->node()->owningBlock() == block) {
        result.push_back(i);
      }
    }
    // Sort in reverse topological order
    std::sort(result.begin(), result.end(), [&](Value * a, Value * b) {
      return a->node()->isAfter(b->node());
    });
    return result;
  }

  graph_node_list::iterator scanNodeForChunks(Node * consumer) {
    if (consumer->kind() == prim::FusionGroup) {
      auto inputs = sortReverseTopological(consumer->inputs());
      for(auto producer : inputs) {
        if (!canFuseChunk(consumer, producer)) {
          continue;
        }
        return fuseChunk(consumer, producer);
      }
    }
    return ++consumer->reverseIterator();
  }

  void insertExplicitBroadcast(Node *node) {
    WithInsertPoint insert_guard { node };
    auto tensors = tensorInputs(node);
    auto new_tensors = SymbolicVariable::broadcast_tensors(fmap<SymbolicVariable>(tensors));

    // Replace tensors inputs with broadcasted values
    auto new_tensors_it = new_tensors.begin();
    for (size_t i = 0; i < node->inputs().size(); ++i) {
      if (node->inputs()[i]->type()->isSubtypeOf(DynamicType::get())) {
        JIT_ASSERT(new_tensors_it != new_tensors.end());
        node->replaceInput(i, *(new_tensors_it++));
      }
    }
  }


  // in places where op can be fused into a consumer but chunk is in the way
  // distribute chunk to op's operands:
  // replace a,b = chunk(op(x,y,z)) with:
  // x0,x1 = chunk(x) (x0 has a's type, x1 has b's type)
  // y0,y1 = chunk(y) (y0 has a's type, y1 has b's type)
  // z0,z1 = chunk(z) (z0 has a's type, z1 has b's type)
  // a = op(x0,y0,z0) (a,b have their same size but are now contiguous)
  // b = op(x1,y1,x1)
  //
  // NB: Chunk motion only occurs with fusable consumers, which implies
  // that there is always some other operation, e.g., a+b, that happens
  // after the chunk, and will be put into the fusion group. This is
  // important, because distributing the chunk changes the contiguity
  // of a and b, and so the results would be invalid, except that we know
  // that simple_mappable operations will restore contiguity before
  // we exit the fusion group.

  bool tryToMoveChunk(Node * consumer, Value * producer) {
    // is the output from a chunk node?
    auto * chunk = producer->node();
    if (chunk->kind() != prim::ConstantChunk)
      return false;
    // and the thing being chunked is fusable into the consumer
    Value * producer_for_chunk = chunk->input();
    if (!isFusable(producer_for_chunk->node()) ||
        !allUsersAreThisConsumer(chunk,producer_for_chunk))
      return false;
    // and all uses of the chunk are in this consumer
    for (auto s : chunk->outputs()) {
      for (auto u : s->uses()) {
        if (u.user != consumer)
          return false;
      }
    }
    // multiple return operators
    Node * producer_for_chunk_node = producer_for_chunk->node();
    JIT_ASSERT(producer_for_chunk_node->outputs().size() == 1);

    // First, we'll add explicit broadcasts where necessary to make the chunk
    // valid. Let's say we have:
    // %z = aten::mul(%x, %y)
    // %z.1, %z.2 = aten::chunk(%z, ...)
    // ... = prim::FusionGroup(%z.1, %z.2, ...)
    // It's possible that %x and %y do not have the same size as %z and
    // need to be expanded first so that they can be chunked like %z
    insertExplicitBroadcast(producer_for_chunk_node);

    // Make sure we lay out the nodes in the correct topological order.
    // TODO: There should be some more enshrined way to do this
    Node * insertion_point = chunk;

    // apply chunk to each of op's operands
    // chunked_inputs[input_nr][chunk_output_idx]
    //  = Node* for chunk_output_idx'th output of the chunk(inputs[input_nr])
    std::vector<std::vector<Value*>> chunked_inputs;
    for (auto input : producer_for_chunk_node->inputs()) {
      // XXX: we only work with pointwise ops in here, so we know it is valid to push
      // the concat only through tensor arguments (and all other args can be safely ignored).
      if (!input->type()->isSubtypeOf(DynamicType::get()))
        continue;
      // NB: I decided not to use cloneFrom here, because if we make cloneFrom
      // copy selects one day, it is definitely not what you want here (selects
      // have different types).
      // TODO: Perhaps we should use cloneFrom now, as it seems unlikely
      // to copy select nodes now that we have refactored to have a Value
      // distinct from Node.
      Node * input_chunk = block->owningGraph()->create(prim::ConstantChunk, 0);
      input_chunk->addInput(input);
      input_chunk->i_(attr::chunks, chunk->i(attr::chunks));
      input_chunk->i_(attr::dim, chunk->i(attr::dim));
      insertAt(&insertion_point, input_chunk);

      chunked_inputs.emplace_back(); // alas, to not be C++17
      for (auto chunk_sel : chunk->outputs()) {
          Value * input_chunk_sel = input_chunk->addOutput();
          input_chunk_sel->setType(chunk_sel->type());
          chunked_inputs.back().push_back(input_chunk_sel);
      }
    }

    // apply the op to each chunk of the chunked operands,
    // and then rewrite the graph to use them!
    for (auto chunk_sel : chunk->outputs()) {
      auto original_inputs = producer_for_chunk_node->inputs();
      Node * chunked_op = block->owningGraph()->create(producer_for_chunk_node->kind());
      chunked_op->copyAttributes(*producer_for_chunk_node);
      chunked_op->output()->setType(chunk_sel->type());
      auto chunked_inputs_it = chunked_inputs.begin();
      for (Value* original_input : original_inputs) {
        if (original_input->type()->isSubtypeOf(DynamicType::get())) {
          JIT_ASSERT(chunked_inputs_it != chunked_inputs.end());
          chunked_op->addInput(chunked_inputs_it->at(chunk_sel->offset()));
          ++chunked_inputs_it;
        } else {
          chunked_op->addInput(original_input);
        }
      }
      insertAt(&insertion_point, chunked_op);
      chunk_sel->replaceAllUsesWith(chunked_op->output());
    }
    chunk->destroy();
    producer_for_chunk_node->destroy();
    return true;
  }

  // returns where to continue scanning, and whether any fusion was made
  std::pair<graph_node_list::iterator, bool> scanNode(Node * consumer) {
    if(isFusableAsExitNode(consumer)) {
      auto consumer_inputs = consumer->kind() == aten::cat ?
        consumer->namedInput(attr::tensors)->node()->inputs() :
        consumer->inputs();
      // handle inputs in reverse topological order as well...
      // otherwise in f(a,a+b) it will appear a is used twice if we consider
      // the f-a fusion before the f-(a+b) fusion first.
      auto inputs = sortReverseTopological(consumer_inputs);
      for(auto producer : inputs) {
        // Don't fuse if producer must come from a FusionGroup exit node
        if (mustRemainAsFusionGroupOutput(producer)) continue;
        if(tryToMoveChunk(consumer,producer)) {
          // the chunk before this consumer was re-arranged to allow fusion,
          // we scan this consumer again to perform the fusion
          return std::make_pair(consumer->reverseIterator(), true);
        }
        auto fusion_group = tryFuse(consumer, producer);
        if (fusion_group) {
          // after fusion, consumer moves into a FusionGroup, so inputs is no
          // longer valid so we rescan the new FusionGroup for more fusions...
          return std::make_pair(fusion_group.value()->reverseIterator(), true);
        }
      }
    }
    return std::make_pair(++consumer->reverseIterator(), false);
  }

  void run() {
    // Run the pass until no changes are made.
    // This is neccessary, because the algorithm can miss out on certain fusion
    // opportunities if ran only once. Consider this graph:
    //
    // %1 = f(...)
    // %2 = g(%1)
    // %3 = h(%1)
    // %4 = l(%3)
    // return (%4, %2)
    //
    // where f, g, h, l are simple map ops.
    // The first iteration will fuse %4 and %3, and see that %1 is an input, but
    // can't be fused, because it has a different use before the fusion group
    // in our topological ordering. Then, %2 will be considered, and fused with %1.
    // If we do another iteration, the algorithm will consider the fusion of these
    // two groups and fix the situation.
    bool any_changed = true;
    while (any_changed) {
      any_changed = false;
      for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
        bool changed;
        std::tie(it, changed) = scanNode(*it);
        any_changed |= changed;
      }
    }
    // Fuse starting chunks into the group.
    for (auto it = block->nodes().rbegin(); it != block->nodes().rend();) {
      it = scanNodeForChunks(*it);
    }
    for (Node * node : block->nodes()) {
      for (Block * sub_block : node->blocks()) {
        GraphFuser(sub_block).run();
      }
    }
  }
};

} // anonymous namespace

void FuseGraph(std::shared_ptr<Graph>& graph) {
  // NYI on Windows
  #ifndef _WIN32

  GraphFuser(graph->block()).run();
  // After FuseGraph some common subexpressions may come back
  EliminateCommonSubexpression(graph);

  #endif
}

}}
