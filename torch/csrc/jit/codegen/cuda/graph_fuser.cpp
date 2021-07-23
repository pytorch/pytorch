#include <torch/csrc/jit/passes/cuda_graph_fuser.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/codegen/cuda/partition.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

#include <queue>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

constexpr size_t NVRTC_KERNEL_ARG_LIMIT = 128;

namespace {

Value* broadcastSizes(at::ArrayRef<Value*> sizes) {
  AT_ASSERT(!sizes.empty());
  Graph* graph = sizes[0]->owningGraph();
  Node* broadcast_n =
      graph->insertNode(graph->create(prim::BroadcastSizes, sizes));
  broadcast_n->output()->setType(ListType::ofInts());
  return broadcast_n->output();
}

struct CudaGraphFuser {
  using FusionCallback = std::function<bool(Node*)>;

  Block* block_;
  std::unique_ptr<AliasDb> aliasDb_;
  std::shared_ptr<Graph> graph_;
  Symbol kind_ = prim::CudaFusionGroup;

  // nvrtc has a limit on the number of arguments allowed in a CUDA kernel.
  // The specific limit is a function of constant memory size, amount available
  // to pass arguments, and some implementation dependence. Select a safe
  // limit here.
  // This limit is also applied to other devices in the fuser by default.
  // Change with setInputArgLimit
  size_t subgraph_arg_limit_ = NVRTC_KERNEL_ARG_LIMIT;

  CudaGraphFuser(Block* block, std::shared_ptr<Graph> graph)
      : block_(block), graph_(std::move(graph)) {}

  void setInputArgLimit(size_t limit) {
    subgraph_arg_limit_ = limit;
  }

  value_list tensorInputs(Node* node) {
    return filter(node->inputs(), [](Value* v) {
      return v->type()->isSubtypeOf(TensorType::get());
    });
  }

  bool calculatesSize(Node* node) {
    return node->matches("aten::size(Tensor self) -> int[]");
  }

  bool allUsersAreThisConsumerOrCalcSizes(Node* consumer, Value* producer) {
    auto defining_node = producer->node();
    for (auto o : defining_node->outputs()) {
      for (auto u : o->uses()) {
        if (u.user != consumer && !calculatesSize(u.user))
          return false;
      }
    }
    return true;
  }

  Graph& getSubgraph(Node* n) {
    AT_ASSERT(n->kind() == kind_);
    return *n->g(attr::Subgraph);
  }

  void mergeFusionGroups(Node* consumer_group, Node* producer_group) {
    // Now we have two fusion groups!
    // Revert the fusion - place all inner nodes of producer back in the outer
    // graph.
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
      Node* outer = block_->owningGraph()->createClone(
          inner, [&](Value* k) -> Value* { return inner_to_outer.at(k); });
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
    producer_group =
        nullptr; // Just to get a clear error in case someone uses it

    // Inline the temporary nodes into the first group
    auto consumer_subgraph = &getSubgraph(consumer_group);
    for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend();
         ++it) {
      Node* node = *it;
      Node* merged = mergeNodeIntoGroup(consumer_group, node);
      // If any of the outputs are still used then we need to add them
      auto outputs = node->outputs();
      for (size_t i = 0; i < outputs.size(); ++i) {
        auto output = outputs[i];
        if (output->uses().size() == 0)
          continue;
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
  Node* mergeNodeIntoGroup(Node* group, Node* n) {
    AT_ASSERT(n->kind() != kind_);
    auto& subgraph = getSubgraph(group);
    // map from nodes in the surrounding graph to parameters in the fusion
    // group's subgraph that correspond to them
    std::unordered_map<Value*, Value*> inputs_map;
    size_t i = 0;
    size_t tensor_insert_idx = 0;
    AT_ASSERT(group->inputs().size() == subgraph.inputs().size());
    for (auto input : group->inputs()) {
      inputs_map[input] = subgraph.inputs()[i++];
      if (input->type()->isSubtypeOf(TensorType::get()))
        tensor_insert_idx = i;
    }
    // add n's inputs to the fusion group's input list if we don't already have
    // them
    // we insert tensors first because the fuser assumes that to be the case
    // (as a legacy from tensors only)
    WithInsertPoint guard(*subgraph.nodes().begin());
    for (auto input : n->inputs()) {
      if (inputs_map.count(input) == 0) {
        // TODO: we are following the convention for no good reason;
        //       we don't need tensor to come before any other inputs.
        if (input->type()->isSubtypeOf(TensorType::get())) {
          auto in_group = subgraph.insertInput(tensor_insert_idx);
          in_group->setType(input->type());
          inputs_map[input] = in_group;
          group->insertInput(tensor_insert_idx, input);
          tensor_insert_idx++;
        } else if (
            // TODO: extend the supporting inputs here.
            (input->type()->isSubtypeOf(FloatType::get()) &&
             input->node()->kind() != prim::Constant) ||
            (n->kind() == aten::_grad_sum_to_size &&
             input->type()->isSubtypeOf(ListType::ofInts()))) {
          auto in_group = subgraph.addInput();
          in_group->setType(input->type());
          inputs_map[input] = in_group;
          group->addInput(input);
        } else if (input->node()->kind() == prim::Constant) {
          // inline the constants directly in the body of the fused group.
          Node* in_const =
              subgraph.createClone(input->node(), [](Value*) -> Value* {
                throw std::runtime_error("unexpected input");
              });
          subgraph.insertNode(in_const);
          inputs_map[input] = in_const->output();
        } else {
          // TODO: we need to figure out what are supported input scalar
          auto in_group = subgraph.addInput();
          in_group->setType(input->type());
          inputs_map[input] = in_group;
          group->addInput(input);
        }
      }
    }
    // copy n into the graph, remapping its inputs to internal nodes
    Node* in_graph = subgraph.createClone(
        n, [&](Value* k) -> Value* { return inputs_map[k]; });
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
      if (it != inputs.end()) {
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
  Node* createSingletonFusionGroup(Node* n) {
    auto group = block_->owningGraph()->createWithSubgraph(kind_);
    // propogate position information for the new node so we can always
    // have a valid mapping
    group->insertBefore(n);
    Node* mergedNode = mergeNodeIntoGroup(group, n);
    getSubgraph(group).registerOutput(mergedNode->output());
    auto sel = group->addOutput();
    sel->copyMetadata(n->output());
    n->replaceAllUsesWith(group);
    n->destroy();
    return group;
  }

  at::optional<Node*> tryFuse(Node* consumer, Value* producer) {
    // this handles cases where producer can be moved _into_ the fusion group of
    // consumer.
    // TODO: extend to fusion of consumer into _producer's_ fusion blob
    // if the consumer allInputsAreThisProducer(consumer,producer)
    // we can move the consumer up into the producer.
    // but this requires better handling of merging fusion groups so it is not
    // done now
    bool shouldFuse =
        fuser::cuda::isFusableCudaFusionGroup(consumer, producer->node()) &&
        // Rearrange nodes such that all uses of producer are after the
        // consumer. Fusion will rewrite those later uses to use the version of
        // producer generated by the fused blob. In this case, producer becomes
        // an output of the fusion group.
        aliasDb_->moveBeforeTopologicallyValid(producer->node(), consumer);

    if (!shouldFuse) {
      return at::nullopt;
    }

    if ((consumer->inputs().size() + consumer->outputs().size() +
         producer->node()->inputs().size() +
         producer->node()->outputs().size()) > subgraph_arg_limit_) {
      return at::nullopt;
    }

    auto group = consumer;
    if (consumer->kind() != kind_) {
      group = createSingletonFusionGroup(consumer);
    }

    if (producer->node()->kind() == kind_) {
      mergeFusionGroups(group, producer->node());
      return group;
    }
    AT_ASSERT(producer->node()->outputs().size() == 1);
    Node* merged = mergeNodeIntoGroup(group, producer->node());
    // remaining uses of this producer can occur because we allow
    // fusion in cases where uses remain after the consumer
    // if these exist, re-route them to the version of producer
    // created in FusionGroup
    if (producer->uses().size() != 0) {
      getSubgraph(group).registerOutput(merged->output());
      Value* new_producer = group->addOutput();
      new_producer->copyMetadata(producer);
      producer->replaceAllUsesWith(new_producer);
    }
    producer->node()->destroy();
    return group;
  }

  c10::optional<Node*> findFusedChunk(Node* group, Value* input) {
    AT_ASSERT(group->kind() == kind_);
    auto it = std::find(group->inputs().begin(), group->inputs().end(), input);
    if (it == group->inputs().end()) {
      return c10::nullopt;
    }
    size_t input_index = it - group->inputs().begin();
    auto& subgraph = getSubgraph(group);
    auto* subgraph_input = subgraph.inputs().at(input_index);
    // If subgraph_input is an input to prim::ConstantChunk, it will have 1 use
    auto* node = subgraph_input->uses().at(0).user;
    if (node->kind() == prim::ConstantChunk) {
      AT_ASSERT(subgraph_input->uses().size() == 1);
      return node;
    }
    return c10::nullopt;
  }

  void fuseChunkByReusingExistingFusedChunk(
      Node* group,
      Node* chunk,
      Node* existingFusedChunk) {
    if (chunk->outputs().size() != existingFusedChunk->outputs().size()) {
      return;
    }
    auto& subgraph = getSubgraph(group);
    for (size_t i = 0; i < chunk->outputs().size(); ++i) {
      // Find the input to the FusionGroup (group)
      auto* replacement_val = existingFusedChunk->outputs().at(i);
      auto* val = chunk->outputs().at(i);
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

  value_list sortReverseTopological(ArrayRef<Value*> inputs) {
    value_list result;
    for (auto i : inputs) {
      if (i->node()->owningBlock() == block_) {
        result.push_back(i);
      }
    }
    // Sort in reverse topological order
    std::sort(result.begin(), result.end(), [&](Value* a, Value* b) {
      return a->node()->isAfter(b->node());
    });
    return result;
  }

  at::ArrayRef<Value*> broadcast_tensors(value_list inputs) {
    AT_ASSERT(inputs.size() > 0);
    auto* g = inputs[0]->owningGraph();
    auto* input_list =
        g->insertNode(g->createList(TensorType::get(), inputs))->output();
    auto* output_list = g->insert(aten::broadcast_tensors, {input_list});
    auto* unpack_node = g->insertNode(
        g->create(prim::ListUnpack, {output_list}, inputs.size()));
    return unpack_node->outputs();
  }

  void insertExplicitBroadcast(Node* node) {
    WithInsertPoint insert_guard{node};
    auto tensors = tensorInputs(node);
    auto new_tensors = broadcast_tensors(tensors);

    // Replace tensors inputs with broadcasted values
    auto new_tensors_it = new_tensors.begin();
    for (size_t i = 0; i < node->inputs().size(); ++i) {
      if (node->inputs()[i]->type()->isSubtypeOf(TensorType::get())) {
        AT_ASSERT(new_tensors_it != new_tensors.end());
        node->replaceInput(i, *(new_tensors_it++));
      }
    }
  }

  Node* promoteChunkToBroadcastingChunk(Node* chunk) {
    AT_ASSERT(chunk->kind() == prim::ConstantChunk);

    size_t nchunks = chunk->i(attr::chunks);
    Node* bchunk =
        chunk->owningGraph()->create(prim::BroadcastingChunk, nchunks);
    bchunk->addInput(chunk->input());
    for (size_t i = 0; i < nchunks; ++i) {
      auto* old_output = chunk->outputs().at(i);
      auto* new_output = bchunk->outputs().at(i);
      new_output->copyMetadata(old_output);
      old_output->replaceAllUsesWith(new_output);
    }
    bchunk->copyAttributes(*chunk);
    bchunk->insertAfter(chunk);
    chunk->destroy();
    return bchunk;
  }

  // in places where op can be fused into a consumer but chunk is in the way
  // distribute chunk to op's operands:
  // replace a,b = chunk(op(x,y,z)) with:
  // x', y', z' = broadcast_tensors([x, y, z])
  // x0,x1 = chunk(x') (x0 has a's type, x1 has b's type)
  // y0,y1 = chunk(y') (y0 has a's type, y1 has b's type)
  // z0,z1 = chunk(z') (z0 has a's type, z1 has b's type)
  // a = op(x0,y0,z0) (a,b have their same size but are now contiguous)
  // b = op(x1,y1,x1)
  //
  // The graph fuser uses an intermediate prim::BroadcastingChunk node to
  // represent this behavior concisely. BroadcastingChunk(x, y, z) broadcasts
  // all of its inputs and then chunks each input, in order, the same way.
  // The above graph is equivalent to:
  // x0, x1, y0, y1, z0, z1 = BroadcastingChunk(x, y, z)
  // a = op(x0,y0,z0)
  // b = op(x1,y1,x1)
  //
  // NB: The explicit broadcast is important for correctness.
  // Let's say we have:
  // %z = aten::mul(%x, %y)
  // %z.1, %z.2 = aten::chunk(%z, ...)
  // ... = prim::CudaFusionGroup(%z.1, %z.2, ...)
  // It's possible that %x and %y do not have the same size as %z and
  // need to be expanded first so that they can be chunked like %z
  //
  // NB: Chunk motion only occurs with fusable consumers, which implies
  // that there is always some other operation, e.g., a+b, that happens
  // after the chunk, and will be put into the fusion group. This is
  // important, because distributing the chunk changes the contiguity
  // of a and b, and so the results would be invalid, except that we know
  // that simple_mappable operations will restore contiguity before
  // we exit the fusion group.
  //
  // NB: The intermediate BroadcastingChunk is important for moving chunks past
  // more than one operation: the graph fuser is not able to easily move
  // operations around broadcast_tensors + chunk nodes. Let f, g, h be fusable
  // ops
  //   x = f(v, w)
  //   z = g(x, y)
  //   a, b = chunk(z)
  //   c = h(a, b)
  // becomes (with the broadcast_tensors + chunk approach):
  //   x = f(v, w)
  //   x', y' = broadcast_tensors([x, y])
  //   ax, bx = chunk(x')
  //   ay, by = chunk(y')
  //   a = g(ax, ay)
  //   b = g(bx, by)
  //   c = h(a, b)
  // The broadcast_tensors node makes it harder to move f into the resulting
  // FusionGroup of g, g, and h. Keeping the broadcasting and chunk behavior
  // together results in:
  //   x = f(v, w)
  //   ax, bx, ay, by = BroadcastingChunk(x, y)
  //   a = g(ax, ay)
  //   b = g(bx, by)
  //   c = h(a, b)
  // making it easier to move f after the BroadcastingChunk:
  //   ay, by, av, bv, aw, bw = BroadcastingChunk(y, v, w)
  //   ax = f(av, aw)
  //   by = f(bv, bw)
  //   a = g(ax, ay)
  //   b = g(bx, by)
  //   c = h(a, b)

  bool tryToMoveChunk(Node* consumer, Value* producer) {
    // is the output from a chunk/bchunk node?
    auto* chunk = producer->node();
    if (chunk->kind() != prim::ConstantChunk &&
        chunk->kind() != prim::BroadcastingChunk)
      return false;

    // try to find a producer to move after the chunk/bchunk. The producer must
    // be fusable into the consumer.
    auto it = std::find_if(
        chunk->inputs().begin(),
        chunk->inputs().end(),
        [&](Value* producer_for_chunk) {
          return fuser::cuda::isFusableCudaFusionGroup(
                     consumer, producer_for_chunk->node()) &&
              allUsersAreThisConsumerOrCalcSizes(chunk, producer_for_chunk);
        });
    if (it == chunk->inputs().end()) {
      return false;
    }
    Value* producer_for_chunk = *it;
    size_t producer_index = it - chunk->inputs().begin();

    // all uses of the chunk must be in in this consumer
    for (auto s : chunk->outputs()) {
      for (auto u : s->uses()) {
        if (u.user != consumer)
          return false;
      }
    }
    // multiple return operators
    Node* producer_for_chunk_node = producer_for_chunk->node();
    AT_ASSERT(producer_for_chunk_node->outputs().size() == 1);

    // Convert chunk to bchunk, if it isn't one already. The bchunk represents a
    // broadcast and one or more chunk operations.
    auto* bchunk = chunk;
    if (chunk->kind() == prim::ConstantChunk) {
      bchunk = promoteChunkToBroadcastingChunk(chunk);
    }
    size_t nchunks = bchunk->i(attr::chunks);
    WithInsertPoint guard(bchunk->next());

    std::vector<Value*> producer_chunk_outputs;
    for (const auto i : c10::irange(nchunks)) {
      producer_chunk_outputs.push_back(
          bchunk->output(nchunks * producer_index + i));
    }

    // Add each of op's operands to the bchunk node.
    // chunked_inputs[input_nr][chunk_output_idx]
    //  = Node* for chunk_output_idx'th output of the chunk(inputs[input_nr])
    std::vector<std::vector<Value*>> chunked_inputs;

    for (auto input : producer_for_chunk_node->inputs()) {
      // XXX: we only work with pointwise ops in here, so we know it is valid to
      // push the concat only through tensor arguments (and all other args can
      // be safely ignored).
      if (!input->type()->isSubtypeOf(TensorType::get()))
        continue;

      // if 'input' is already an input to the bchunk, reuse it.
      auto bchunk_inputs = bchunk->inputs();
      auto it = std::find(bchunk_inputs.begin(), bchunk_inputs.end(), input);
      if (it != bchunk_inputs.end()) {
        chunked_inputs.emplace_back();
        auto input_index = std::distance(bchunk_inputs.begin(), it);
        for (size_t chunk = 0; chunk < nchunks; ++chunk) {
          chunked_inputs.back().push_back(
              bchunk->outputs().at(nchunks * input_index + chunk));
        }
        continue;
      }

      // NB: I decided not to use cloneFrom here, because if we make cloneFrom
      // copy selects one day, it is definitely not what you want here (selects
      // have different types).
      // TODO: Perhaps we should use cloneFrom now, as it seems unlikely
      // to copy select nodes now that we have refactored to have a Value
      // distinct from Node.
      bchunk->addInput(input);
      chunked_inputs.emplace_back(); // alas, to not be C++17
      for (auto chunk_sel : producer_chunk_outputs) {
        Value* input_chunk_sel = bchunk->addOutput();
        input_chunk_sel->setType(chunk_sel->type());
        chunked_inputs.back().push_back(input_chunk_sel);
      }
    }

    // apply the op to each chunk of the chunked operands,
    // and then rewrite the graph to use them!
    for (auto chunk_sel : producer_chunk_outputs) {
      auto original_inputs = producer_for_chunk_node->inputs();
      Node* chunked_op =
          block_->owningGraph()->create(producer_for_chunk_node->kind());
      chunked_op->copyAttributes(*producer_for_chunk_node);
      chunked_op->output()->setType(chunk_sel->type());
      auto chunked_inputs_it = chunked_inputs.begin();
      for (Value* original_input : original_inputs) {
        if (original_input->type()->isSubtypeOf(TensorType::get())) {
          AT_ASSERT(chunked_inputs_it != chunked_inputs.end());
          chunked_op->addInput(
              // NOLINTNEXTLINE(clang-analyzer-core.DivideZero)
              chunked_inputs_it->at(chunk_sel->offset() % nchunks));
          ++chunked_inputs_it;
        } else {
          chunked_op->addInput(original_input);
        }
      }
      bchunk->owningGraph()->insertNode(chunked_op);
      chunk_sel->replaceAllUsesWith(chunked_op->output());
    }

    bchunk->removeInput(producer_index);
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores,clang-diagnostic-unused-variable)
    for (const auto i : c10::irange(nchunks)) {
      bchunk->eraseOutput(nchunks * producer_index);
    }

    // The output of producer_for_chunk_node could have been used in some
    // aten::size operators, so we need to clean those up as well (we simply
    // broadcast all its tensor inputs).
    // We need to insert these early in the graph, i.e. immediately after
    // the producer_for_chunk_node as we will have the _size_if_not_same
    // that may be before the bchunk.
    WithInsertPoint guard2(producer_for_chunk_node);
    auto size_calc_uses = producer_for_chunk_node->output()->uses();
    if (!size_calc_uses.empty()) {
      auto tensor_inputs = filter(
          producer_for_chunk_node->inputs(),
          [](Value* v) { return v->type()->isSubtypeOf(TensorType::get()); });
      auto tensor_sizes = fmap(tensor_inputs, [](Value* v) {
        return v->owningGraph()->insert(aten::size, {v});
      });
      AT_ASSERT(!tensor_sizes.empty());
      Value* output_size = tensor_sizes.size() == 1
          ? tensor_sizes[0]
          : broadcastSizes(tensor_sizes);
      for (Use u : size_calc_uses) {
        u.user->output()->replaceAllUsesWith(output_size);
        u.user->destroy();
      }
    }
    producer_for_chunk_node->destroy();
    return true;
  }

  // returns where to continue scanning, and whether any fusion was made
  std::pair<graph_node_list::iterator, bool> scanNode(Node* consumer) {
    if (fuser::cuda::isFusableCudaFusionGroup(consumer)) {
      // handle inputs in reverse topological order as well...
      // otherwise in f(a,a+b) it will appear a is used twice if we consider
      // the f-a fusion before the f-(a+b) fusion first.
      auto inputs = sortReverseTopological(consumer->inputs());
      for (auto producer : inputs) {
        if (tryToMoveChunk(consumer, producer)) {
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

  void replaceIntermediateBroadcastingChunks() {
    for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();) {
      auto* node = *it;
      ++it; // We might delete node, so increment the iterator now.
      if (node->kind() != prim::BroadcastingChunk) {
        continue;
      }
      auto* bchunk = node;
      insertExplicitBroadcast(bchunk);

      auto* graph = block_->owningGraph();
      size_t nchunks = bchunk->i(attr::chunks);
      WithInsertPoint guard(bchunk->next());

      // Split the bchunk into bchunks.inputs().size() number of chunk nodes.
      for (size_t input_offset = 0; input_offset < bchunk->inputs().size();
           input_offset++) {
        auto* input = bchunk->inputs().at(input_offset);

        Node* new_chunk =
            graph->insertNode(graph->create(prim::ConstantChunk, input, 0));
        new_chunk->copyAttributes(*bchunk);
        for (size_t output_offset = 0; output_offset < nchunks;
             output_offset++) {
          auto new_output = new_chunk->addOutput();
          auto old_output =
              bchunk->outputs().at(input_offset * nchunks + output_offset);
          new_output->copyMetadata(old_output);
          old_output->replaceAllUsesWith(new_output);
        }
      }
      bchunk->destroy();
    }
  }

  bool usedOnlyInSize(Value* v) {
    const auto& uses = v->uses();
    return std::all_of(uses.begin(), uses.end(), [](const Use& u) {
      return u.user->matches("aten::size(Tensor self) -> int[]");
    });
  }

  // Builds up expressions that compute shapes of all intermediates (and
  // outputs) of the fusion group, based on the sizes of inputs. You should run
  // DCE to remove those that you end up not using.
  std::unordered_map<Value*, Value*> buildShapeExpressions(Node* fusion_group) {
    WithInsertPoint insert_guard{fusion_group->next()};
    std::unordered_map<Value*, Value*> shape_of;

    Graph* graph = fusion_group->owningGraph();
    auto subgraph = fusion_group->g(attr::Subgraph);

    auto inputs = fusion_group->inputs();
    auto sinputs = subgraph->inputs();
    AT_ASSERT(inputs.size() == sinputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i]->type()->isSubtypeOf(TensorType::get())) {
        shape_of[sinputs[i]] = graph->insert(aten::size, {inputs[i]});
      }
    }

    // When we have a guarantee that an output won't be removed, because it's
    // used in expressions that don't involve size checks, we can use its size
    // instead of computing a long chain of broadcasts, starting from the
    // beginning of the kernel.
    auto outputs = fusion_group->outputs();
    auto soutputs = subgraph->outputs();
    AT_ASSERT(outputs.size() == soutputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (usedOnlyInSize(outputs[i]))
        continue;
      shape_of[soutputs[i]] = graph->insert(aten::size, {outputs[i]});
    }

    for (Node* n : subgraph->nodes()) {
      // XXX: Use of shape_of.emplace is crucial to the output shape
      // optimization!
      if (n->kind() == prim::FusedConcat) {
        // This is a bit more involved, because we have to account for the case
        // when inputs have different shapes, but fortunately those tensors are
        // always outputs, and so we can simply avoid replacing their queries,
        // because it won't help us.
        continue;
      }
      if (n->kind() == prim::Constant) {
        continue;
      }
      if (n->kind() == prim::ConstantChunk) {
        Node* sizes_node = graph->insertNode(
            graph->create(prim::ChunkSizes, shape_of.at(n->input()), 2));
        sizes_node->i_(attr::dim, n->i(attr::dim));
        sizes_node->i_(attr::chunks, n->i(attr::chunks));
        Value* regular_size = sizes_node->outputs().at(0);
        Value* last_size = sizes_node->outputs().at(1);
        regular_size->setType(ListType::ofInts());
        last_size->setType(ListType::ofInts());
        auto outputs = n->outputs();
        for (Value* o : outputs.slice(0, outputs.size() - 1)) {
          shape_of.emplace(o, regular_size);
        }
        shape_of.emplace(outputs.at(outputs.size() - 1), last_size);
        continue;
      }
      // extended shape expression support to reduction operations
      // TODO: `aten::sum` is too flexible, we should restrict for a better
      // match
      if (n->kind() == aten::sum) {
        // TODO: expand support to wire non-constant inputs, this is currently
        // blocked by profiling executor not capable of profiling scalar inputs.
        TORCH_INTERNAL_ASSERT(
            n->input(1)->node()->kind() == prim::Constant &&
                n->input(2)->node()->kind() == prim::Constant,
            "only supports reduction axes and keepdim being constant");

        // hmmm, do I need to setInsertPoint...
        Node* in1_const =
            graph->createClone(n->input(1)->node(), [](Value*) -> Value* {
              throw std::runtime_error("unexpected input");
            });
        graph->insertNode(in1_const);
        Node* in2_const =
            graph->createClone(n->input(2)->node(), [](Value*) -> Value* {
              throw std::runtime_error("unexpected input");
            });
        graph->insertNode(in2_const);

        std::vector<Value*> inputs = {
            shape_of.at(n->input(0)), in1_const->output(), in2_const->output()};
        Node* size_node =
            graph->insertNode(graph->create(prim::ReductionSizes, inputs, 1));
        Value* size = size_node->output(0);
        size->setType(ListType::ofInts());
        shape_of.emplace(n->output(), size);
        continue;
      }
      auto tensor_inputs = filter(n->inputs(), [](Value* v) {
        return v->type()->isSubtypeOf(TensorType::get());
      });
      auto shapes =
          fmap(tensor_inputs, [&](Value* v) { return shape_of.at(v); });
      AT_ASSERT(!shapes.empty());
      shape_of.emplace(
          n->output(), shapes.size() == 1 ? shapes[0] : broadcastSizes(shapes));
    }
    return shape_of;
  }

  void removeOutputsUsedOnlyInSize(Node* fusion_group) {
    if (fusion_group->kind() != prim::CudaFusionGroup)
      return;
    auto subgraph = fusion_group->g(attr::Subgraph);

    // TODO: failure in buildShapeExpressions should not break fusion execution,
    // we can add a try/catch here to bailout from removeOutputsUsedOnlyInSize.
    auto shape_of = buildShapeExpressions(fusion_group);
    auto outputs = fusion_group->outputs().vec();
    auto soutputs = subgraph->outputs().vec();
    // XXX: Iterating in this order is not only good for performance reasons!
    // It is also crucial for correctness (i has to reflect the current true
    // index of outputs[i])!
    for (int64_t i = static_cast<int64_t>(outputs.size()) - 1; i >= 0; --i) {
      auto output = outputs[i];
      auto soutput = soutputs[i];
      if (usedOnlyInSize(output) && shape_of.count(soutput) > 0) {
        auto uses = output->uses();
        for (Use u : uses) {
          AT_ASSERT(u.user->matches("aten::size(Tensor self) -> int[]"));
          u.user->output()->replaceAllUsesWith(shape_of.at(soutput));
          u.user->destroy();
        }
        fusion_group->eraseOutput(i);
        subgraph->eraseOutput(i);
      }
    }
  }

  void refreshAliasDb() {
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
  }

  void optimizeFusedGraphs() {
    for (Node* node : block_->nodes()) {
      if (node->kind() != kind_) {
        continue;
      }
      auto subgraph = node->g(attr::Subgraph);
      EliminateDeadCode(subgraph);
      EliminateCommonSubexpression(subgraph);
      ConstantPooling(subgraph);
    }
  }

  void run() {
    // Run the pass until no changes are made.
    // This is necessary, because the algorithm can miss out on certain fusion
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
    // in our topological ordering. Then, %2 will be considered, and fused with
    // %1. If we do another iteration, the algorithm will consider the fusion of
    // these two groups and fix the situation.
    bool any_changed = true;
    while (any_changed) {
      any_changed = false;
      refreshAliasDb();
      for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        bool changed;
        std::tie(it, changed) = scanNode(*it);
        any_changed |= changed;
      }
    }
    refreshAliasDb();

    // fuseConcats();

    optimizeFusedGraphs();

    // The graph fuser can add intermediate prim::BroadcastingChunk nodes.
    // Replace them with broadcasts + chunks.
    replaceIntermediateBroadcastingChunks();

    // Fuse starting chunks into the group.
    // for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();) {
    //  it = scanNodeForChunks(*it);
    //}

    // Remove outputs that have been added only because we need their size
    for (Node* n : block_->nodes()) {
      removeOutputsUsedOnlyInSize(n);
    }

    for (Node* node : block_->nodes()) {
      for (Block* sub_block : node->blocks()) {
        CudaGraphFuser(sub_block, graph_).run();
      }
    }
  }
};

void compileFusionRecursive(Block* block) {
  FUSER_PERF_SCOPE("compileFusionRecursive");

  for (auto node : block->nodes()) {
    if (node->kind() == prim::CudaFusionGroup) {
      fuser::cuda::compileFusionGroup(node);
    }
    for (auto sub_block : node->blocks()) {
      compileFusionRecursive(sub_block);
    }
  }
}

void PeepholeOptimizeShapeExpressions(Block* block) {
  FUSER_PERF_SCOPE("PeepholeOptimizeShapeExpressions");

  auto nodes = block->nodes();
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    Node* node = *it;
    for (Block* subblock : node->blocks()) {
      PeepholeOptimizeShapeExpressions(subblock);
    }
    if (node->kind() == prim::BroadcastSizes) {
      // Remove no-op broadcasts.
      if (node->inputs().size() == 1) {
        node->output()->replaceAllUsesWith(node->input());
        it.destroyCurrent();
        continue;
      }
      // Deduplicate inputs, but use their unique() values to ensure
      // this process only depends on the graph.
      std::map<size_t, Value*> unique_to_value;
      for (Value* input : node->inputs()) {
        unique_to_value.emplace(input->unique(), input);
      }
      if (unique_to_value.size() != node->inputs().size()) {
        std::vector<Value*> inputs;
        inputs.reserve(unique_to_value.size());
        for (auto& entry : unique_to_value) {
          inputs.push_back(entry.second);
        }
        if (inputs.size() == 1) {
          node->output()->replaceAllUsesWith(inputs[0]);
        } else {
          WithInsertPoint insert_guard{node};
          node->output()->replaceAllUsesWith(broadcastSizes(inputs));
        }
        it.destroyCurrent();
        --it; // Revisit the node with deduplicated inputs
        continue;
      }
      // Remove compose simple chains of broadcasts into a single node.
      const auto& uses = node->output()->uses();
      if (uses.size() == 1 && uses[0].user->kind() == prim::BroadcastSizes) {
        Node* user = uses[0].user;
        user->removeInput(uses[0].offset);
        // NB: we don't care about deduplication in here, as we will visit user
        // later.
        for (Value* i : node->inputs()) {
          user->addInput(i);
        }
        it.destroyCurrent();
      }
    }
  }
}

//! [ Note -- CudaFusionGuard implementation ]
//!
//! shamelessly copying code from NNC (tensorexpr_fuser)  with very little
//! modification, original code at:
//! `../../passes/tensorexpr_fuser.cpp:guardFusionGroup`
//!
//! Add prim::CudaFusionGuard node to ensure that accepted profiling information
//! is not violated at runtime.
//!
//! We replace a single
//!
//!   outputs = prim::CudaFusionGroup[cache_id](inputs)
//!
//! with the following pattern:
//!
//!   %1 : bool = prim::CudaFusionGuard[types=[...]](inputs)
//!   outputs = prim::If(%1)
//!     block0():
//!       outputs = prim::CudaFusionGroup[cache_id](inputs)
//!       -> (outputs)
//!     block1():
//!       %2 : Function = prim::Constant[name="fallback_function", fallback=1]()
//!       otuputs = prim::CallFunction(%2, inputs)
//!       -> (outputs)
//!
//! `prim::CudaFusionGuard` stores all profiled data type in attribute
//! `attr::types`.
//! At runtime, we check input tensors against our profiled data type and return
//! an output holds the result of the check (bool).
//! See [ Note -- type guard logic in CudaFusionGuard ]
//!
//! This ensures that `prim::CudaFusionGroup` only execute compatible inputs.
//! In case of check failure, execution goes through false block, which
//! recursively goes along another profiling / optimization iteration. (could be
//! tuned by `bailout_depth`)
//!
//! TODO: we also need to assert/check reduction axes and replace it with
//! constants in `CudaFusionGroup`
void guardFusionGroup(Node* fusion) {
  // Fixup types of the subgraph inputs
  std::vector<TypePtr> guard_types;
  std::vector<Value*> inputs_to_check;
  for (Value* input : fusion->inputs()) {
    // We only check inputs of the fusion group and expect NNC to infer
    // intermediates and outputs shapes
    if (!input->type()->cast<TensorType>()) {
      continue;
    }

    // note: modified from original implementation, we are guarding fusion
    //       outputs
    if (input->node()->kind() == prim::Constant) {
      continue;
    }
    inputs_to_check.push_back(input);
    guard_types.push_back(input->type());
  }
  if (!inputs_to_check.size()) {
    return;
  }

  Node* typecheck_node = fusion->owningGraph()
                             ->create(prim::CudaFusionGuard, inputs_to_check, 1)
                             ->insertBefore(fusion);
  // fix output to BoolType
  typecheck_node->output()->setType(BoolType::get());
  Value* typecheck_result = typecheck_node->output();
  typecheck_node->tys_(attr::types, guard_types);

  std::unordered_map<Value*, Value*> typechecked_inputs;

  // Insert if block
  auto versioning_if =
      fusion->owningGraph()
          ->create(prim::If, {typecheck_result}, fusion->outputs().size())
          ->insertAfter(typecheck_node);
  for (size_t idx = 0; idx < fusion->outputs().size(); ++idx) {
    versioning_if->output(idx)->setType(fusion->output(idx)->type());
    fusion->output(idx)->replaceAllUsesWith(versioning_if->output(idx));
  }
  auto true_block = versioning_if->addBlock();
  auto false_block = versioning_if->addBlock();

  // Fill in the false block. It should contain the unoptimized
  // copy of the fused subgraph.
  auto& subgraph = *fusion->g(attr::Subgraph);
  WithInsertPoint guard(false_block->return_node());
  const auto subgraph_outputs =
      insertGraph(*fusion->owningGraph(), subgraph, fusion->inputs());
  for (Value* output : subgraph_outputs) {
    false_block->registerOutput(output);
  }

  // types get copied to the fallback graph, so remove specializations before
  // replacing
  // TODO: this is not exposed here, I need to remove that before inserting the
  //       graph
  // removeTensorTypeSpecializations(false_block);
  replaceBlockWithFallbackGraph(false_block, fusion->inputs());

  // Fill in the true block. It has all inputs type-checked and its
  // body should be the fusion group node.
  fusion->moveBefore(true_block->return_node());
  for (Value* output : fusion->outputs()) {
    true_block->registerOutput(output);
  }
}

void guardFusionGroups(Block* block) {
  std::vector<Node*> fusions;
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      guardFusionGroups(b);
    }
    if (n->kind() == prim::CudaFusionGroup) {
      fusions.push_back(n);
    }
  }
  for (Node* fusion : fusions) {
    guardFusionGroup(fusion);
  }
}

} // anonymous namespace

void CudaFuseGraph(std::shared_ptr<Graph>& graph) {
  FUSER_PERF_SCOPE("CudaFuseGraph");
  // TODO: we need to properly restore shape information after fusion.
  // shamelessly use tool from NNC.
  RemoveProfileNodesAndSpecializeTypes(graph);

  CudaGraphFuser(graph->block(), graph).run();
  guardFusionGroups(graph->block());
  // After FuseGraph some common subexpressions may come back
  EliminateCommonSubexpression(graph);
  // We might have emitted a fair amount of useless shape propagating code, so
  // remove it
  EliminateDeadCode(graph);
  // Improve the quality of shape propagation code that was left
  PeepholeOptimizeShapeExpressions(graph->block());

  // TODO: we need to properly restore shape information after fusion.
  // shamelessly use tool from NNC.
  RemoveTensorTypeSpecializations(graph);

  // Compile CudaFusionGroup
  compileFusionRecursive(graph->block());
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
