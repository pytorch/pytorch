#include <torch/csrc/jit/passes/cuda_graph_fuser.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/codegen/cuda/partition.h>
#include <torch/csrc/jit/codegen/cuda/transform_view.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/restore_mutation.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

#include <queue>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

constexpr size_t NVRTC_KERNEL_ARG_LIMIT = 128;

namespace {

bool usedOnlyInDtype(Value* v) {
  const auto& uses = v->uses();
  if (uses.empty()) {
    return false;
  }
  return std::all_of(uses.begin(), uses.end(), [](const Use& u) {
    return u.user->matches("prim::dtype(Tensor a) -> int");
  });
}

Value* broadcastSizes(at::ArrayRef<Value*> sizes) {
  AT_ASSERT(!sizes.empty());
  Graph* graph = sizes[0]->owningGraph();
  Node* insertion_point = sizes[0]->node()->next();
  for (size_t i = 1; i < sizes.size(); i++) {
    if (insertion_point->isBefore(sizes[i]->node()->next())) {
      insertion_point = sizes[i]->node()->next();
    }
  }
  WithInsertPoint guard(insertion_point);
  Node* broadcast_n =
      graph->insertNode(graph->create(prim::BroadcastSizes, sizes));
  broadcast_n->output()->setType(ListType::ofInts());
  return broadcast_n->output();
}

Value* createConditionalConstant(Node* profile_ivalue) {
  TORCH_INTERNAL_ASSERT(profile_ivalue->kind() == prim::profile_ivalue);

  auto graph = profile_ivalue->owningGraph();

  IValue val; // default to None
  if (profile_ivalue->hasAttribute(Symbol::attr("profiled_int_list"))) {
    // int[]
    val = IValue(profile_ivalue->is(Symbol::attr("profiled_int_list")));
  } else if (profile_ivalue->hasAttribute(Symbol::attr("profiled_bool_list"))) {
    // bool[]
    auto int_list = profile_ivalue->is(Symbol::attr("profiled_bool_list"));
    std::vector<bool> bool_list(int_list.begin(), int_list.end());
    val = IValue(bool_list);
  } else if (profile_ivalue->hasAttribute(
                 Symbol::attr("profiled_reduction_size"))) {
    // int[]
    val = IValue(profile_ivalue->is(Symbol::attr("profiled_reduction_size")));
  } else if (profile_ivalue->hasAttribute(Symbol::attr("profiled_view_size"))) {
    // int[]
    val = IValue(profile_ivalue->is(Symbol::attr("profiled_view_size")));
  } else if (profile_ivalue->hasAttribute(Symbol::attr("profiled_bool"))) {
    // bool
    val = IValue(
        static_cast<bool>(profile_ivalue->i(Symbol::attr("profiled_bool"))));
  } else if (profile_ivalue->hasAttribute(Symbol::attr("profiled_int"))) {
    // int
    val = IValue(
        static_cast<int>(profile_ivalue->i(Symbol::attr("profiled_int"))));
  } else if (profile_ivalue->hasAttribute(Symbol::attr("profiled_str"))) {
    // str
    val = IValue(static_cast<std::string>(
        profile_ivalue->s(Symbol::attr("profiled_str"))));
  } else if (profile_ivalue->hasAttribute(Symbol::attr("profiled_ival"))) {
    // ival
    val = IValue(profile_ivalue->ival(Symbol::attr("profiled_ival")));
  } else {
    GRAPH_DEBUG("no profile info in profile_ivalue node: ", *profile_ivalue);
    TORCH_WARN_ONCE(
        __func__,
        " profile_node ",
        *profile_ivalue,
        " does not have profile information");
    return nullptr;
  }

  return graph->insertConstant(val);
}

struct CudaGraphFuser {
  using FusionCallback = std::function<bool(Node*)>;

  Block* block_;
  std::unique_ptr<AliasDb> aliasDb_;
  std::shared_ptr<Graph> graph_;
  Symbol kind_ = prim::CudaFusionGroup;
  std::unordered_map<Value*, Value*> fusion_value_to_runtime_shape_;

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
      return v->type()->isSubtypeOf(*TensorType::get());
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
    for (const auto i : c10::irange(inner_inputs.size())) {
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
      for (const auto i : c10::irange(inner_outputs.size())) {
        inner_to_outer[inner_outputs[i]] = outer_outputs[i];
      }
    }

    // Replace uses of producer_group outputs and destroy the producer
    auto subgraph_outputs = producer_subgraph->outputs();
    for (const auto i : c10::irange(subgraph_outputs.size())) {
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
      for (const auto i : c10::irange(outputs.size())) {
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
    for (auto input : group->inputs()) {
      inputs_map[input] = subgraph.inputs()[i++];
      if (input->type()->isSubtypeOf(*TensorType::get()))
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
        if (input->type()->isSubtypeOf(*TensorType::get())) {
          auto in_group = subgraph.insertInput(tensor_insert_idx);
          in_group->setType(input->type());
          inputs_map[input] = in_group;
          group->insertInput(tensor_insert_idx, input);
          tensor_insert_idx++;
        } else if (
            // TODO: extend the supporting inputs here.
            (input->type()->isSubtypeOf(*FloatType::get()) &&
             input->node()->kind() != prim::Constant)) {
          auto in_group = subgraph.addInput();
          in_group->setType(input->type());
          inputs_map[input] = in_group;
          group->addInput(input);
        } else if (input->node()->kind() == prim::Constant) {
          // inline the constants directly in the body of the fused group.
          Node* in_const =
              subgraph.createClone(input->node(), [&](Value* v) -> Value* {
                if (v->node()->kind() != prim::profile_ivalue) {
                  throw std::runtime_error(
                      std::string(
                          "merging constant with unexpected input from node") +
                      v->node()->kind().toDisplayString());
                }
                group->addInput(v->node()->output());

                // we are doing this just to keep alias_analysis silent with
                // their checks
                auto in_group = subgraph.addInput();
                in_group->setType(v->type());
                return in_group;
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
    for (const auto i : c10::irange(n->outputs().size())) {
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
    for (const auto i : c10::irange(n->outputs().size())) {
      getSubgraph(group).registerOutput(mergedNode->output(i));
      auto sel = group->addOutput();
      sel->copyMetadata(n->output(i));
    }
    n->replaceAllUsesWith(group);
    n->destroy();
    return group;
  }

  at::optional<Node*> tryFuse(Node* consumer, Node* producer) {
    // this handles cases where producer can be moved _into_ the fusion group of
    // consumer.
    // TODO: extend to fusion of consumer into _producer's_ fusion blob
    // if the consumer allInputsAreThisProducer(consumer,producer)
    // we can move the consumer up into the producer.
    // but this requires better handling of merging fusion groups so it is not
    // done now
    bool shouldFuse =
        fuser::cuda::isFusibleCudaFusionGroup(consumer, producer) &&
        // Rearrange nodes such that all uses of producer's outputs are after
        // consumer. Fusion will rewrite those later uses to use the version of
        // producer generated by the fused blob. In this case, producer becomes
        // an output of the fusion group.
        aliasDb_->moveBeforeTopologicallyValid(producer, consumer);

    if (!shouldFuse) {
      return at::nullopt;
    }

    if ((consumer->inputs().size() + consumer->outputs().size() +
         producer->inputs().size() + producer->outputs().size()) >
        subgraph_arg_limit_) {
      return at::nullopt;
    }

    auto group = consumer;
    if (consumer->kind() != kind_) {
      group = createSingletonFusionGroup(consumer);
    }

    if (producer->kind() == kind_) {
      mergeFusionGroups(group, producer);
      return group;
    }
    Node* merged = mergeNodeIntoGroup(group, producer);
    // remaining uses of this producer can occur because we allow
    // fusion in cases where uses remain after the consumer
    // if these exist, re-route them to the version of producer
    // created in FusionGroup

    // We need to apply this to all outputs from producer->node();
    auto producer_outputs = producer->outputs();
    for (const auto i : c10::irange(producer_outputs.size())) {
      if (producer_outputs[i]->uses().size() != 0) {
        getSubgraph(group).registerOutput(merged->outputs()[i]);
        Value* new_producer = group->addOutput();
        new_producer->copyMetadata(producer_outputs[i]);
        producer_outputs[i]->replaceAllUsesWith(new_producer);
      }
    }
    producer->destroy();
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
    for (const auto i : c10::irange(chunk->outputs().size())) {
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
    for (const auto i : c10::irange(node->inputs().size())) {
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
    for (const auto i : c10::irange(nchunks)) {
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
          return fuser::cuda::isFusibleCudaFusionGroup(
                     consumer, producer_for_chunk->node()) &&
              isElementWiseNode(consumer) &&
              allUsersAreThisConsumerOrCalcSizes(chunk, producer_for_chunk);
        });
    if (it == chunk->inputs().end()) {
      return false;
    }
    Value* producer_for_chunk = *it;
    size_t producer_index = it - chunk->inputs().begin();

    // all uses of the chunk must be in this consumer
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
    TORCH_INTERNAL_ASSERT(nchunks > 0, "number of chunks cannot be zero");
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

    // We have asserted single output earlier
    auto producer_output_sizes =
        producer_for_chunk_node->output()->type()->cast<TensorType>()->sizes();

    for (auto input : producer_for_chunk_node->inputs()) {
      // XXX: we only work with pointwise ops in here, so we know it is valid to
      // push the concat only through tensor arguments (and all other args can
      // be safely ignored).
      if (!input->type()->isSubtypeOf(*TensorType::get()))
        continue;

      // if 'input' is already an input to the bchunk, reuse it.
      auto bchunk_inputs = bchunk->inputs();
      auto it = std::find(bchunk_inputs.begin(), bchunk_inputs.end(), input);
      if (it != bchunk_inputs.end()) {
        chunked_inputs.emplace_back();
        auto input_index = std::distance(bchunk_inputs.begin(), it);
        for (const auto chunk : c10::irange(nchunks)) {
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

      // properly compute strides for BroadcastingChunk
      //
      // We copy stride of each dimension from input to output for
      // BroadcastingChunk. A note is that Chunk should not alter strides,
      // However, broadcasted dimension should have a stride 0. We could have
      // broadcasting happening on existing dimensions in input (case1), as well
      // as extended dimension that does not exist in input (case2).
      // e.g.
      // If we look at an input tensor t0 with shape [3, 1] broadcasted to
      // output tensor t1 with shape [4, 1, 3, 3],
      // We set stride to zero in case of broadcast, which could happen in:
      //   case1: t1.dim[3] (broadcasted as in the description above)
      //   case2: t1.dim[0] (broadcasted implicitly)
      std::vector<int64_t> strides;
      auto input_type = input->type()->cast<TensorType>();
      auto input_sizes = input_type->sizes();
      auto input_strides = input_type->strides();
      if (producer_output_sizes.isComplete() && input_sizes.isComplete() &&
          input_strides.isComplete()) {
        auto input_c_sizes = input_sizes.concrete_sizes().value();
        auto input_c_strides = input_strides.concrete_sizes().value();
        auto output_c_sizes = producer_output_sizes.concrete_sizes().value();
        int output_index = int(output_c_sizes.size()) - 1;
        strides.resize(output_index + 1);
        AT_ASSERT(output_index >= int(input_c_sizes.size()) - 1);
        for (int input_index = int(input_c_sizes.size()) - 1; input_index >= 0;
             input_index--, output_index--) {
          // in braodcast case 1, we set stride to 0;
          // otherwise, stride remain the same.
          if (input_c_sizes[input_index] == 1 &&
              output_c_sizes[output_index] != 1) {
            strides[output_index] = 0;
          } else {
            strides[output_index] = input_c_strides[input_index];
          }
        }

        // continue on expanding dimensions to set stride to 0 for case2
        while (output_index >= 0) {
          strides[output_index] =
              output_c_sizes[output_index] == 1 ? strides[output_index + 1] : 0;
          output_index--;
        }
      }

      for (auto chunk_sel : producer_chunk_outputs) {
        Value* input_chunk_sel = bchunk->addOutput();
        auto chunk_sel_type = chunk_sel->type()->cast<TensorType>();
        if (strides.empty() || !chunk_sel_type->sizes().isComplete()) {
          input_chunk_sel->setType(chunk_sel_type);
        } else {
          input_chunk_sel->setType(chunk_sel_type->withSizesStrides(
              chunk_sel_type->sizes().concrete_sizes().value(), strides));
        }
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
        if (original_input->type()->isSubtypeOf(*TensorType::get())) {
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
      (void)i; // Suppress unused variable warning
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
          [](Value* v) { return v->type()->isSubtypeOf(*TensorType::get()); });
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
    if (fuser::cuda::isFusibleCudaFusionGroup(consumer)) {
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
        if (getSingletonFusion() && consumer->kind() != kind_) {
          consumer = createSingletonFusionGroup(consumer);
        }
        auto fusion_group = tryFuse(consumer, producer->node());
        if (fusion_group) {
          // after fusion, consumer moves into a FusionGroup, so inputs is no
          // longer valid so we rescan the new FusionGroup for more fusions...
          return std::make_pair(fusion_group.value()->reverseIterator(), true);
        }

        // horizontal fusion only applies on non-scalar tensor inputs
        if (getHorizontalFusion() &&
            producer->type()->isSubtypeOf(*TensorType::get()) &&
            !is_cpu_scalar(*producer->type()->cast<TensorType>())) {
          // fusing nodes sharing inputs, this could save memory bandwidth by
          // reducing number of tensor read.
          for (const auto& u : producer->uses()) {
            // only merge nodes before consumer, since any sibling after
            // consumer has already considered merging this consumer to them
            // already.
            if (u.user->isBefore(consumer)) {
              auto fusion_group = tryFuse(consumer, u.user);
              if (fusion_group) {
                return std::make_pair(
                    fusion_group.value()->reverseIterator(), true);
              }
            }
          }
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
      for (const auto input_offset : c10::irange(bchunk->inputs().size())) {
        auto* input = bchunk->inputs().at(input_offset);

        Node* new_chunk =
            graph->insertNode(graph->create(prim::ConstantChunk, input, 0));
        new_chunk->copyAttributes(*bchunk);
        for (const auto output_offset : c10::irange(nchunks)) {
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

  bool usedInDtype(Value* v) {
    const auto& uses = v->uses();
    return std::any_of(uses.begin(), uses.end(), [](const Use& u) {
      return u.user->matches("prim::dtype(Tensor a) -> int");
    });
  }

  bool usedOnlyInDtypeAndSize(Value* v) {
    const auto& uses = v->uses();
    return std::all_of(uses.begin(), uses.end(), [](const Use& u) {
      return u.user->matches("prim::dtype(Tensor a) -> int") ||
          u.user->matches("aten::size(Tensor self) -> int[]");
    });
  }

  // Builds up expressions that compute shapes of all intermediates (and
  // outputs) of the fusion group, based on the sizes of inputs. You should run
  // DCE to remove those that you end up not using.
  // TODO: Add shape support for view, reshape, unsqueeze, and squeeze
  std::unordered_map<Value*, Value*> buildShapeExpressions(Node* fusion_group) {
    WithInsertPoint insert_guard{fusion_group->next()};
    std::unordered_map<Value*, Value*> shape_of;

    Graph* graph = fusion_group->owningGraph();
    auto subgraph = fusion_group->g(attr::Subgraph);

    auto inputs = fusion_group->inputs();
    auto sinputs = subgraph->inputs();
    AT_ASSERT(inputs.size() == sinputs.size());
    for (const auto i : c10::irange(inputs.size())) {
      if (inputs[i]->type()->isSubtypeOf(*TensorType::get())) {
        auto sinput_value = graph->insert(aten::size, {inputs[i]});
        shape_of[sinputs[i]] = sinput_value;
        sinput_value->node()->moveBefore(fusion_group);
      }
    }

    // When we have a guarantee that an output won't be removed, because it's
    // used in expressions that don't involve size checks, we can use its size
    // instead of computing a long chain of broadcasts, starting from the
    // beginning of the kernel.
    auto outputs = fusion_group->outputs();
    auto soutputs = subgraph->outputs();
    AT_ASSERT(outputs.size() == soutputs.size());
    for (const auto i : c10::irange(outputs.size())) {
      if (usedOnlyInDtypeAndSize(outputs[i]))
        continue;
      if (soutputs[i]->type()->isSubtypeOf(TensorType::get())) {
        shape_of[soutputs[i]] = graph->insert(aten::size, {outputs[i]});
      }
    }

    // Place all the shape expressions for intermediates in fusion
    // before the CudaFusionGroup
    graph->setInsertPoint(fusion_group);

    // hmmm, do I need to setInsertPoint...
    const auto map_inputs = [&](Value* v) -> Value* {
      // if constant ever has an input, it has to come from
      // profile_ivalue dependency
      if (v->node()->kind() == prim::Param &&
          fusion_group->input(v->offset())->node()->kind() ==
              prim::profile_ivalue) {
        // we need to map it along profile_ivalue dependency
        return fusion_group->input(v->offset());
      } else {
        throw std::runtime_error(
            std::string("unexpected input from node") +
            v->node()->kind().toDisplayString());
      }
    };

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
        TORCH_INTERNAL_ASSERT(
            shape_of.count(n->input()) > 0,
            "buildShapeExpressions failed at accessing input shapes");
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
      // TODO: Add python tests where we check for existing ops and their
      // shape expression logic.
      static std::unordered_set<Symbol> reduction_ops(
          {aten::sum, aten::mean, aten::var, aten::std});
      if (reduction_ops.find(n->kind()) != reduction_ops.end()) {
        // TODO: expand support to wire non-constant inputs, this is currently
        // blocked by profiling executor not capable of profiling scalar inputs.
        TORCH_INTERNAL_ASSERT(
            n->input(1)->node()->kind() == prim::Constant &&
                n->input(2)->node()->kind() == prim::Constant,
            "only supports reduction axes and keepdim being constant");

        Node* in1_const = graph->createClone(n->input(1)->node(), map_inputs);
        graph->insertNode(in1_const);
        Node* in2_const = graph->createClone(n->input(2)->node(), map_inputs);
        graph->insertNode(in2_const);

        TORCH_INTERNAL_ASSERT(
            shape_of.count(n->input(0)) > 0,
            "buildShapeExpressions failed at accessing input shapes");
        std::vector<Value*> inputs = {
            shape_of.at(n->input(0)), in1_const->output(), in2_const->output()};
        Node* size_node =
            graph->insertNode(graph->create(prim::ReductionSizes, inputs, 1));
        Value* size = size_node->output(0);
        size->setType(ListType::ofInts());
        shape_of.emplace(n->output(), size);
        continue;
      }
      // TODO: output(1) & output(2) should also be marked
      if (n->kind() == aten::native_layer_norm) {
        TORCH_INTERNAL_ASSERT(
            shape_of.count(n->input(0)) > 0,
            "buildShapeExpressions failed at accessing input shapes");
        shape_of.emplace(n->output(0), shape_of.at(n->input(0)));
        continue;
      }
      // TODO: output(1) & output(2) should also be marked
      if (n->kind() == aten::native_layer_norm_backward) {
        TORCH_INTERNAL_ASSERT(
            shape_of.count(n->input(0)) > 0,
            "buildShapeExpressions failed at accessing input shapes");
        shape_of.emplace(n->output(0), shape_of.at(n->input(0)));
        if (shape_of.count(n->input(5)) > 0) {
          shape_of.emplace(n->output(1), shape_of.at(n->input(5)));
        }
        if (shape_of.count(n->input(6)) > 0) {
          shape_of.emplace(n->output(2), shape_of.at(n->input(6)));
        }
        continue;
      }
      // TODO: output(1) & output(2) should also be marked
      if (n->kind() == aten::native_batch_norm ||
          n->kind() == aten::_batch_norm_impl_index) {
        TORCH_INTERNAL_ASSERT(
            shape_of.count(n->input(0)) > 0,
            "buildShapeExpressions failed at accessing input shapes");
        shape_of.emplace(n->output(0), shape_of.at(n->input(0)));
        continue;
      }
      // TODO: output(1) & output(2) should also be marked
      if (n->kind() == aten::native_batch_norm_backward) {
        TORCH_INTERNAL_ASSERT(
            shape_of.count(n->input(0)) > 0,
            "buildShapeExpressions failed at accessing input shapes");
        shape_of.emplace(n->output(0), shape_of.at(n->input(0)));
        if (shape_of.count(n->input(2)) > 0) {
          shape_of.emplace(n->output(1), shape_of.at(n->input(2)));
          // use shape of weight here for grad_bias
          shape_of.emplace(n->output(2), shape_of.at(n->input(2)));
        }
        continue;
      }
      if (n->kind() == aten::_batch_norm_impl_index_backward) {
        TORCH_INTERNAL_ASSERT(
            shape_of.count(n->input(1)) > 0,
            "buildShapeExpressions failed at accessing input shapes");
        shape_of.emplace(n->output(0), shape_of.at(n->input(1)));
        if (shape_of.count(n->input(3)) > 0) {
          shape_of.emplace(n->output(1), shape_of.at(n->input(3)));
          // use shape of weight here for grad_bias
          shape_of.emplace(n->output(2), shape_of.at(n->input(3)));
        }
        continue;
      }
      if (n->kind() == aten::native_dropout) {
        TORCH_INTERNAL_ASSERT(
            shape_of.count(n->input(0)) > 0,
            "buildShapeExpressions failed at accessing input shapes");
        shape_of.emplace(n->output(0), shape_of.at(n->input(0)));
        shape_of.emplace(n->output(1), shape_of.at(n->input(0)));
        continue;
      }
      if (n->kind() == prim::unsqueeze_copy) {
        TORCH_INTERNAL_ASSERT(
            shape_of.count(n->input(0)) > 0,
            "buildShapeExpressions failed at accessing input shapes");
        TORCH_INTERNAL_ASSERT(
            n->input(1)->node()->kind() == prim::Constant,
            "only supports unsqueeze axes being constant");
        Node* dim_const = graph->createClone(n->input(1)->node(), map_inputs);
        graph->insertNode(dim_const);
        std::vector<Value*> inputs = {
            shape_of.at(n->input(0)), dim_const->output()};
        Node* size_node = graph->insertNode(graph->create(
            Symbol::fromQualString("prim::infer_unsqueeze_size"), inputs, 1));
        Value* size = size_node->output(0);
        size->setType(ListType::ofInts());
        shape_of.emplace(n->output(), size);
        continue;
      }
      if (n->kind() == prim::squeeze_copy) {
        TORCH_INTERNAL_ASSERT(
            shape_of.count(n->input(0)) > 0,
            "buildShapeExpressions failed at accessing input shapes");
        TORCH_INTERNAL_ASSERT(
            n->inputs().size() == 2 || n->inputs().size() == 1,
            "prim::squeeze_copy expects one or two inputs");
        std::vector<Value*> inputs = {shape_of.at(n->input(0))};

        if (n->inputs().size() == 2) {
          TORCH_INTERNAL_ASSERT(
              n->input(1)->node()->kind() == prim::Constant,
              "only supports squeeze axes being constant");
          Node* dim_const = graph->createClone(n->input(1)->node(), map_inputs);
          graph->insertNode(dim_const);
          inputs.push_back(dim_const->output());
        }
        Node* size_node = graph->insertNode(graph->create(
            Symbol::fromQualString("prim::infer_squeeze_size"), inputs, 1));
        Value* size = size_node->output(0);
        size->setType(ListType::ofInts());
        shape_of.emplace(n->output(), size);
        continue;
      }

      auto tensor_inputs = filter(n->inputs(), [](Value* v) {
        return v->type()->isSubtypeOf(*TensorType::get());
      });
      auto shapes = fmap(tensor_inputs, [&](Value* v) {
        TORCH_INTERNAL_ASSERT(
            shape_of.count(v) > 0,
            "buildShapeExpressions failed at accessing input shapes");
        return shape_of.at(v);
      });
      AT_ASSERT(!shapes.empty());
      shape_of.emplace(
          n->output(0),
          shapes.size() == 1 ? shapes[0] : broadcastSizes(shapes));
    }
    return shape_of;
  }

  void removeOutputsUsedOnlyInSize(Node* fusion_group) {
    if (fusion_group->kind() != prim::CudaFusionGroup)
      return;
    auto subgraph = fusion_group->g(attr::Subgraph);

    // TODO: failure in buildShapeExpressions should not break fusion execution,
    // we can add a try/catch here to bailout from removeOutputsUsedOnlyInSize.
    GRAPH_DEBUG("before build shape expression: ", *graph_);
    auto shape_map = buildShapeExpressions(fusion_group);
    fusion_value_to_runtime_shape_.insert(shape_map.begin(), shape_map.end());
    GRAPH_DEBUG("after build shape expression: ", *graph_);

    auto outputs = fusion_group->outputs().vec();
    auto soutputs = subgraph->outputs().vec();
    // XXX: Iterating in this order is not only good for performance reasons!
    // It is also crucial for correctness (i has to reflect the current true
    // index of outputs[i])!
    for (int64_t i = static_cast<int64_t>(outputs.size()) - 1; i >= 0; --i) {
      auto output = outputs[i];
      auto soutput = soutputs[i];
      if (usedOnlyInDtypeAndSize(output) && shape_map.count(soutput) > 0) {
        bool has_dtype = usedInDtype(output);
        auto uses = output->uses();
        for (Use u : uses) {
          if (u.user->matches("aten::size(Tensor self) -> int[]")) {
            u.user->output()->replaceAllUsesWith(shape_map.at(soutput));
            u.user->destroy();
          } else if (u.user->matches("prim::dtype(Tensor a) -> int")) {
            continue;
          } else {
            AT_ASSERT(
                false,
                "unrecognized consumer should not trigger removeOutputsUsedOnlyInSize");
          }
        }
        // We only wipe the output when there's no more dtype consumer.
        // This is to be removed by `removeOutputUsedOnlyInDtype`
        if (!has_dtype) {
          fusion_group->eraseOutput(i);
          subgraph->eraseOutput(i);
        }
      }
    }
    GRAPH_DEBUG("after build shape expression and re-wiring: ", *graph_);
  }

  void refreshAliasDb() {
    aliasDb_ = torch::make_unique<AliasDb>(graph_);
  }

  void removeNoopBinaryOps(Block* block) {
    for (Node* node : block->nodes()) {
      for (Block* b : node->blocks()) {
        removeNoopBinaryOps(b);
      }

      if (node->matches(
              "aten::add(Tensor self, Scalar other, Scalar alpha) -> Tensor",
              /*const_inputs=*/{attr::alpha, attr::other}) ||
          node->matches(
              "aten::sub(Tensor self, Scalar other, Scalar alpha) -> Tensor",
              /*const_inputs=*/{attr::alpha, attr::other})) {
        // x + 0 == x - 0 == x
        // if either scalar input is a float, than removing this operator could
        // remove type promotion and affect semantics
        auto scalar_type =
            node->input(0)->type()->expectRef<TensorType>().scalarType();
        if (!scalar_type.has_value() ||
            !at::isFloatingType(scalar_type.value())) {
          auto inps = node->inputs();
          if (!inps.at(1)->type()->isSubtypeOf(IntType::get()) ||
              !inps.at(2)->type()->isSubtypeOf(IntType::get())) {
            continue;
          }
        }

        if (node->get<at::Scalar>(attr::alpha)->toDouble() == 1 &&
            node->get<at::Scalar>(attr::other)->toDouble() == 0) {
          GRAPH_UPDATE(
              getHeader(node),
              " (x + 0 == x - 0 == x) is replaced with ",
              node->input(0)->debugName());
          node->output()->replaceAllUsesWith(node->input(0));
        }
      } else if (
          node->matches(
              "aten::mul(Tensor self, Scalar other) -> Tensor",
              /*const_inputs=*/attr::other) ||
          node->matches(
              "aten::div(Tensor self, Scalar other) -> Tensor",
              /*const_inputs=*/attr::other)) {
        // x * 1 == x / 1 == x
        // is the node is a division or other isn't an integer, than removing
        // this operator could remove type promotion and affect semantics
        auto scalar_type =
            node->input(0)->type()->expectRef<TensorType>().scalarType();
        if (!scalar_type.has_value() ||
            !at::isFloatingType(scalar_type.value())) {
          if (node->kind() == aten::div ||
              !node->input(1)->type()->isSubtypeOf(IntType::get())) {
            continue;
          }
        }

        if (node->get<at::Scalar>(attr::other)->toDouble() == 1) {
          GRAPH_UPDATE(
              getHeader(node),
              " (x * 1 == x / 1 == x) is replaced with ",
              node->input(0)->debugName());
          node->output()->replaceAllUsesWith(node->input(0));
        }
      }
    }
  }

  void optimizeFusedGraphs() {
    for (Node* node : block_->nodes()) {
      if (node->kind() != kind_) {
        continue;
      }
      auto subgraph = node->g(attr::Subgraph);
      GRAPH_DEBUG("before optimizing: ", *subgraph);
      removeNoopBinaryOps(subgraph->block());
      EliminateDeadCode(subgraph);
      EliminateCommonSubexpression(subgraph);
      ConstantPooling(subgraph);
      GRAPH_DEBUG("after optimizing: ", *subgraph);
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
        bool changed = false;
        std::tie(it, changed) = scanNode(*it);
        any_changed |= changed;
      }
    }

    GRAPH_DEBUG("after scan and merge", *graph_);
    refreshAliasDb();

    optimizeFusedGraphs();

    // The graph fuser can add intermediate prim::BroadcastingChunk nodes.
    // Replace them with broadcasts + chunks.
    replaceIntermediateBroadcastingChunks();

    // Fuse starting chunks into the group.
    // for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();) {
    //  it = scanNodeForChunks(*it);
    //}

    GRAPH_DEBUG("before removeOutputsUsedOnlyInSize", *graph_);
    // Remove outputs that have been added only because we need their size
    for (Node* n : block_->nodes()) {
      removeOutputsUsedOnlyInSize(n);
    }
    GRAPH_DEBUG("after removeOutputsUsedOnlyInSize", *graph_);

    for (Node* node : block_->nodes()) {
      for (Block* sub_block : node->blocks()) {
        CudaGraphFuser sub_block_cfg(sub_block, graph_);
        sub_block_cfg.run();
        // Accumulate runtime shapes for all sub-blocks
        fusion_value_to_runtime_shape_.insert(
            sub_block_cfg.fusion_value_to_runtime_shape_.begin(),
            sub_block_cfg.fusion_value_to_runtime_shape_.end());
      }
    }
  }
};

void removeCudaFusionPathForGuardNode(Node* n) {
  auto uses = n->output()->uses();
  TORCH_INTERNAL_ASSERT(
      uses.size() == 1,
      "CudaFusionGuard should only be used once by prim::If or prim::ListConstruct");
  Node* if_node = uses[0].user;
  if (if_node->kind() != prim::If) {
    TORCH_INTERNAL_ASSERT(
        if_node->kind() == prim::ListConstruct,
        "CudaFusionGuard is not used by neither prim::If or prim::ListConstruct");
    // break all inputs so producer prim::CudaFusionGuard can be removed later
    if_node->removeAllInputs();
    auto list_use = if_node->output()->uses();
    TORCH_INTERNAL_ASSERT(
        list_use.size() == 1 && list_use[0].user->kind() == aten::all,
        "prim::ListConstruct should only be used once by aten::all");
    auto all_use = list_use[0].user->output()->uses();
    TORCH_INTERNAL_ASSERT(
        all_use.size() == 1 && all_use[0].user->kind() == prim::If,
        "aten::all should only be used once by prim::If");
    if_node = all_use[0].user;
  }

  auto fall_back_graph = if_node->blocks()[1];
  Node* fallback_node = nullptr;
  for (auto fb_n : fall_back_graph->nodes()) {
    TORCH_INTERNAL_ASSERT(
        fb_n->kind() == prim::FallbackGraph,
        "CudaFusionGuard fallback path should only have single fallback node");
    TORCH_INTERNAL_ASSERT(
        fallback_node == nullptr,
        "CudaFusionGuard fallback path should only have single fallback node");
    fallback_node = fb_n;
  }

  TORCH_INTERNAL_ASSERT(
      fallback_node != nullptr,
      "CudaFusionGuard fallback path found no fallback node");
  fallback_node->moveBefore(n);

  TORCH_INTERNAL_ASSERT(
      fallback_node->outputs().size() == if_node->outputs().size(),
      "CudaFusionGuard fallback should have same number of outputs as with nesting if block");

  if_node->replaceAllUsesWith(fallback_node);
  if_node->destroy();
  n->destroy();
}

bool missingCompleteTypes(const std::vector<TypePtr>& types) {
  for (const auto& type : types) {
    if (auto tensor_type = type->cast<TensorType>()) {
      // if we found one missing value, we know that we are not going to able to
      // generate a kernel, so we bail out;
      if (!tensor_type->device().has_value() ||
          !tensor_type->dim().has_value() ||
          !tensor_type->scalarType().has_value()) {
        return true;
      }
    }
  }
  return false;
}

void removeFusionWithMissingProfilingInformation(Block* block) {
  FUSER_PERF_SCOPE("compileFusionRecursive");
  std::vector<Node*> removeCudaFusionNodes;

  for (auto node : block->nodes()) {
    if (node->kind() == prim::CudaFusionGuard &&
        missingCompleteTypes(node->tys(attr::types))) {
      removeCudaFusionNodes.push_back(node);
    }
    for (auto sub_block : node->blocks()) {
      removeFusionWithMissingProfilingInformation(sub_block);
    }
  }

  for (auto node : removeCudaFusionNodes) {
    removeCudaFusionPathForGuardNode(node);
  }
}

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

// view_sizes_runtime is the profiled-ivalue argument for view-size.
// view_sizes_constant_list is the constant list recorded during profiling runs.
Value* guardView(
    Node* fusion,
    std::unordered_map<Value*, Value*>& fusion_value_to_runtime_size,
    Node* versioning_if,
    Node* view,
    Value* view_sizes_runtime) {
  // 1. Get self tensor sizes and view_sizes
  auto self_value = view->inputs().front();
  auto self_type = self_value->type()->cast<TensorType>();
  auto self_sizes_constant_list = getTensorSizes(self_type);

  auto view_sizes_constant_list =
      constant_as<c10::List<int64_t>>(view->inputs().back());
  TORCH_INTERNAL_ASSERT(view_sizes_constant_list.has_value());
  std::vector<int64_t> view_sizes = view_sizes_constant_list->vec();
  // 2. Get constraints for self tensor and view_sizes
  auto constraints =
      analyzeViewConstraint(self_sizes_constant_list, view_sizes);

  // 3. Add constraints as constant to graph
  auto full_constraints = fusion->owningGraph()->insertConstant(
      IValue(constraints.conglomerateString()));
  full_constraints->node()->moveBefore(versioning_if);

  // 4. Create CudaFusionViewGuard using input tensor, profile_ivalue
  // for view_sizes list, and constraints
  TORCH_INTERNAL_ASSERT(
      fusion_value_to_runtime_size.find(self_value) !=
          fusion_value_to_runtime_size.end(),
      "Failed to find runtime size for fusion value:\t",
      self_value->node()->kind().toDisplayString());
  Node* viewcheck_node =
      fusion->owningGraph()
          ->create(
              c10::Symbol::fromQualString("prim::CudaFusionViewGuard"),
              {fusion_value_to_runtime_size.at(self_value),
               view_sizes_runtime,
               full_constraints},
              1)
          ->insertBefore(versioning_if);
  return viewcheck_node->output();
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
void guardFusionGroup(
    Node* fusion,
    std::unordered_map<Value*, Value*>& fusion_value_to_runtime_size) {
  // Fixup types of the subgraph inputs
  std::vector<TypePtr> guard_types;
  std::vector<Value*> tensor_inputs_to_check;
  std::set<size_t> profiled_ivalue_indices;

  for (const auto index : c10::irange(fusion->inputs().size())) {
    Value* input = fusion->inputs()[index];
    if (input->type()->cast<TensorType>()) {
      // We only check inputs of the fusion group and expect NNC to infer
      // intermediates and outputs shapes

      // note: modified from original implementation, we are guarding fusion
      //       outputs
      if (input->node()->kind() == prim::Constant) {
        continue;
      }
      tensor_inputs_to_check.push_back(input);
      guard_types.push_back(input->type());
    } else if (input->node()->kind() == prim::profile_ivalue) {
      // Conditional constant from profiled_ivalue, should be guarded
      profiled_ivalue_indices.insert(index);
    }
  }

  // insert the if block first;
  auto versioning_if =
      fusion->owningGraph()->create(prim::If, fusion->outputs().size());
  for (const auto idx : c10::irange(fusion->outputs().size())) {
    versioning_if->output(idx)->setType(fusion->output(idx)->type());
    fusion->output(idx)->replaceAllUsesWith(versioning_if->output(idx));
  }
  auto true_block = versioning_if->addBlock();
  auto false_block = versioning_if->addBlock();

  // insert typecheck_node;
  Node* typecheck_node =
      fusion->owningGraph()
          ->create(prim::CudaFusionGuard, tensor_inputs_to_check, 1)
          ->insertBefore(fusion);
  // fix output to BoolType
  typecheck_node->output()->setType(BoolType::get());
  Value* typecheck_result = typecheck_node->output();
  typecheck_node->tys_(attr::types, guard_types);

  versioning_if->insertAfter(typecheck_node);

  auto fusion_graph = fusion->g(attr::Subgraph);
  std::vector<Value*> check_flags = {};

  // Fill in the false block. It should contain the unoptimized
  // copy of the fused subgraph, unless we have conditional constants from
  // profiled_ivalue;
  std::shared_ptr<Graph> fb_graph; // resource holder;
  // Restore the dependency for constant introduced by profiled_ivalue within
  // the graph.
  if (!profiled_ivalue_indices.empty()) {
    // This is necessary as it cleans up the fallback graph, which was copied
    // from subgraph, since the two graph would differ as we cannot use
    // conditional constant in fallback

    // 1. RESTORE conditional constant dependency in fallback group;
    fb_graph = fusion_graph->copy();
    GRAPH_DEBUG("re-wiring fallback graph", *fb_graph);

    for (const auto& offset : profiled_ivalue_indices) {
      auto val = fb_graph->inputs()[offset];
      auto uses = val->uses();
      // since we are updating use of val in the loop, we have to copy
      // val->uses() before hand.
      for (const auto& use : uses) {
        // re-wire inputs and remove conditional constant nodes;
        TORCH_INTERNAL_ASSERT(
            use.user->kind() == prim::Constant,
            "profile_ivalue at index: ",
            offset,
            " can only be used by conditional constant, instead got: ",
            use.user->kind().toDisplayString());
        use.user->output()->replaceAllUsesWith(val);
        use.user->destroy();
      }
    }

    WithInsertPoint guard(false_block->return_node());
    const auto subgraph_outputs =
        insertGraph(*fusion->owningGraph(), *fb_graph, fusion->inputs());
    for (Value* output : subgraph_outputs) {
      false_block->registerOutput(output);
    }
    // types get copied to the fallback graph, so remove specializations before
    // replacing
    // TODO: this is not exposed here, I need to remove that before inserting
    // the graph
    // removeTensorTypeSpecializations(false_block);
    replaceBlockWithFallbackGraph(false_block, fusion->inputs());

    // 2. REMOVE conditional constant dependency in fusion group
    size_t compensation = 0;

    // get a constant true, which is used by `and` pattern later
    auto const_true = fusion->owningGraph()->insertConstant(IValue(true));
    const_true->node()->moveBefore(versioning_if);

    for (const auto& original_offset : profiled_ivalue_indices) {
      size_t offset = original_offset - compensation;

      // step a. handle fusion
      // remove inputs to fusion, and update check logic for fallback
      auto profiled_ival = fusion->input(offset)->node()->input();
      auto const_o = createConditionalConstant(fusion->input(offset)->node());
      TORCH_INTERNAL_ASSERT(
          const_o,
          "profile_ivalue node are expected to have profile information, at node: ",
          *fusion->input(offset)->node());
      const_o->node()->moveBefore(versioning_if);
      Value* ivalue_check = nullptr;

      if (fusion->input(offset)->node()->hasAttribute(
              Symbol::attr("profiled_bool"))) {
        // aten::eq doesn't support comparison between two boolean
        auto xor_n = fusion->owningGraph()
                         ->create(aten::__xor__, {profiled_ival, const_o}, 1)
                         ->insertBefore(versioning_if);
        xor_n->output()->setType(BoolType::get());
        ivalue_check =
            fusion->owningGraph()
                ->create(aten::__xor__, {xor_n->output(), const_true}, 1)
                ->insertBefore(versioning_if)
                ->output();
      } else if (fusion->input(offset)->node()->hasAttribute(
                     Symbol::attr("profiled_reduction_size"))) {
        // TODO(profile_size): check sizes here with special size comparison op
        // TORCH_INTERNAL_ASSERT(false, "not implemented yet");
        ivalue_check =
            fusion->owningGraph()
                ->create(
                    c10::Symbol::fromQualString("prim::CudaFusionSizeEq"),
                    {profiled_ival, const_o},
                    1)
                ->insertBefore(versioning_if)
                ->output();
      } else if (fusion->input(offset)->node()->hasAttribute(
                     Symbol::attr("profiled_view_size"))) {
        // TODO: Add support for dynamic split to view guard

        // Path from profile-ivalue to prim::view_copy operation
        // profile-ivalue -> Constant -> CudaFusionGroup
        // Get argument position in CudaFusionGroup
        // Get argument in subgraph for CudaFusionGroup
        // CudaFusionGroup argument -> Constant List -> prim::view_copy
        auto subgraph_arg = fusion_graph->inputs()[offset];
        auto constant = subgraph_arg->uses().front().user->output();

        TORCH_INTERNAL_ASSERT(!constant->uses().empty());
        auto view = constant->uses().front().user;
        TORCH_INTERNAL_ASSERT(
            view->kind() == prim::view_copy ||
            view->kind() == prim::reshape_copy);

        ivalue_check = guardView(
            fusion,
            fusion_value_to_runtime_size,
            versioning_if,
            view,
            profiled_ival);
      } else if (fusion->input(offset)->node()->hasAttribute(
                     Symbol::attr("profiled_ival"))) {
        ivalue_check =
            fusion->owningGraph()
                ->create(
                    c10::Symbol::fromQualString("prim::CudaFusionIvalGuard"),
                    {profiled_ival, const_o},
                    1)
                ->insertBefore(versioning_if)
                ->output();
      } else {
        ivalue_check = fusion->owningGraph()
                           ->create(aten::eq, {profiled_ival, const_o}, 1)
                           ->insertBefore(versioning_if)
                           ->output();
      }
      ivalue_check->setType(BoolType::get());

      // aggregate flags;
      check_flags.emplace_back(ivalue_check);

      // remove inputs to fusion;
      fusion->removeInput(offset);

      // step b. remove the extra dependency inside fusion;
      for (const auto& use : fusion_graph->inputs()[offset]->uses()) {
        TORCH_INTERNAL_ASSERT(
            use.user->kind() == prim::Constant,
            "profile_ivalue at index: ",
            offset,
            " can only be used by conditional constant, instead got: ",
            use.user->kind().toDisplayString());
        use.user->removeAllInputs();
      }
      fusion_graph->eraseInput(offset);
      compensation++;
    }
    // update graph in fusion node
    fusion->g_(attr::Subgraph, fusion_graph);
  }

  if (!check_flags.empty()) {
    // attaching output from CudaFusionGuard to profile ivalue checks
    check_flags.emplace_back(typecheck_result);
    auto graph = fusion->owningGraph();
    auto bool_list_node =
        graph->insertNode(graph->createList(BoolType::get(), check_flags));
    bool_list_node->moveBefore(versioning_if);
    Value* bool_list = bool_list_node->output();
    // new typecheck_result
    typecheck_result = graph->insert(aten::all, {bool_list});
    typecheck_result->node()->moveBefore(versioning_if);
  }

  if (profiled_ivalue_indices.empty()) {
    WithInsertPoint guard(false_block->return_node());
    const auto subgraph_outputs =
        insertGraph(*fusion->owningGraph(), *fusion_graph, fusion->inputs());
    for (Value* output : subgraph_outputs) {
      false_block->registerOutput(output);
    }
    // types get copied to the fallback graph, so remove specializations before
    // replacing
    // TODO: this is not exposed here, I need to remove that before inserting
    // the graph
    // removeTensorTypeSpecializations(false_block);
    replaceBlockWithFallbackGraph(false_block, fusion->inputs());
  }

  // wiring up if block
  versioning_if->addInput(typecheck_result);

  // Fill in the true block. It has all inputs type-checked and its
  // body should be the fusion group node.
  fusion->moveBefore(true_block->return_node());
  for (Value* output : fusion->outputs()) {
    true_block->registerOutput(output);
  }
}

void guardFusionGroups(
    Block* block,
    std::unordered_map<Value*, Value*>& fusion_value_to_runtime_size) {
  std::vector<Node*> fusions;
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      guardFusionGroups(b, fusion_value_to_runtime_size);
    }
    if (n->kind() == prim::CudaFusionGroup) {
      fusions.push_back(n);
    }
  }
  for (Node* fusion : fusions) {
    // step 1: a. add prim::CudaFusionGuard and fallback logic
    //         b. insert guard logic of profile_ivalue with if block
    //         c. restore conditional constant to non-constant for fallback
    guardFusionGroup(fusion, fusion_value_to_runtime_size);
  }
}

void dumpFusionGroups(std::shared_ptr<Graph>& g) {
  DepthFirstGraphNodeIterator it(g);
  Node* n = nullptr;
  GRAPH_DEBUG("Exporting all NVFuser fusions:");
  while ((n = it.next()) != nullptr) {
    if (n->kind() == prim::FallbackGraph) {
      GRAPH_EXPORT("", n->g(attr::Subgraph));
    }
  }
}

// rewire const integer index & empty byte-typed reserve space tensor outputs,
// so `CudaFusionGroup` doesn't have to handle those
void alterBatchNormImplIndex(Node* node) {
  std::set<size_t> bn_index_out_indices;
  std::set<size_t> bn_buffer_out_indices;

  auto subgraph = node->g(attr::Subgraph);
  for (const auto i : c10::irange(subgraph->outputs().size())) {
    auto val = subgraph->outputs()[i];
    if (val->node()->kind() == aten::_batch_norm_impl_index &&
        val->offset() == 4) {
      bn_index_out_indices.emplace(i);
    } else if (
        val->node()->kind() == aten::_batch_norm_impl_index &&
        val->offset() == 3) {
      bn_buffer_out_indices.emplace(i);
    }
  }

  if (!bn_index_out_indices.empty()) {
    // we output index to 0 so backwards go through native_batch_norm, which is
    // what we support;
    auto const_1 = node->owningGraph()->insertConstant(IValue(0));
    const_1->node()->moveBefore(node);
    for (auto i : bn_index_out_indices) {
      node->outputs()[i]->replaceAllUsesWith(const_1);
    }
  }

  if (!bn_buffer_out_indices.empty()) {
    auto graph = node->owningGraph();
    std::vector<int64_t> sizes{0}; // empty tensor with no size;
    // std::vector<int64_t> sizes; // empty tensor with no size;
    auto const_size_0 = node->owningGraph()->insertConstant(IValue(sizes));
    const_size_0->node()->moveBefore(node);
    auto const_0 = node->owningGraph()->insertConstant(IValue(0));
    const_0->node()->moveBefore(node);
    auto none_val = node->owningGraph()->insertConstant(IValue());
    none_val->node()->moveBefore(node);
    auto device =
        graph->insertNode(graph->create(prim::device, {node->inputs()[0]}, 1));
    device->moveBefore(node);
    device->output()->setType(DeviceObjType::get());
    auto empty_tensor = graph->insertNode(graph->create(
        aten::empty,
        {const_size_0, const_0, none_val, device->output(), none_val, none_val},
        1));
    empty_tensor->moveBefore(node);
    for (auto i : bn_buffer_out_indices) {
      node->outputs()[i]->replaceAllUsesWith(empty_tensor->output());
    }
  }

  bn_index_out_indices.insert(
      bn_buffer_out_indices.begin(), bn_buffer_out_indices.end());
  for (auto iter = bn_index_out_indices.crbegin();
       iter != bn_index_out_indices.crend();
       ++iter) {
    subgraph->eraseOutput(*iter);
    node->eraseOutput(*iter);
  }
}

// rewire empty byte-typed reserve space tensor input to an empty float-typed
// tensor, because `CudaFusionGroup` doesn't support byte-typed tensor, nor does
// it use reserve space.
void alterBatchNormImplIndexBackward(Node* node) {
  std::set<size_t> bn_buffer_in_indices;

  auto subgraph = node->g(attr::Subgraph);
  for (auto n : subgraph->nodes()) {
    if (n->kind() == aten::_batch_norm_impl_index_backward) {
      // 11th inputs are `reserve`, which is not used by codegen kernel and its
      // type is not supported `Byte`. So we disconnect it here to avoid codegen
      // error
      auto byte_input = n->inputs()[11];
      // TODO: let's check the data type for buffer and skip if it's good
      // TODO: we can actually support it by adding an extra inputs to the
      // subgraph
      // TODO: assert on empty buffer
      TORCH_INTERNAL_ASSERT(
          byte_input->node() == subgraph->param_node(),
          "Assumption that reserve input to aten::_batch_norm_impl_index_backward comes from forward graph is broken");
      bn_buffer_in_indices.emplace(byte_input->offset());
    }
  }

  if (!bn_buffer_in_indices.empty()) {
    auto graph = node->owningGraph();
    std::vector<int64_t> sizes{0}; // empty tensor with no size;
    // std::vector<int64_t> sizes{}; // empty tensor with no size;
    auto const_size_0 = node->owningGraph()->insertConstant(IValue(sizes));
    const_size_0->node()->moveBefore(node);
    auto const_0 = node->owningGraph()->insertConstant(IValue(6));
    const_0->node()->moveBefore(node);
    auto none_val = node->owningGraph()->insertConstant(IValue());
    none_val->node()->moveBefore(node);
    auto device =
        graph->insertNode(graph->create(prim::device, {node->inputs()[1]}, 1));
    device->moveBefore(node);
    device->output()->setType(DeviceObjType::get());
    auto empty_tensor = graph->insertNode(graph->create(
        aten::empty,
        {const_size_0, const_0, none_val, device->output(), none_val, none_val},
        1));
    empty_tensor->moveBefore(node);

    for (const auto& item : bn_buffer_in_indices) {
      subgraph->inputs()[item]->setType(
          node->inputs()[item]->type()->cast<TensorType>()->withScalarType(
              at::ScalarType::Float));
      node->replaceInput(item, empty_tensor->output());
    }
  }
}

void alterBatchNormImpls(Block* block) {
  std::vector<Node*> fusions;
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      alterBatchNormImpls(b);
    }
    if (n->kind() == prim::CudaFusionGroup) {
      fusions.push_back(n);
    }
  }
  for (Node* fusion : fusions) {
    // remove index & reserve from outputs;
    alterBatchNormImplIndex(fusion);
    // remove reserve from inputs;
    alterBatchNormImplIndexBackward(fusion);
  }
}

// We absorb `prim::dtype` node into CudaFusion structure. The structure below
//
// %1 = prim::CudaFusionGuard(...)
// %2, %3 = prim::If(...)
//   block0():
//     %4, %5 = prim::CudaFusionGroup(...)
//     -> (%4, %5)
//   block1():
//     %6, %7 = prim::FallbackGraph(...)
//     -> (%6, %7)
// %4 = prim::dtype(%3)
//   ... (uses %2, %4, but never reference to %3 any more)
//
// is updated to:
//
// %1 = prim::CudaFusionGuard(...)
// %2, %3 = prim::If(...)
//   block0():
//     %4 = prim::CudaFusionGroup(...)  # %5 is also removed from subgraph
//     %8 = prim::Constant[value=...]()
//     -> (%4, %8)
//   block1():
//     %6, %7 = prim::FallbackGraph(...)
//     %9 = prim::dtype(%7)
//     -> (%6, %9)
// # %4 = prim::dtype(%3) is removed. All reference to %4 is replaced with %3
//   ... (uses %2, %4, but never reference to %3 any more)
void removeOutputUsedOnlyInDtype(Node* fusion_node) {
  auto fusion_block = fusion_node->owningBlock();
  TORCH_INTERNAL_ASSERT(
      fusion_block->owningNode() &&
          fusion_block->owningNode()->kind() == prim::If,
      "CudaFusionGroup should be inside `prim::CudaFusionGuard` / `prim::If`");

  auto if_node = fusion_block->owningNode();
  auto fusion_node_graph = fusion_node->g(attr::Subgraph);
  auto fallback_block = if_node->blocks()[1];

  bool updated = false;
  // Iterating in this order is crucial for correctness (i has to reflect the
  // current true index of outputs[i])!
  for (int64_t i = static_cast<int64_t>(if_node->outputs().size()) - 1; i >= 0;
       --i) {
    auto output = if_node->outputs()[i];
    // output only used in dtype, we eliminate the output and rely on
    // profiled/static scalar type inference to save on memory IO.
    if (usedOnlyInDtype(output)) {
      updated = true;
      {
        // update fusion_block to output profiled scalar type
        auto fusion_output = fusion_block->outputs()[i];
        auto tensor_type = fusion_output->type()->cast<TensorType>();
        TORCH_INTERNAL_ASSERT(
            tensor_type, "non tensor fed to dtype is not supported");
        auto scalar_type = tensor_type->scalarType();
        TORCH_INTERNAL_ASSERT(
            scalar_type.has_value(),
            "ScalarType should be static for Tensors in fusion for amp optimization");
        auto type_const =
            fusion_block->owningGraph()->insertConstant(IValue(scalar_type));
        type_const->setType(IntType::get());
        type_const->node()->moveBefore(fusion_block->return_node());
        fusion_block->replaceOutput(i, type_const);

        // removing the dangling output tensor from CudaFusionGroup would
        // require tracing output i from block to output j in CudaFusionGroup.
        // We choose to instead do that later by simply checking uses
      }

      {
        // update fallback_block to output dtype instead of tensor
        auto tensor_output = fallback_block->outputs()[i];
        auto dtype_node = fallback_block->owningGraph()->create(
            prim::dtype, tensor_output, 1);
        dtype_node->output()->setType(IntType::get());
        fallback_block->appendNode(dtype_node);
        fallback_block->replaceOutput(i, dtype_node->output());
      }

      // we just shot-cut the `dtype` node since we are already outputing dtype
      auto uses = output->uses();
      for (Use u : uses) {
        AT_ASSERT(u.user->matches("prim::dtype(Tensor a) -> int"));
        u.user->output()->replaceAllUsesWith(output);
        u.user->destroy();
      }
      output->setType(IntType::get());
    }
  }

  if (updated) {
    // Remove fusion node output with no uses;
    for (int64_t i = static_cast<int64_t>(fusion_node->outputs().size()) - 1;
         i >= 0;
         --i) {
      if (fusion_node->output(i)->uses().empty()) {
        GRAPH_UPDATE(
            "removing output: ", i, " from fusion node: ", *fusion_node);
        fusion_node->eraseOutput(i);
        fusion_node_graph->eraseOutput(i);
      }
    }

    fusion_node->g_(attr::Subgraph, fusion_node_graph);
  }
}

// For output tensors in fusion group that is only used by dtype node, with
// CudaFusionGuard, we can short-cut it with constant dtype directly instead to
// save IO memory bandwidth.
// The reason that we do it after we insert the guard, instead of doing it along
// during graph fusion/partitioning, is that we needed to handle the fallback
// differently, since fallback is not inside CudaFusionGuard, and hence doesn't
// have the dtype as a constant.
void removeOutputUsedOnlyInDtype(Block* block) {
  std::vector<Node*> fusions;
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      removeOutputUsedOnlyInDtype(b);
    }
    if (n->kind() == prim::CudaFusionGroup) {
      fusions.push_back(n);
    }
  }
  for (Node* fusion : fusions) {
    // remove index & reserve from outputs;
    removeOutputUsedOnlyInDtype(fusion);
  }
}

void RemoveProfileIValue(Node* profile_ivalue) {
  for (const auto& use : profile_ivalue->output()->uses()) {
    if (use.user->kind() == prim::Constant) {
      use.user->output()->replaceAllUsesWith(profile_ivalue->input());
      use.user->destroy();
    }
  }
  profile_ivalue->output()->replaceAllUsesWith(profile_ivalue->input());
  profile_ivalue->destroy();
}

void ExtractProfileIValue(Node* profile_ivalue) {
  auto const_o = createConditionalConstant(profile_ivalue);
  if (const_o) {
    auto const_n = const_o->node();
    const_n->moveAfter(profile_ivalue);
    profile_ivalue->output()->replaceAllUsesAfterNodeWith(const_n, const_o);
    // special wiring, we add this input to constant simply in order to create
    // dependency, which we can trace and remove later;
    const_n->addInput(profile_ivalue->output());
  } else {
    // no profile value available, remove profile_ivalue node;
    RemoveProfileIValue(profile_ivalue);
  }
}

// break `linear` layer into `matmul` and `add_optional`. This allows us to fuse
// the binary operation without supporting gemm.
// Note that we are not breaking `linear` layer without bias.
void decomposeLinearOps(Block* block) {
  std::vector<Node*> linear_nodes;
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      decomposeLinearOps(b);
    }
    // only decompose `linear` layer with bias
    if (n->kind() == aten::linear &&
        !n->input(2)->type()->isSubtypeOf(
            static_cast<c10::TypePtr>(NoneType::get()))) {
      linear_nodes.push_back(n);
    }
  }

  auto graph = block->owningGraph();
  for (Node* n : linear_nodes) {
    WithInsertPoint guard(n);
    auto weight_t = graph->insertNode(graph->create(aten::t, {n->input(1)}, 1));
    auto matmul = graph->insertNode(
        graph->create(aten::matmul, {n->input(0), weight_t->output()}, 1));
    auto input_tensor_type = n->input(0)->type()->cast<c10::TensorType>();
    if (!input_tensor_type) {
      TORCH_WARN_ONCE(
          "linear input 0 is required to be tensor for linear decompose");
      continue;
    }
    auto mat0_size = input_tensor_type->sizes().concrete_sizes();
    auto mat1_size =
        n->input(1)->type()->cast<c10::TensorType>()->sizes().concrete_sizes();

    // TODO: Continuing here is not necessary when we can handle matmul, right
    // now we are splitting the linear between matmul & bias_add. Our fuser can
    // only take the second half and we would need the size information.
    if (!mat0_size.has_value() || !mat1_size.has_value()) {
      TORCH_WARN_ONCE(
          "concrete shape for linear input & weight are required to decompose into matmul + bias");
      continue;
    }

    // only decompose for input with nDims >= 4. since lower rank linear eager
    // is already fused
    if (mat0_size->size() < 4) {
      continue;
    }

    auto out_size = mat0_size.value();
    TORCH_INTERNAL_ASSERT(
        mat1_size->size() == 2 || mat1_size->size() == 1,
        "weight dimension for linear is expected to be 1 or 2, but got: ",
        mat1_size->size());
    if (mat1_size->size() == 2) {
      out_size[out_size.size() - 1] = mat1_size.value()[0];
    } else if (mat1_size->size() == 1) {
      out_size.pop_back();
    }
    matmul->output()->setType(input_tensor_type->withSizes(out_size));

    // TODO: memory stride should be considered here, our inference above is not
    // safe.
    auto bias = graph->insertNode(
        graph->create(prim::add_optional, {matmul->output(0), n->input(2)}, 1));
    bias->output()->setType(matmul->output(0)->type());

    n->output()->replaceAllUsesWith(bias->output());
    n->destroy();
  }
}

// Replace 'operation' with 'operation_copy' to guard alias operations.
// Supports View, Reshape, Squeeze, and Unsqueeze
void replaceAliasOpsWithCopy(std::shared_ptr<Graph>& graph, Block* block) {
  static std::unordered_map<Symbol, Symbol> alias_to_copy_mapping(
      {{aten::expand, prim::expand_copy},
       {aten::expand_as, prim::expand_as_copy}});
  // TODO: revert disabled aten::view
  //    ({{aten::view, prim::view_copy},
  //     {aten::reshape, prim::reshape_copy},
  //     {aten::squeeze, prim::squeeze_copy},
  //     {aten::unsqueeze, prim::unsqueeze_copy},
  //     {aten::flatten, prim::flatten_copy}});

  std::vector<Node*> maybe_safe_alias_nodes;
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      replaceAliasOpsWithCopy(graph, b);
    }
    if (alias_to_copy_mapping.find(n->kind()) != alias_to_copy_mapping.end()) {
      maybe_safe_alias_nodes.push_back(n);
    }
  }

  auto alias_db = std::make_unique<AliasDb>(graph);

  auto safeToChangeAliasToCopy = [&alias_db](Node* n) {
    return !alias_db->hasWriters(n->input(0)) &&
        !alias_db->hasWriters(n->output(0));
  };

  auto replaceAliasWithCopy = [&graph, &alias_db](Node* n) {
    WithInsertPoint guard(n);
    auto copy_op = graph->insertNode(
        graph->create(alias_to_copy_mapping[n->kind()], n->inputs(), 1));
    copy_op->output()->setType(n->output(0)->type());

    // adding newly created value into alias_db;
    alias_db->createValue(copy_op->output());

    n->output()->replaceAllUsesWith(copy_op->output());
    n->destroy();
  };

  for (Node* n : maybe_safe_alias_nodes) {
    if (!safeToChangeAliasToCopy(n)) {
      continue;
    }
    replaceAliasWithCopy(n);
  }
}

// Revert all 'operation_copy' with 'operation' except in CudaFusionGroup
// e.g., Any non-fused alias operation including within the prim::FallbackGraph
// Supports View, Reshape, Squeeze, and Unsqueeze
void revertAliasCopyOps(std::shared_ptr<Graph>& graph, Block* block) {
  static std::unordered_map<Symbol, Symbol> copy_to_alias_mapping(
      {{prim::expand_copy, aten::expand},
       {prim::expand_as_copy, aten::expand_as}});
  // TODO: revert disabled aten::view
  //    ({{prim::view_copy, aten::view},
  //     {prim::flatten_copy, aten::flatten},
  //     {prim::reshape_copy, aten::reshape},
  //     {prim::squeeze_copy, aten::squeeze},
  //     {prim::unsqueeze_copy, aten::unsqueeze}});

  std::vector<Node*> alias_copy_ops;
  for (Node* n : block->nodes()) {
    // Allow alias copy ops in CudaFusionGroup
    if (n->kind() == prim::CudaFusionGroup) {
      continue;
    }
    // Revert alias copy ops within FallbackGraph
    if (n->kind() == prim::FallbackGraph) {
      auto subgraph = n->g(attr::Subgraph);
      revertAliasCopyOps(subgraph, subgraph->block());
    }
    for (Block* b : n->blocks()) {
      revertAliasCopyOps(graph, b);
    }
    // Revert any non-fused alias copy ops
    if (copy_to_alias_mapping.find(n->kind()) != copy_to_alias_mapping.end()) {
      alias_copy_ops.push_back(n);
    }
  }

  auto replaceCopyWithAlias = [&graph](Node* n) {
    WithInsertPoint guard(n);
    auto alias_op = graph->insertNode(
        graph->create(copy_to_alias_mapping[n->kind()], n->inputs(), 1));
    alias_op->output()->setType(n->output(0)->type());
    n->output()->replaceAllUsesWith(alias_op->output());
    n->destroy();
  };

  for (Node* n : alias_copy_ops) {
    replaceCopyWithAlias(n);
  }
}

// break `conv2d` layer into `conv2d` and `add_optional`. This allows us to fuse
// the binary operation without supporting gemm.
// Note that we are not breaking `conv2d` layer without bias.
void decomposeConvOps(Block* block) {
  std::vector<Node*> conv_nodes;
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      decomposeConvOps(b);
    }
    // TODO: expand this to convXd
    // only decompose `conv2d` layer with bias.
    if (n->kind() == aten::conv2d &&
        n->input(2)->type()->isSubtypeOf(TensorType::get())) {
      conv_nodes.push_back(n);
    }
  }

  auto graph = block->owningGraph();
  for (Node* n : conv_nodes) {
    // TODO: only handling conv2d at this moment, expand this to convXd
    WithInsertPoint guard(n);

    auto const_neg_1 = n->owningGraph()->insertConstant(IValue(-1));
    auto const_none = n->owningGraph()->insertConstant(IValue());

    auto bias_tensor_type = n->input(2)->type()->cast<c10::TensorType>();
    auto bias_size_opt = bias_tensor_type->sizes().concrete_sizes();
    if (!bias_size_opt.has_value()) {
      TORCH_WARN_ONCE(
          "concrete shape for bias input is required to decompose into conv + bias");
      continue;
    }
    // bias shape (C)
    auto bias_size = bias_size_opt.value();

    auto tmp = graph->insertNode(
        graph->create(aten::unsqueeze, {n->input(2), const_neg_1}, 1));
    // new shape (C, 1)
    bias_size.emplace_back(1);
    tmp->output()->setType(bias_tensor_type->withSizes(bias_size));

    auto unsqueezed_bias = graph->insertNode(
        graph->create(aten::unsqueeze, {tmp->output(), const_neg_1}, 1));
    // new shape (C, 1, 1)
    bias_size.emplace_back(1);
    unsqueezed_bias->output()->setType(bias_tensor_type->withSizes(bias_size));

    // replace bias input to none
    n->replaceInput(2, const_none);

    // add bias as a new node
    auto bias_n = graph->insertNode(graph->create(
        prim::add_optional, {n->output(0), unsqueezed_bias->output()}, 1));
    bias_n->output()->setType(n->output(0)->type());
    // moving add_optional after conv2d since it uses its output.
    bias_n->moveAfter(n);

    // replace later uses
    n->output(0)->replaceAllUsesAfterNodeWith(bias_n, bias_n->output());
  }
}

bool removeInplaceOperations(const std::shared_ptr<Graph>& graph) {
  // TODO: we should probably get a list that's close to what our fuser handles
  static std::unordered_set<Symbol> inplace_ops = []() {
    std::unordered_set<Symbol> target_ops;
    for (const auto& iter : activation_type_promotion_mapping) {
      std::string name = std::string(iter.first.toQualString()) + "_";
      target_ops.insert(Symbol::fromQualString(name));
    }

    target_ops.insert(Symbol::fromQualString("aten::add_"));
    target_ops.insert(Symbol::fromQualString("aten::mul_"));
    target_ops.insert(Symbol::fromQualString("aten::div_"));
    target_ops.insert(Symbol::fromQualString("aten::sub_"));
    return target_ops;
  }();

  return RemoveTensorMutation(
      graph, [&](Node* node) { return inplace_ops.count(node->kind()) != 0; });
}

// Recursively traverse blocks, gather all nodes with given symbol,
// and then apply mutator function.
void mutateNode(
    Block* block,
    Symbol symbol,
    const std::function<void(Node*)>& func) {
  // Recursively call mutateNode on blocks
  // Gather all nodes with given symbol
  std::vector<Node*> nodes;
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      mutateNode(b, symbol, func);
    }
    if (n->kind() == symbol) {
      nodes.push_back(n);
    }
  }

  // Apply mutator funcion to every node
  for (Node* n : nodes) {
    func(n);
  }
}

// For the given CudaFusionGroup, separate nested views and remove any unused,
// intermediate views
void separateNestedViews(Node* cuda_fusion_group) {
  TORCH_INTERNAL_ASSERT(cuda_fusion_group->kind() == prim::CudaFusionGroup);

  auto isView = [](Node* node) {
    static std::unordered_set<Symbol> alias_op_set(
        {prim::view_copy, prim::reshape_copy});
    return alias_op_set.find(node->kind()) != alias_op_set.end();
  };

  // node -> input / output values
  auto isNestedView = [&isView](Node* node) {
    return isView(node) && isView(node->input(0)->node());
  };

  auto subgraph = cuda_fusion_group->g(attr::Subgraph);
  for (auto node : subgraph->block()->nodes()) {
    if (isNestedView(node)) {
      // grandparent -> (view / reshape) parent -> (view / reshape) node
      auto parent_value = node->input(0);
      auto parent = parent_value->node();

      auto grandparent_value = parent->input(0);
      C10_UNUSED auto grandparent = grandparent_value->node();

      // Before: gp -> x -> n
      // After: gp -> x / gp -> n
      // Delete x if no more uses
      node->replaceInputWith(parent_value, grandparent_value);
      if (!parent->hasUses()) {
        parent->destroy();
      }
    }
  }
}

} // anonymous namespace

void CudaFuseGraph(std::shared_ptr<Graph>& graph) {
  FUSER_PERF_SCOPE("nvFuser::Manager::CudaFuseGraph");
  GRAPH_DUMP("Before Fusion: ", graph);

  // TODO: extract & guard profile_ivalue; but how do we restore it???
  // I don't know how to store edge/node in attribute. so let's abuse data flow
  // dependency and add inputs to conditional constant generated by
  // aten::profile_ivalue
  mutateNode(graph->block(), prim::profile_ivalue, ExtractProfileIValue);
  GRAPH_DEBUG("insert conditional constant from profile_ivalue: ", *graph);

  // TODO: we need to properly restore shape information after fusion.
  // shamelessly use tool from NNC.
  RemoveProfileNodesAndSpecializeTypes(graph);
  GRAPH_DEBUG("After Profiling Nodes Removed: ", *graph);

  // replace inplace operation to functional version to expose fusion
  // opportunities
  removeInplaceOperations(graph);
  GRAPH_DEBUG("Remove inplace operations: ", *graph);

  // TODO: separate passes into different file;
  if (isOptionEnabled(EnableOption::LinearDecomposition)) {
    // TODO: restore decomposition after fusion, in case we are decomposing
    //       operation that can't be fused;
    decomposeLinearOps(graph->block());
  }
  GRAPH_DEBUG("After decompose Linear Ops by nvfuser: ", *graph);

  if (isOptionEnabled(EnableOption::ConvDecomposition)) {
    decomposeConvOps(graph->block());
  }
  GRAPH_DEBUG("After decompose decompose Conv Ops by nvfuser: ", *graph);

  replaceAliasOpsWithCopy(graph, graph->block());
  GRAPH_DEBUG("replace alias_op with alias_copy by nvfuser: ", *graph);

  CudaGraphFuser cgf(graph->block(), graph);
  cgf.run();
  GRAPH_DEBUG("After Fusion: ", *graph);

  // guard input types as well as conditional constants from
  // aten::profile_ivalue
  guardFusionGroups(graph->block(), cgf.fusion_value_to_runtime_shape_);
  GRAPH_DEBUG("After Guard Fusion: ", *graph);

  // mutate `aten::_batch_norm_impl_index` and
  // `aten::_batch_norm_impl_index_backward` node in the fusion group to WAR
  // the lack of fusion support on integer output as well as byte-typed tensor.
  alterBatchNormImpls(graph->block());
  GRAPH_DEBUG("After _batch_norm_impl_index: ", *graph);

  mutateNode(graph->block(), prim::profile_ivalue, RemoveProfileIValue);

  GRAPH_DEBUG("Before remove missing profiling: ", *graph);
  removeFusionWithMissingProfilingInformation(graph->block());
  GRAPH_DEBUG("After remove missing profiling: ", *graph);

  // optimization targeting AMP
  removeOutputUsedOnlyInDtype(graph->block());
  GRAPH_DEBUG("After removeOutputUsedOnlyInDtype: ", *graph);

  mutateNode(graph->block(), prim::CudaFusionGroup, separateNestedViews);
  GRAPH_DEBUG(
      "separate nested and delete redundant views in CudaFusionGroup:", *graph);

  revertAliasCopyOps(graph, graph->block());
  GRAPH_DEBUG("revert alias_copy ops by nvfuser: ", *graph);

  dumpFusionGroups(graph);

  // After FuseGraph some common subexpressions may come back
  EliminateCommonSubexpression(graph);
  // We might have emitted a fair amount of useless shape propagating code, so
  // remove it
  EliminateDeadCode(graph);

  GRAPH_DEBUG("After ECS & Dead code removal: ", *graph);
  // Improve the quality of shape propagation code that was left
  PeepholeOptimizeShapeExpressions(graph->block());
  GRAPH_DEBUG("After PeepholeOptimizeShapeExpressions: ", *graph);

  // TODO: we need to properly restore shape information after fusion.
  // shamelessly use tool from NNC.
  RemoveTensorTypeSpecializations(graph);

  GRAPH_DUMP("Before Compilation: ", graph);
  // Compile CudaFusionGroup
  compileFusionRecursive(graph->block());
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
