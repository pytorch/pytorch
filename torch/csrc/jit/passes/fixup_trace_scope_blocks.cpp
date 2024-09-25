#include <torch/csrc/jit/passes/fixup_trace_scope_blocks.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/lower_tuples.h>

#include <algorithm>

namespace torch::jit {

namespace {

bool isEligibleNode(Node* n) {
  return n->kind() == prim::TracedModuleForward ||
      n->kind() == prim::TracedFork;
}

// This pass does several things:
// 1) It looks at TracedModuleForward nodes and resolves the type of `self`
//    for that (to-be) method call. It adds an input of that type to the
//    block, and adds the TracedAttr value corresponding to that `self`
//    value as a Node input. This ensures `self` is an explicit Use on
//    the node, a property we take advantage of downstream. Example:
// 2) Convert all references to prim::TracedAttr values to prim::GetAttr
//    calls in the tightest scope possible. Concretely, for each use of
//    a prim::TracedAttr value, we compare the scope of that attribute
//    to the scope of the Use. We emit GetAttr nodes for all atoms
//    that are not shared between the two. For example, if an
//    attribute `f.param` is referenced in scope `f`, we emit a
//    GetAttr[name="param"](%self) node in the `f` block, where
//    `self` is the previously-added `self` argument to the block.
// 3) Destroy all the prim::TracedAttr nodes, as they should have
//    no more uses.
//
// A quick example:
//
//
// Input graph:
//
//     graph(%self : ClassType<Module>,
//           %x : Float(3, 4)):
//       %1 : bool = prim::TracedAttr[scope="__module.training"]()
//       %2 : ClassType<Module> = prim::TracedAttr[scope="__module.f"]()
//       %3 : Float(4, 4) = prim::TracedAttr[scope="__module.f.param"]()
//       %4 : bool = prim::TracedAttr[scope="__module.f.training"]()
//       = prim::TracedModuleForward[scope="__module.f"](),
//         block0():
//           %6 : Float(3, 4) = aten::mm(%x, %3),
//           -> ()
//       return (%6)
//
// The diff after step (1)
//
//     -   = prim::TracedModuleForward[scope="__module.f"](),
//     -    block0():
//     +   = prim::TracedModuleForward[scope="__module.f"](%2),
//     +    block0(%self : ClassType<Module>):
//
// The diff after step (2)
//
//       graph(%self.1 : ClassType<Module>,
//             %x : Float(3, 4)):
//       +  %9 : ClassType<Module> = prim::GetAttr[name="f"](%self.1)
//         %1 : bool = prim::TracedAttr[scope="__module.training"]()
//           <....>
//         %4 : bool = prim::TracedAttr[scope="__module.f.training"]()
//       -   = prim::TracedModuleForward[scope="__module.f"](%2),
//       +   = prim::TracedModuleForward[scope="__module.f"](%9),
//           block0(%self : ClassType<Module>):
//       -      %6 : Float(3, 4) = aten::mm(%x, %3),
//       +      %8 : Tensor = prim::GetAttr[name="param"](%self)
//       +      %6 : Float(3, 4) = aten::mm(%x, %8),
//             -> ()
//         return (%6)
//
// The diff after step (3)
//
//       -  %1 : bool = prim::TracedAttr[scope="__module.training"]()
//       -  %2 : ClassType<Module> = prim::TracedAttr[scope="__module.f"]()
//       -  %3 : Float(4, 4) = prim::TracedAttr[scope="__module.f.param"]()
//       -  %4 : bool = prim::TracedAttr[scope="__module.f.training"]()
struct ConvertTracedAttrReferences {
  void run(const std::shared_ptr<Graph>& graph) {
    // Build a table mapping--for each TracedAttr node--the
    // qualified name of the attribute to the Value* output
    // of the Node.
    buildAttrMap(graph);
    // Step 1
    addSelfArgToTracedForwardNodes(graph->block());
    // Step 2
    convertAttrReferencesToLocalGetAttrs(
        graph->block(), "__module", graph->inputs()[0]);
    // Step 3
    destroyTracedAttrNodes(graph);
  }

 private:
  void buildAttrMap(const std::shared_ptr<Graph>& graph) {
    for (Node* n : graph->nodes()) {
      if (n->kind() == prim::TracedAttr) {
        attr_qualname_to_value[n->s(attr::scope)] = n->output();
      }
    }
  }

  void addSelfArgToTracedForwardNodes(Block* b) {
    for (Node* n : b->nodes()) {
      if (n->kind() == prim::TracedModuleForward) {
        n->addInput(attr_qualname_to_value.at(n->s(attr::scope)));
        n->blocks()[0]->addInput("self")->setType(
            attr_qualname_to_value.at(n->s(attr::scope))->type());
        addSelfArgToTracedForwardNodes(n->blocks()[0]);
      }
      if (n->kind() == prim::TracedFork) {
        addSelfArgToTracedForwardNodes(n->blocks()[0]);
      }
    }
  }

  // This is a recursive function that descends down all blocks in the Graph
  // (NB: not just TracedModuleForward blocks). Each descension has a
  // corresponding `prefix`, i.e. the qualified name of the scope this
  // Block represents (or the scope in which this block resides for
  // non-TracedModuleForward nodes). We use this prefix to make decisions
  // about whether to emit a GetAttr node for an attribute reference, or
  // to defer that emission to the caller (in the case where an attribute
  // reference does not reside in the `prefix` scope).
  std::vector<Value*> convertAttrReferencesToLocalGetAttrs(
      Block* b,
      const c10::QualifiedName& prefix,
      Value* self) {
    // Store away Value*'s which are references to TracedAttr's which are
    // not in the `prefix` scope. We pass this back to the caller, who
    // should add these Values as explicit inputs as well as inductively
    // make the same decision on those Values.
    std::vector<Value*> unresolved_tracedattrs;
    // To ensure we don't emit redundant GetAttr Nodes in a given scope,
    // we maintain this map of original TracedAttr Value* to the Value*
    // corresponding to the GetAttr for that attribute.
    // We don't rely on CSE here because we currently can't reason about
    // the correctness of CSE over GetAttr Nodes (i think)
    std::unordered_map<Value*, Value*> local_remaps;

    for (Node* n : b->nodes()) {
      // The only difference between these two branches is for
      // TracedModuleForward we advance the scope, but for other
      // Nodes with Blocks we don't
      if (n->kind() == prim::TracedModuleForward) {
        auto sub_unresolved = convertAttrReferencesToLocalGetAttrs(
            n->blocks()[0], n->s(attr::scope), n->blocks()[0]->inputs()[0]);
        for (Value* v : sub_unresolved) {
          n->addInput(v);
        }
      } else if (!n->blocks().empty()) {
        for (Block* sub_block : n->blocks()) {
          auto sub_unresolved =
              convertAttrReferencesToLocalGetAttrs(sub_block, prefix, self);
          for (Value* v : sub_unresolved) {
            n->addInput(v);
          }
        }
      }

      for (size_t inp_idx = 0; inp_idx < n->inputs().size(); ++inp_idx) {
        Value* inp = n->input(inp_idx);

        // Short circuit: if we've already emitted a new Value for this
        // attribute, just use that.
        if (local_remaps.count(inp)) {
          n->replaceInput(inp_idx, local_remaps[inp]);
          continue;
        }

        WithInsertPoint guard(b->param_node()->next());
        replaceTracedAttrInputOnNode(
            n, inp_idx, prefix, self, local_remaps, unresolved_tracedattrs);
      } // for (Value *inp : n->inputs())
    } // for (Node *n : b->nodes())
    return unresolved_tracedattrs;
  }

  void replaceTracedAttrInputOnNode(
      Node* n,
      size_t inp_idx,
      const c10::QualifiedName& prefix,
      Value* self,
      std::unordered_map<Value*, Value*>& local_remaps,
      std::vector<Value*>& unresolved_tracedattrs) {
    auto inp = n->inputs()[inp_idx];
    auto inp_node = inp->node();
    auto prefix_atoms = prefix.atoms();
    if (inp_node->kind() == prim::TracedAttr) {
      auto attr_qualname = c10::QualifiedName(inp_node->s(attr::scope));
      if (prefix.isPrefixOf(attr_qualname)) {
        // Prefix case: the attribute resides in this scope or a
        // sub-scope. Continually emit GetAttr nodes until we've reached
        // the proper attribute.
        auto attr_atoms = attr_qualname.atoms();
        Value* replaced_value = self;
        for (const auto i : c10::irange(attr_atoms.size())) {
          if (i < prefix_atoms.size()) {
            TORCH_INTERNAL_ASSERT(attr_atoms[i] == prefix_atoms[i]);
          } else {
            replaced_value = n->owningBlock()->owningGraph()->insertGetAttr(
                replaced_value, attr_atoms[i]);
          } // if (i < prefix_atoms.size())
        } // for(const auto i : c10::irange(attr_atoms.size()))
        n->replaceInput(inp_idx, replaced_value);
        local_remaps[inp] = replaced_value;
      } else {
        // Non-prefix case: this is a use of an attribute somewhere
        // higher in the Module hierarchy. Add a captured input to
        // the block for this attribute and add to the vector of
        // Value*'s for the caller to handle.
        Value* remapped = n->owningBlock()->addInput()->copyMetadata(inp);
        n->replaceInput(inp_idx, remapped);
        unresolved_tracedattrs.push_back(inp);
        local_remaps[inp] = remapped;
      } // if (prefix.isPrefixOf(attr_qualname))
    } // if (inp_node->kind() == prim::TracedAttr)
  }

  // The previous pass should have deleted all uses of TracedAttr
  // nodes. Let's explicitly delete them here.
  void destroyTracedAttrNodes(const std::shared_ptr<Graph>& graph) {
    for (auto& kv : attr_qualname_to_value) {
      kv.second->node()->destroy();
    }
  }

  // For each prim::TracedAttr, record the `scope` value mapped
  // to the Value* in the graph for that attribute.
  std::unordered_map<std::string, Value*> attr_qualname_to_value;
};

// Iterate through all the nodes in program order and--for each use--
// if the Value referenced is not in a scope that dominates the node,
// add block and Node outputs to lift it into a scope in which
// it dominates the Use.
struct MakeDefsDominateUses {
  MakeDefsDominateUses() = default;

  void run(Block* b) {
    processNode(b->param_node(), b);
    for (Node* n : b->nodes()) {
      processNode(n, b);
    }
    processNode(b->return_node(), b);
  }

 private:
  void processNode(Node* n, Block* b) {
    for (size_t i = 0; i < n->inputs().size(); ++i) {
      Value* inp = n->inputs()[i];

      // Already lifted to this level by a previously processed Use, switch to
      // remapped value
      Value* inp_remapped = inp;
      if (remap.count(inp_remapped)) {
        n->replaceInput(i, remap[inp_remapped]);
        inp_remapped = remap[inp_remapped];
      }

      // This conditional isn't strictly necessary, but saves a lot of
      // computation in the common case that we're using a local value.
      if (inp_remapped->node()->owningBlock() != b) {
        // Find the common ancestor block between this node and the node that
        // produced this input. For this input Use to be valid, the Value's
        // def must be present in this common ancestor node.
        Block* common_ancestor =
            n->findCommonAncestorBlockWith(inp_remapped->node());

        Value* v_itr = inp_remapped;
        Block* b_itr = inp_remapped->node()->owningBlock();

        // Starting from the initial def for this input, iterate to
        // wider and wider blocks, adding Block outputs and Node outputs
        // along the way. Then, log the lifted values in the remap table
        // so we can make subsequent Uses refer to the lifted value, if
        // the domination condition is met.
        while (b_itr != common_ancestor) {
          b_itr->registerOutput(v_itr);
          Value* remapped =
              b_itr->owningNode()->addOutput()->setType(v_itr->type());
          v_itr = remapped;
          b_itr = b_itr->owningNode()->owningBlock();
        }
        // From now on, references to `inp` will be replaced with
        // references to `v_itr`, the lifted Value
        remap[inp] = v_itr;
        n->replaceInput(i, remap[inp]);
      }
    }

    if (isEligibleNode(n)) {
      run(n->blocks()[0]);
    }
  }

  // This holds the mapping between a Value* we would see in a Use
  // and the lifted value, if present. We use this to ensure that
  // Uses refer to a Value* that is in a dominating scope.
  using RemappingTable = std::unordered_map<Value*, Value*>;
  RemappingTable remap;
};

// For all blocks except graph->block(), convert multiple block
// returns to a TupleConstruct. This is required for turning the
// blocks into Methods. (and in the case that self is nullptr,
// it is required to properly inline the blocks).
void convertReturnsToTuples(Block* b) {
  for (Node* n : b->nodes()) {
    if (n->kind() == prim::TracedFork) {
      convertReturnsToTuples(n->blocks()[0]);
    } else if (n->kind() == prim::TracedModuleForward) {
      TORCH_INTERNAL_ASSERT(n->blocks().size() == 1);
      convertReturnsToTuples(n->blocks()[0]);

      Graph* g = b->owningGraph();
      Block* sub_block = n->blocks()[0];
      if (sub_block->outputs().size() > 1) {
        {
          // Make block returns go through a Tuple
          WithInsertPoint guard(sub_block->return_node());
          Node* return_tup =
              g->insertNode(g->createTuple(sub_block->outputs()));
          while (!sub_block->outputs().empty()) {
            sub_block->eraseOutput(0);
          }
          sub_block->registerOutput(return_tup->output());
        }

        // Make node outputs a single tuple;
        std::vector<TypePtr> types;
        for (size_t i = 0; i < n->outputs().size(); ++i) {
          types.push_back(n->output(i)->type());
        }
        Value* tup_output = n->addOutput()->setType(TupleType::create(types));
        Node* tup_unpack = g->createTupleUnpack(tup_output)->insertAfter(n);
        for (size_t i = 0; i < tup_unpack->outputs().size(); ++i) {
          auto rev_idx = tup_unpack->outputs().size() - i - 1;
          n->output(rev_idx)->replaceAllUsesWith(tup_unpack->output(rev_idx));
          n->eraseOutput(rev_idx);
        }
      } else if (sub_block->outputs().empty()) {
        WithInsertPoint guard(sub_block->return_node());
        sub_block->registerOutput(g->insertNode(g->createNone())->output());
        n->addOutput()->setType(NoneType::get());
      }
    }
  }
}

// Lambda lift Values (i.e. add Graph inputs for the purpose of
// referencing values that dominate the block) and convert
// the block to a Graph. blocks()[0] on each TracedModuleForward then
// appears as a Graph attribute attr::Subgraph
void lambdaLiftBlocksAndConvertToGraph(Block* b) {
  for (Node* n : b->nodes()) {
    if (isEligibleNode(n)) {
      lambdaLiftBlocksAndConvertToGraph(n->blocks()[0]);

      auto graph = std::make_shared<Graph>();
      std::unordered_map<Value*, Value*> remaps;
      graph->block()->cloneFrom(n->blocks()[0], [&](Value* v) {
        if (!remaps.count(v)) {
          remaps[v] = graph->addInput()->copyMetadata(v);
          n->addInput(v);
        }
        return remaps[v];
      });
      LintGraph(graph);
      n->g_(attr::Subgraph, graph);
      n->eraseBlock(0);
    }
  }
}

// Find a unique name to add this method as
// We try {method_name}, {method_name}1, {method_name}2, ...
std::string mangleMethodName(
    const std::string& method_name,
    const ClassTypePtr& mod_type) {
  for (size_t method_idx = 0;; method_idx++) {
    auto mangled = method_name;
    if (method_idx != 0) {
      mangled += std::to_string(method_idx);
    }
    bool found = false;
    for (Function* fn : mod_type->methods()) {
      if (fn->name() == mangled) {
        found = true;
        break;
      }
    }
    if (!found) {
      return mangled;
    }
  }
  TORCH_INTERNAL_ASSERT(false);
}

// Register the attr::Subgraph Graph values as Functions in the
// class compilation unit and register that Function as a method
// on the corresponding Module in the Module hierarchy. Note that we
// unique the methods by naming them forward, forward1, forward2...
void createMethodCalls(const std::shared_ptr<Graph>& g) {
  for (auto node_itr = g->nodes().begin(); node_itr != g->nodes().end();) {
    Node* n = *node_itr++;
    if (n->kind() == prim::TracedFork) {
      createMethodCalls(n->g(attr::Subgraph));
    } else if (n->kind() == prim::TracedModuleForward) {
      WithInsertPoint ip(n);

      ClassTypePtr callee_mod_type = n->input(0)->type()->expect<ClassType>();

      createMethodCalls(n->g(attr::Subgraph));

      auto mangled_method_name = mangleMethodName("forward", callee_mod_type);
      auto qualname = c10::QualifiedName(
          callee_mod_type->name().value(), mangled_method_name);
      Function* f = callee_mod_type->compilation_unit()->create_function(
          qualname, n->g(attr::Subgraph));
      callee_mod_type->addMethod(f);

      std::vector<NamedValue> nvs;
      for (Value* i : n->inputs()) {
        nvs.emplace_back(i->node()->sourceRange(), i);
      }
      auto schema = matchSchema(f->getSchema(), n->sourceRange(), *g, nvs, {});
      Value* retval = g->insertMethodCall(f->qualname().name(), schema);
      n->output()->replaceAllUsesWith(retval);
      n->destroy();
    }
  }
}

void inlineScopeBlocks(Block* b) {
  for (auto n_itr = b->nodes().begin(); n_itr != b->nodes().end();) {
    Node* n = *n_itr++;
    for (Block* sub_b : n->blocks()) {
      inlineScopeBlocks(sub_b);
    }
    if (n->kind() == prim::TracedModuleForward) {
      // Convert the block to a graph so we can inline it
      auto graph = std::make_shared<Graph>();
      std::unordered_map<Value*, Value*> remaps;
      graph->block()->cloneFrom(n->blocks()[0], [&](Value* v) {
        remaps[v] = graph->block()->addInput()->copyMetadata(v);
        n->addInput(v);
        return remaps[v];
      });

      WithInsertPoint insert_point(n);
      AT_ASSERT(n->inputs().size() == graph->inputs().size());
      auto new_outputs = insertGraph(*n->owningGraph(), *graph, n->inputs());
      const auto& old_outputs = n->outputs();

      AT_ASSERT(new_outputs.size() == old_outputs.size());
      for (const auto i : c10::irange(old_outputs.size())) {
        old_outputs[i]->replaceAllUsesWith(new_outputs[i]);
      }
      n->destroy();
    }
  }
}

void convertTracedForksToRealForks(const std::shared_ptr<Graph>& g) {
  for (auto itr = g->nodes().begin(); itr != g->nodes().end();) {
    Node* n = *itr++;
    if (n->kind() == prim::TracedFork) {
      WithInsertPoint guard(n);
      Node* new_fork_node =
          g->insertNode(g->create(prim::fork, n->outputs().size()))
              ->copyAttributes(*n);
      for (Value* i : n->inputs()) {
        new_fork_node->addInput(i);
      }
      for (size_t i = 0; i < new_fork_node->outputs().size(); ++i) {
        new_fork_node->outputs()[i]->copyMetadata(n->outputs()[i]);
        n->outputs()[i]->replaceAllUsesWith(new_fork_node->outputs()[i]);
      }
      n->destroy();
    }
  }
}

// Run a few clean-up passes to make the graph a bit cleaner.
void runCleanupPasses(const std::shared_ptr<Graph>& g) {
  for (Node* n : g->nodes()) {
    if (n->kind() == prim::TracedFork) {
      auto subgraph = n->g(attr::Subgraph);
      if (getInlineEverythingMode()) {
        Inline(*subgraph);
      }
      convertTracedForksToRealForks(subgraph);
      LowerSimpleTuples(subgraph);
      EliminateDeadCode(subgraph);
      LintGraph(subgraph);
    }
  }
  if (getInlineEverythingMode()) {
    Inline(*g);
  }
  convertTracedForksToRealForks(g);
  LowerSimpleTuples(g);
  EliminateDeadCode(g);
  LintGraph(g);
}

void runCleanupPasses(Module* m) {
  auto methods = m->get_methods();
  for (auto module : m->children()) {
    runCleanupPasses(&module);
  }
  for (auto& method : methods) {
    runCleanupPasses(method.graph());
  }
}

} // namespace

void FixupTraceScopeBlocks(std::shared_ptr<Graph>& graph, Module* self) {
  if (self) {
    ConvertTracedAttrReferences().run(graph);
  } else {
    for (Node* n : graph->nodes()) {
      TORCH_INTERNAL_ASSERT(n->kind() != prim::TracedAttr);
    }
  }
  MakeDefsDominateUses().run(graph->block());
  convertReturnsToTuples(graph->block());
  if (!self) {
    // We have no Module, so we're just going to inline everything.
    // This should give us a totally flat graph.
    inlineScopeBlocks(graph->block());
    // For TracedFork nodes
    lambdaLiftBlocksAndConvertToGraph(graph->block());
    runCleanupPasses(graph);
  } else {
    lambdaLiftBlocksAndConvertToGraph(graph->block());
    createMethodCalls(graph);
    runCleanupPasses(self);
    // `graph` isn't referenced in `self` yet, so we need to run
    // this separately
    runCleanupPasses(graph);
  }
}

} // namespace torch::jit
