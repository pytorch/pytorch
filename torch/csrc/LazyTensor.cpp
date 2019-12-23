#ifndef C10_MOBILE

#include <ATen/core/LazyTensor.h>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/interpreter.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/function_schema_parser.h>


/* LazyTensor Design
 *
 *
 * LazyTensor is an abstraction that enables lazy execution in PyTorch.  It does so by
 * providing a virtual tensor implementation that looks and feels like a regular Tensor
 * but does not necessarily execute any of the underlying computation immediately.
 *
 * Instead of executing the computation "eagerly,"  LazyTensors will record requested
 * operations into an implicit graph and later lower that graph to TorchScript IR.
 * Once `to_eager()` is called, it is guaranteed that any computation required to
 * resolve the tensor has been completed.
 *
 * For backends that digest TorchScript IR, this design is interesting because it enables
 * these backends to be used in PyTorch without the need for a TorchScript frontend.
 *
 * Note: shape propagation is not supported! This means `to_eager` is required before
 * any shape dependent code is run.  The implication is that some core modules,
 * such as nn.BatchNorm, cannot be executed lazily because their implementations
 * check shapes in Python.
 *
 * Note: LazyTensor also accumulates a graph during calls to `backward`.  Shape
 * information is required for error checking during backward propagation,
 * but with LazyTensors we skip these checks.
 *
 * Note: LazyTensor assumes all operations are re-computable!  This assumption
 * saves us from using too much memory without explicit need.  However, it also
 * induces potentially unwanted re-compute and BREAKS semantics for operations
 * that require state.  TODO: blacklist broken ops
 *
 * Note: LazyTensors cannot be used in multi-threaded environment.  This paradigm
 * should subvert the need for parallelism at this level anyway.
 *
 */

namespace at {

// @returns pair of output and required stack inputs
std::pair<torch::jit::Node*, std::vector<const LazyTensorImpl*>> getRequiredGraphValues(
    const LazyTensorImpl* lt_,
    std::shared_ptr<torch::jit::Graph> g) {
  // Convert LazyTensorImpl to node in the graph, inserting before
  // the "dep" node.
  auto createNode = [&g](
                     const LazyTensorImpl* lt,
                     torch::jit::Node* dep) -> torch::jit::Node* {
    auto fs = lt->schema();
    auto s = Symbol::fromQualString(fs.name());
    auto* node = g->create(s, fs.returns().size());
    if (dep) {
      g->setInsertPoint(dep);
    }
    node = g->insertNode(node);
    return node;
  };

  // Setup output of the whole graph
  auto* root_node = createNode(lt_, nullptr);
  torch::jit::Value* output = root_node->outputs()[lt_->outputIndex()];
  std::vector<const LazyTensorImpl*> inputs;

  // Create frontier for resolving deps
  // The form is { IValue, Node (that depends on IValue) }
  std::vector<std::pair<c10::IValue, torch::jit::Node*>> frontier;
  for (const auto& input : lt_->inputs()) {
    frontier.emplace_back(std::make_pair(input, root_node));
  }

  while (frontier.size()) {
    std::vector<std::pair<c10::IValue, torch::jit::Node*>> next_frontier;
    for (auto dep_pair : frontier) {
      auto ival = dep_pair.first;
      auto dep_node = dep_pair.second;

      // Is the IValue a LazyTensor?
      if (ival.isTensor() && ival.toTensor().is_lazy()) {
        auto t = ival.toTensor();
        auto* lt = static_cast<LazyTensorImpl*>(t.unsafeGetTensorImpl());
        // Is lt just storing a tensor? This is an input
        if (lt->isResolved()) {
          auto* v = g->addInput();
          inputs.emplace_back(lt);
          dep_node->addInput(v);
        } else {
          auto* node = createNode(lt, dep_node);
          dep_node->addInput(node->outputs()[lt->outputIndex()]);
          for (const auto& input : lt->inputs()) {
            next_frontier.emplace_back(std::make_pair(input, node));
          }
        }
        // It's a non-LazyTensor
      } else {
        g->setInsertPoint(dep_node);
        torch::jit::Value* v;
        if (ival.isNone()) {
          v = g->insertNode(g->createNone())->output();
        } else {
          v = insertConstant(*g, ival);
        }
        dep_node->addInput(v);
      }
    }

    frontier = next_frontier;
  }

  return std::make_pair(output->node(), inputs);
}

/* Cache bytecode
 *
 * We cache the compiled graphs produced by lazy tensors,
 * acting under the assumption that we will repeatedly call the same
 * graphs.  This cache is implemented with a simple bytecode used
 * as the key.  It is a vector containing elements of two types:
 * 1) schema strings + output indices
 * 2) ivalues themselves
 *
 * We can see each element is isomorphic to the representation of a LazyTensor
 * and it is straightforward to extend this model to a bytecode representing
 * the underlying graph using prefix notation.
 *
 * Here's an example:
 *   Construct a straight line repr of the LT
 *    a = ?.to_lazy()
 *    b = ?.to_lazy()
 *    c = a @ b
 *    d = c * 2
 *   this becomes
 *    mul[0], mm[0], 2, a, b
 *   that is, all unresolved LTs in the graph are "ExprName[index]" and
 *   resolved LTs are just their corresponding IValues
 *
 * As a later optimization, we could do memory comparison on this bytcode.
 */
struct Item {
  Item(std::pair<std::string, size_t> var) : tag(VAR), var_(var) {}
  Item(c10::IValue val) : tag(CONST), val_(val) {}

  enum { VAR, CONST } tag;

  const std::pair<std::string, size_t>& var() const {
    TORCH_INTERNAL_ASSERT(tag == VAR);
    return var_;
  }
  const c10::IValue& val() const {
    TORCH_INTERNAL_ASSERT(tag == CONST);
    return val_;
  }

  // union {
  std::pair<std::string, size_t> var_ = std::make_pair("", 0);
  c10::IValue val_;
  //};
  Item(Item&&) = default;
  ~Item() {}
};

using Key = std::vector<Item>;

bool keyEqual(const Key& k1, const Key& k2) {
  if (k1.size() != k2.size()) {
    return false;
  }
  for (auto i = 0; i < k1.size(); ++i) {
    const auto& a = k1[i];
    const auto& b = k2[i];
    if (a.tag != b.tag) {
      return false;
    }
    if (a.tag == Item::VAR) {
      if (a.var() != b.var()) {
        return false;
      }
    } else {
      if (a.val().isSameIdentity(b.val())) {
        return true;
      }
      if (a.val().isTensor() && b.val().isTensor()) {
        // TODO type check
        return true;
      }
      return false;
    }
  }
  return true;
}

struct GraphCache {
  std::shared_ptr<torch::jit::GraphExecutor> get(const Key& k) {
    for (const auto& pair : gs_) {
      if (keyEqual(pair.first, k)) {
        return pair.second;
      }
    }
    return std::shared_ptr<torch::jit::GraphExecutor>(nullptr);
  }

  void put(Key&& k, std::shared_ptr<torch::jit::GraphExecutor> g) {
    gs_.emplace_back(std::make_pair(std::move(k), g));
  }

  std::list<std::pair<Key, std::shared_ptr<torch::jit::GraphExecutor>>> gs_;
};

static GraphCache graph_cache;

std::pair<
    std::shared_ptr<torch::jit::GraphExecutor>,
    std::vector<const LazyTensorImpl*>>
getGraph(const LazyTensorImpl* lt_) {

  std::vector<Item> record;
  std::vector<c10::IValue> frontier = lt_->inputs();
  std::vector<const LazyTensorImpl*> inputs;
  while (frontier.size()) {
    std::vector<c10::IValue> new_frontier;
    for (const auto& ival : frontier) {
      if (ival.isTensor()) {
        auto t = ival.toTensor();
        if (t.is_lazy()) {
          auto* lt = static_cast<LazyTensorImpl*>(t.unsafeGetTensorImpl());
          if (!lt->isResolved()) {
            // make into Item, push to record
            record.emplace_back(
                std::make_pair(lt->schema().name(), lt->outputIndex()));
            // maybe add to new_frontier
            new_frontier.reserve(new_frontier.size() + lt->inputs().size());
            new_frontier.insert(
                new_frontier.end(), lt->inputs().begin(), lt->inputs().end());
            continue;
          } else {
            inputs.emplace_back(lt);
          }
        }
      }
      record.emplace_back(ival);
    }
    frontier = new_frontier;
  }

  auto ge = graph_cache.get(record);
  if (ge) {
    return std::make_pair(ge, inputs);
  } else {
    inputs.clear();
  }

  auto g = std::make_shared<torch::jit::Graph>();
  torch::jit::Node* output_node;
  std::tie(output_node, inputs) = getRequiredGraphValues(lt_, g);
  for (auto output : output_node->outputs()) {
    g->registerOutput(output);
  }
  ge = std::make_shared<torch::jit::GraphExecutor>(g);
  graph_cache.put(std::move(record), ge);

  return std::make_pair(ge, inputs);
}

Tensor to_eager(LazyTensorImpl* lt) {
  if (lt->isResolved()) {
    auto t = lt->value().toTensor();
    return t;
  }

  std::shared_ptr<torch::jit::GraphExecutor> ge;
  std::vector<const LazyTensorImpl*> inputs;
  std::tie(ge, inputs) = getGraph(lt);
  torch::jit::Stack stack;
  for (auto& inp : inputs) {
    stack.push_back(inp->value());
  }

  ge->run(stack);

  // We registered only the output we care about
  size_t index = 0;
  for (auto output : stack) {
    // Since we calculated the values for siblings, we store those in the tensors
    // themselves
    if (index != lt->outputIndex() && lt->siblings()[index]) {
      lt->siblings()[index]->resolve(stack[index]);
    }
    index++;
  }
  auto output_t = stack[lt->outputIndex()].toTensor();
  return output_t;
}

} // namespace at

namespace at {

using namespace torch::jit;
void lazyRun(const OperatorHandle& op, torch::jit::Stack* s) {
  const FunctionSchema& fs = op.schema();
  // TODO: move this logic into native_functions when it makes sense
  if (fs.name() == "aten::is_leaf") {
    auto t = pop(*s).toTensor();
    auto lt = static_cast<LazyTensorImpl*>(t.unsafeGetTensorImpl());
    if (lt->isResolved()) {
      auto t_ = lt->value().toTensor();
      bool is_leaf = t_.is_leaf();
      push(*s, is_leaf);
      return;
    } else {
      push(*s, false);
      return;
    }
  }
  if (fs.name() == "aten::output_nr") {
    auto t = pop(*s).toTensor();
    auto lt = static_cast<LazyTensorImpl*>(t.unsafeGetTensorImpl());
    if (lt->isResolved()) {
      auto t_ = lt->value().toTensor();
      bool out_nr = t_.output_nr();
      push(*s, out_nr);
      return;
    } else {
      push(*s, 0);
      return;
    }
  }
  if (fs.name() == "aten::to_eager") {
    auto t = pop(*s).toTensor();
    auto lt = static_cast<LazyTensorImpl*>(t.unsafeGetTensorImpl());
    auto tensor = lt->to_eager();
    push(*s, tensor);
    return;
  }
  auto arg_len = fs.arguments().size();
  auto ret_len = fs.returns().size();
  auto args = last(*s, arg_len);
  if (ret_len == 1) {
    auto tensor = detail::make_tensor<LazyTensorImpl>(std::move(fs), args.vec());
    auto var = torch::autograd::make_variable(tensor);
    drop(*s, arg_len);
    push(*s, var);
  } else {
    drop(*s, arg_len);

    std::vector<IValue> ret_vec;
    std::vector<LazyTensorImpl*> siblings;

    for (auto i = 0; i < ret_len; ++i) {
      auto tensor = detail::make_tensor<LazyTensorImpl>(std::move(fs), args.vec(), i);
      auto lt = static_cast<LazyTensorImpl*>(tensor.unsafeGetTensorImpl());

      siblings.emplace_back(lt);
      ret_vec.emplace_back(tensor);
    }

    for (auto sib : siblings) {
      sib->setSiblings(siblings);
    }

    auto ret = c10::ivalue::Tuple::create(ret_vec);
    push(*s, ret);
  }
}

struct RegisterLazy {
  RegisterLazy() {
    static c10::RegistrationHandleRAII registration = c10::Dispatcher::singleton().registerBackendFallbackKernel(
        TensorTypeId::LazyTensorId,
        KernelFunction::makeFromBoxedFunction<&lazyRun>()
        );
    LazyTensorImpl::getResolver() = to_eager;
  }
};

static RegisterLazy r;

} // namespace at

#endif // C10_MOBILE
