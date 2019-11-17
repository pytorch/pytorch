#include <ATen/core/LazyTensor.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/autodiff.h>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/interpreter.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/script/function_schema_parser.h>

namespace at {

std::pair<torch::jit::Value*, std::vector<const LazyTensorImpl*>> getValue(
    const LazyTensorImpl* lt_,
    std::shared_ptr<torch::jit::Graph> g) {
  // Convert LazyTensorImpl to node in the graph, inserting before
  // the "dep" node.
  auto getNode = [&g](
                     const LazyTensorImpl* lt,
                     torch::jit::Node* dep) -> torch::jit::Node* {
    auto fs = *lt->schema_;
    auto s = Symbol::fromQualString(fs.name());
    auto* node = g->create(s, fs.returns().size());
    if (dep) {
      g->setInsertPoint(dep);
    }
    node = g->insertNode(node);
    return node;
  };

  // Setup output of the whole graph
  auto* root_node = getNode(lt_, nullptr);
  torch::jit::Value* output = root_node->outputs()[lt_->index_];
  std::vector<const LazyTensorImpl*> inputs;

  // Create frontier for resolving deps
  // The form is { IValue, Node (that depends on IValue) }
  std::vector<std::pair<c10::IValue, torch::jit::Node*>> frontier;
  for (const auto& input : lt_->inps_) {
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
        if (!lt->schema_) {
          auto* v = g->addInput();
          inputs.emplace_back(lt);
          dep_node->addInput(v);
        } else {
          auto* node = getNode(lt, dep_node);
          dep_node->addInput(node->outputs()[lt->index_]);
          for (const auto& input : lt->inps_) {
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

  return std::make_pair(output, inputs);
}

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
      // TODO other IValue checks
      if (a.val().isInt() && b.val().isInt() &&
          a.val().toInt() == b.val().toInt()) {
        return true;
      }
      if (a.val().isBool() && b.val().isBool() &&
          a.val().toBool() == b.val().toBool()) {
        return true;
      }
      if (a.val().isDouble() && b.val().isDouble() &&
          a.val().toDouble() == b.val().toDouble()) {
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
  // Construct a straight line repr of the LT
  //  a = ?.to_lazy()
  //  b = ?.to_lazy()
  //  c = a @ b
  //  d = c * 2
  // becomes
  //  *[0], @[0], 2, a, b
  // that is, all LTs with exprs are "ExprName[index]", constants are constant
  // and LTs with tensor bindings are LT*
  // We can theoretically do memory comparison on these

  std::vector<Item> record;
  std::vector<c10::IValue> frontier = lt_->inps_;
  std::vector<const LazyTensorImpl*> inputs;
  while (frontier.size()) {
    std::vector<c10::IValue> new_frontier;
    for (const auto& ival : frontier) {
      if (ival.isTensor()) {
        auto t = ival.toTensor();
        if (t.is_lazy()) {
          auto* lt = static_cast<LazyTensorImpl*>(t.unsafeGetTensorImpl());
          if (lt->schema_) {
            // make into Item, push to record
            record.emplace_back(
                std::make_pair(lt->schema_->name(), lt->index_));
            // maybe add to new_frontier
            new_frontier.reserve(new_frontier.size() + lt->inps_.size());
            new_frontier.insert(
                new_frontier.end(), lt->inps_.begin(), lt->inps_.end());
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
  torch::jit::Value* output;
  std::tie(output, inputs) = getValue(lt_, g);
  g->registerOutput(output);
  ge = std::make_shared<torch::jit::GraphExecutor>(g);
  graph_cache.put(std::move(record), ge);

  return std::make_pair(ge, inputs);
}

Tensor LazyTensorImpl::to_eager() const {
  if (!schema_) {
    auto t = inps_[0].toTensor();
    return t;
  }

  std::shared_ptr<torch::jit::GraphExecutor> ge;
  std::vector<const LazyTensorImpl*> inputs;
  std::tie(ge, inputs) = getGraph(this);
  torch::jit::Stack stack;
  for (auto& inp : inputs) {
    stack.push_back(inp->inps_[0]);
  }

  ge->run(stack);

  // We registered only the output we care about
  auto output_t = stack[0].toTensor();
  if (output_t.is_variable()) {
    return output_t;
  }
  auto output_var = torch::autograd::make_variable(output_t);
  output_t = output_var.tensor_data();
  TORCH_INTERNAL_ASSERT(output_t.is_variable());
  return output_t;
}

} // namespace at

namespace at {

using namespace torch::jit;
void lazyRun(const OperatorHandle& op, torch::jit::Stack* s) {
  const FunctionSchema& fs = op.schema();
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
    for (auto i = 0; i < ret_len; ++i) {
      auto tensor = detail::make_tensor<LazyTensorImpl>(std::move(fs), args.vec(), i);

      auto var = torch::autograd::make_variable(tensor);
      ret_vec.emplace_back(var);
    }
    auto ret = c10::ivalue::Tuple::create(ret_vec);
    push(*s, ret);
  }
}

struct RegisterLazy {
  RegisterLazy() {
    static c10::RegistrationHandleRAII wtf = c10::Dispatcher::singleton().registerBackendFallbackKernel(
        TensorTypeId::LazyTensorId,
        KernelFunction::makeFromBoxedFunction<&lazyRun>()
        );
  }
};

static RegisterLazy r;

namespace native {

CAFFE2_API at::Tensor to_lazy(at::Tensor const& self) {
  auto tensor = detail::make_tensor<LazyTensorImpl>(self);
  return torch::autograd::make_variable(tensor);
}

CAFFE2_API at::Tensor to_eager(at::Tensor const& self) {
  auto lt = static_cast<LazyTensorImpl*>(self.unsafeGetTensorImpl());
  auto tensor = lt->to_eager();
  if (!tensor.is_variable()) {
    auto var = torch::autograd::make_variable(tensor);
    tensor = var.tensor_data();
  }
  TORCH_INTERNAL_ASSERT(tensor.is_variable());
  return tensor;
}

} // namespace native
} // namespace at
