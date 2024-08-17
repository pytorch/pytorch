#include <torch/csrc/jit/runtime/operator.h>

#include <ATen/ATen.h>
#include <ATen/core/interned_strings.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/edit_distance.h>

#include <queue>
#include <utility>
#include <vector>

namespace torch::jit {

namespace {
using OperatorMap =
    std::unordered_map<Symbol, std::vector<std::shared_ptr<Operator>>>;
struct OperatorRegistry {
 private:
  std::mutex lock;
  OperatorMap operators;
  // list of operators whose schema have not yet been parsed, and must
  // be registered before any call to lookup an operator
  std::vector<std::shared_ptr<Operator>> to_register;
  // Those two maps are used to implement lookupByLiteral, which is needed for
  // the n->match(...) calls. Basically, every function schema is assigned a
  // unique string you can use to match it. However, parsing those strings or
  // comparing and hashing them character by character would be very slow, so we
  // use a trick here! Every string literal in your program is guaranteed to
  // have static storage duration and so its address won't change at runtime.
  // This allows us to memoize answers for every pointer, which is done by the
  // operators_by_sig_literal map. Still, this map is initially empty, and so we
  // still need to do the complete string matching at the first time, which is
  // implemented by performing a lookup in the operators_by_sig map.
  std::unordered_map<std::string, std::shared_ptr<Operator>> operators_by_sig;
  std::unordered_map<const char*, std::shared_ptr<Operator>>
      operators_by_sig_literal;

  // Remember all registered operator names to check that they aren't
  // registered a second time. Registering an op multiple times is
  // fragile because it might depend on static initialization order
  // which one is picked at runtime.
#ifdef C10_MOBILE
  std::unordered_set<c10::OperatorName> registered_operator_names;
#endif

  // XXX - caller must be holding lock
  void registerPendingOperators() {
    for (const auto& op : to_register) {
      Symbol sym = Symbol::fromQualString(op->schema().name());
      operators[sym].push_back(op);
      operators_by_sig[canonicalSchemaString(op->schema())] = op;
    }
    to_register.clear();
  }

 public:
  void registerOperator(Operator&& op) {
    std::lock_guard<std::mutex> guard(lock);
#ifdef C10_MOBILE
    TORCH_INTERNAL_ASSERT(
        0 == registered_operator_names.count(op.schema().operator_name()),
        "Tried to register operator \"",
        op.schema(),
        "\" to JIT but the operator name was already registered before. Please add or change the overload name.");
    registered_operator_names.insert(op.schema().operator_name());
#endif
    to_register.push_back(std::make_shared<Operator>(std::move(op)));
  }

  void deregisterOperator(const FunctionSchema& schema) {
    Symbol sym = Symbol::fromQualString(schema.name());
    auto sig = canonicalSchemaString(schema);

    std::lock_guard<std::mutex> guard(lock);
#ifdef C10_MOBILE
    TORCH_INTERNAL_ASSERT(
        1 == registered_operator_names.count(schema.operator_name()),
        "Tried to remove operator ",
        schema,
        " from JIT but it wasn't found.");
    registered_operator_names.erase(schema.operator_name());
#endif
    // Try removing from pending operators list first
    auto pending_it = to_register.begin();
    while (pending_it != to_register.end() && (*pending_it)->schema() != schema)
      ++pending_it;

    if (pending_it != to_register.end()) {
      to_register.erase(pending_it);
      return;
    }

    // Remove operator from signature map
    auto sig_it = operators_by_sig.find(sig);
    if (sig_it == operators_by_sig.end()) {
      return;
    }

    operators_by_sig.erase(sig_it);

    // Remove operator from symbol map
    auto op_it = operators.find(sym);
    TORCH_CHECK(
        op_it != operators.end(),
        "operator with signature ",
        sig,
        " is missing from symbol registry");

    auto& op_vec = op_it->second;
    auto it = op_vec.begin();
    while (it != op_vec.end() && (*it)->schema() != schema)
      ++it;
    if (it != op_vec.end()) {
      op_vec.erase(it);
    }
    if (op_vec.empty()) {
      operators.erase(op_it);
    }
  }

  const std::shared_ptr<Operator>& lookupByLiteral(const char* name) {
    std::lock_guard<std::mutex> guard(lock);
    registerPendingOperators();
    auto it = operators_by_sig_literal.find(name);
    if (it == operators_by_sig_literal.end()) {
      auto op_ptr_it =
          operators_by_sig.find(canonicalSchemaString(parseSchema(name)));
      // Handy debugging code that dumps all operators we know about on mismatch
#if 0
      if (op_ptr_it == operators_by_sig.end()) {
        for (auto & entry : operators_by_sig) {
          std::cout << entry.first << std::endl;
        }
      }
#endif
      TORCH_CHECK(
          op_ptr_it != operators_by_sig.end(),
          "Couldn't find an operator for ",
          name,
          ". Do you have to update a set of hardcoded JIT ops?");
      it = operators_by_sig_literal.emplace_hint(it, name, op_ptr_it->second);
    }
    return it->second;
  }

  const std::vector<std::shared_ptr<Operator>>& getOperators(Symbol name) {
    std::lock_guard<std::mutex> guard(lock);
    registerPendingOperators();
    static std::vector<std::shared_ptr<Operator>> empty;
    auto it = operators.find(name);
    if (it != operators.end())
      return it->second;
    return empty;
  }

  std::vector<Symbol> findSimilarOperators(Symbol input_op) {
    std::lock_guard<std::mutex> guard(lock);
    registerPendingOperators();

    using EntryPair = std::pair<int64_t, Symbol>;
    auto cmp = [](const EntryPair& lhs, const EntryPair& rhs) {
      return lhs.first > rhs.first;
    };

    std::priority_queue<EntryPair, std::vector<EntryPair>, decltype(cmp)>
        rankings(cmp);
    static constexpr size_t MAX_EDIT_DIST = 2u;
    for (const auto& op : operators) {
      auto edit_dist = ComputeEditDistance(
          input_op.toQualString(), op.first.toQualString(), MAX_EDIT_DIST);
      if (edit_dist <= MAX_EDIT_DIST) {
        rankings.emplace(edit_dist, op.first);
      }
    }
    std::vector<Symbol> ret;
    while (!rankings.empty()) {
      ret.push_back(rankings.top().second);
      rankings.pop();
    }
    return ret;
  }

  const std::vector<std::shared_ptr<Operator>> getAllOperators() {
    std::lock_guard<std::mutex> guard(lock);
    registerPendingOperators();
    std::vector<std::shared_ptr<Operator>> values;
    values.clear();
    for (auto& kv : operators) {
      values.insert(values.end(), kv.second.begin(), kv.second.end());
    }
    return values;
  }
};

OperatorRegistry& getRegistry() {
  static OperatorRegistry r;
  return r;
}

bool printerHasSpecialCaseFor(Symbol sym) {
  using namespace at;
  // WARNING: by adding a value to this set, you are asserting
  // that you have also added special handling of this symbol to
  // the python_print.cpp. Not adding handling will cause import and export
  // of modules with this new operator to fail. This is only required
  // for operators without schema. Prefer registering your operator with
  // schema to editing this list here. These cases should only be things
  // that require special handling because they do not fit normal schema
  const static std::unordered_set<Symbol> handled = {
      prim::Constant,       prim::Uninitialized, prim::fork,
      prim::awaitable,      prim::ListConstruct, prim::DictConstruct,
      prim::ListUnpack,     prim::Print,         prim::PythonOp,
      prim::TupleConstruct, prim::TupleIndex,    prim::TupleSlice,
      prim::TupleUnpack,    prim::CreateObject,  prim::GetAttr,
      prim::SetAttr,        prim::CallFunction,  prim::isinstance,
      prim::unchecked_cast, prim::tolist,        prim::rpc_async,
      prim::rpc_sync,       prim::rpc_remote};

  // WARNING: by adding a value to this set, you are asserting that your
  // primitive is only ever added during optimization and does not need
  // to be correctly printed for export (a process that happens before
  // optimization passes run)
  const static std::unordered_set<Symbol> unneeded = {
      c10::onnx::Reshape, // only used in onnx
      c10::onnx::Shape, // only used in onnx
      prim::AutogradZero, // temporarily inserted by autograd
      prim::AutogradAnyNonZero, // temporarily inserted by autograd
      prim::AutogradAllNonZero, // temporarily inserted by autograd
      prim::AutogradAllZero, // temporarily inserted by autograd
      prim::AutogradAdd, // temporarily inserted by autograd
      prim::ConstantChunk, // optimization pass adds it
      prim::DifferentiableGraph, // optimization pass adds it,
      prim::FunctionalGraph, // optimization pass adds it,
      prim::ReductionSizes, // optimization pass (fuser) adds it
      prim::BroadcastSizes, // optimization pass (fuser) adds it
      prim::ChunkSizes, // optimization pass (fuser) adds it
      prim::Drop, // used in interpreter only
      prim::FusedConcat, // optimization pass adds it
      prim::FusionGroup, // optimization pass adds it
      prim::CudaFusionGroup, // optimization pass adds it
      prim::CudaFusionGuard, // optimization pass adds it
      prim::TensorExprGroup, // optimization pass adds it
      prim::TensorExprDynamicGroup, // optimization pass adds it
      prim::StaticSubgraph, // optimization pass adds it
      prim::ConstantONEDNNTensor, // optimization pass adds it
      prim::BroadcastONEDNNTensors, // optimization pass adds it
      prim::oneDNNFusionGroup, // optimization pass adds it
      prim::oneDNNFusionGuard, // optimization pass adds it
      prim::StaticRuntimeCopyOuts, // used in SR only
      prim::Load, // used in interpreter only
      prim::MMTreeReduce, // used as an optimization
      prim::MMBatchSide, // used as an optimization
      prim::Store, // used in interpreter only
      prim::profile, // used in interpreter only
      prim::profile_ivalue, // used in interpreter only
      prim::TypeCheck, // used in interpreter only
      prim::RequiresGradCheck, // used in interpreter only
      prim::FallbackGraph, // converted into prim::CallFunction

  };

  // These namespaces are required to have Python printers unless
  // otherwise noted in unneeded.
  const static std::unordered_set<Symbol> required_namespaces = {
      c10::namespaces::prim,
      c10::namespaces::aten,
      c10::namespaces::onnx,
  };

  return handled.count(sym) || unneeded.count(sym) ||
      !required_namespaces.count(sym.ns());
}

} // anonymous namespace

bool aliasAnalysisHasSpecialCaseFor(Symbol symbol) {
  using namespace at;
  // WARNING: by adding a case to this list, you are asserting that you have
  // added a case for the unschematized node in AliasDb::analyze
  const static std::unordered_set<Symbol> handled = {
      prim::If,
      prim::Loop,
      prim::FusionGroup,
      prim::CudaFusionGroup,
      prim::oneDNNFusionGroup,
      prim::DifferentiableGraph,
      prim::TensorExprGroup,
      prim::TensorExprDynamicGroup,
      prim::StaticSubgraph,
      prim::FunctionalGraph,
      prim::Constant,
      prim::Uninitialized,
      prim::DictConstruct,
      prim::ListConstruct,
      prim::TupleConstruct,
      prim::AutogradZero,
      prim::FusedConcat,
      prim::GradOf,
      prim::MMTreeReduce,
      prim::MMBatchSide,
      prim::BroadcastSizes,
      prim::ChunkSizes,
      prim::Closure,
      prim::TupleUnpack,
      prim::TupleIndex,
      prim::TupleSlice,
      prim::ListUnpack,
      prim::PythonOp,
      prim::ConstantChunk,
      prim::BroadcastingChunk,
      prim::ONEDNNGroup,
      prim::ConstantONEDNNTensor,
      prim::BroadcastONEDNNTensors,
      prim::fork,
      prim::awaitable,
      prim::awaitable_nowait,
      prim::awaitable_wait,
      prim::CreateObject,
      prim::AutogradAdd,
      prim::GetAttr,
      prim::SetAttr,
      prim::profile,
      prim::profile_ivalue,
      prim::TypeCheck,
      prim::RequiresGradCheck,
      prim::Print,
      prim::CallFunction,
      prim::CallMethod,
      aten::wait,
      prim::isinstance,
      prim::unchecked_cast,
      prim::tolist,
      prim::rpc_async,
      prim::rpc_sync,
      prim::rpc_remote,
      prim::Enter,
      prim::Exit,
      prim::FallbackGraph,
  };

  // Operators that should not be used by alias analysis
  const static std::unordered_set<Symbol> purposefully_not_handled = {
      prim::Load,
      prim::Store,
      prim::Drop,
      at::onnx::Reshape,
      at::onnx::Shape,
      prim::AutogradAdd,
  };

  return handled.count(symbol) || purposefully_not_handled.count(symbol);
}

void registerOperator(Operator&& op) {
  if (op.schema().is_varret()) {
    Symbol s = Symbol::fromQualString(op.schema().name());
    if (!printerHasSpecialCaseFor(s)) {
      AT_ERROR(
          "Missing special case in python printer for non-schematized"
          " operator ",
          op.schema().name(),
          ". File a bug to add a case for this operator.\n");
    }
    if (aliasAnalysisHasSpecialCaseFor(s) &&
        op.aliasAnalysisKind() == AliasAnalysisKind::CONSERVATIVE) {
      AT_ERROR(
          "Conflict in special casing in alias analysis for non-schematized"
          " operator ",
          op.schema().name(),
          ". File a bug to add a case for this operator.\n");
    }
    if (aliasAnalysisHasSpecialCaseFor(s) &&
        op.aliasAnalysisKind() == AliasAnalysisKind::FROM_SCHEMA) {
      AT_ERROR(
          "The operator ",
          op.schema().name(),
          " is special cased and cannot use explicit alias analysis.");
    }
  }
  getRegistry().registerOperator(std::move(op));
}

void deregisterOperator(const FunctionSchema& schema) {
  getRegistry().deregisterOperator(schema);
}

const std::vector<std::shared_ptr<Operator>> getAllOperators() {
  return getRegistry().getAllOperators();
}

const std::vector<std::shared_ptr<Operator>>& getAllOperatorsFor(Symbol name) {
  return getRegistry().getOperators(name);
}

std::vector<std::shared_ptr<Operator>> getAllSortedOperatorsFor(Symbol name) {
  const auto& unsortedOps = getAllOperatorsFor(name);
  // Depending on the order of registration, aten or jit ops may be
  // registered first. This sorting is helpful in cases where
  // deterministic (i.e. not dependent on build config) behavior is
  // desired; e.g. torch.ops.aten.* uses this function, and tries to
  // find the "first" op that matches input args. Without the sorting,
  // the "first" op may change depending on registration order.
  std::vector<std::shared_ptr<Operator>> sortedOps;
  sortedOps.reserve(unsortedOps.size());
  std::copy_if(
      unsortedOps.begin(),
      unsortedOps.end(),
      std::back_inserter(sortedOps),
      [](const std::shared_ptr<Operator>& op) { return op->isC10Op(); });
  std::copy_if(
      unsortedOps.begin(),
      unsortedOps.end(),
      std::back_inserter(sortedOps),
      [](const std::shared_ptr<Operator>& op) { return !op->isC10Op(); });
  return sortedOps;
}

std::shared_ptr<Operator> findOperatorFor(const c10::OperatorName& full_name) {
  for (const auto& op :
       getRegistry().getOperators(Symbol::fromQualString(full_name.name))) {
    if (op->schema().overload_name() == full_name.overload_name) {
      return op;
    }
  }
  return nullptr;
}

std::vector<Symbol> findSimilarOperators(Symbol input_op) {
  return getRegistry().findSimilarOperators(input_op);
}

std::shared_ptr<Operator> getOperatorForLiteral(const char* signature) {
  return getRegistry().lookupByLiteral(signature);
}

std::string canonicalSchemaString(const FunctionSchema& schema) {
  std::string out = schema.name();
  out.push_back('(');

  bool seen_kwarg_only = false;
  for (const auto i : c10::irange(schema.arguments().size())) {
    if (i > 0) {
      out += ", ";
    }
    if (schema.arguments()[i].kwarg_only() && !seen_kwarg_only) {
      out += "*, ";
      seen_kwarg_only = true;
    }
    const auto& arg = schema.arguments()[i];
    out += arg.type()->str();
    out.push_back(' ');
    out += arg.name();
  }

  out += ") -> ";
  if (schema.returns().size() == 1) {
    out += schema.returns().at(0).type()->str();
  } else if (schema.returns().size() > 1) {
    out.push_back('(');
    for (const auto i : c10::irange(schema.returns().size())) {
      if (i > 0) {
        out += ", ";
      }
      out += schema.returns()[i].type()->str();
    }
    out.push_back(')');
  }
  return out;
}

} // namespace torch::jit
