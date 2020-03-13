#include <ATen/ATen.h>
#include <ATen/core/alias_info.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/frontend/edit_distance.h>
#include <ATen/core/OpsAlreadyMovedToC10.h>

#include <queue>
#include <utility>
#include <vector>

namespace torch {
namespace jit {

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
    to_register.push_back(std::make_shared<Operator>(std::move(op)));
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
    for (auto & kv : operators) {
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
      prim::Constant,
      prim::Uninitialized,
      prim::fork,
      prim::ListConstruct,
      prim::DictConstruct,
      prim::ListUnpack,
      prim::Print,
      prim::PythonOp,
      prim::TupleConstruct,
      prim::TupleIndex,
      prim::TupleSlice,
      prim::TupleUnpack,
      prim::CreateObject,
      prim::GetAttr,
      prim::SetAttr,
      prim::CallFunction,
      prim::isinstance,
      prim::unchecked_cast,
      prim::tolist,
      prim::rpc_async,
  };

  // WARNING: by adding a value to this set, you are asserting that your
  // primitive is only ever added during optimization and does not need
  // to be correctly printed for export (a process that happens before
  // optimization passes run)
  const static std::unordered_set<Symbol> unneeded = {
      c10::onnx::Reshape, // only used in onnx
      c10::onnx::Shape, // only used in onnx
      prim::AutogradZero, // temporarily inserted by autograd
      prim::AutogradAnyNonZero, // temporarily inserted by autograd
      prim::AutogradAdd, // temporarily inserted by autograd
      prim::ConstantChunk, // optimization pass adds it
      prim::DifferentiableGraph, // optimization pass adds it
      prim::BroadcastSizes, // optimization pass (fuser) adds it
      prim::ChunkSizes, // optimization pass (fuser) adds it
      prim::Drop, // used in interpreter only
      prim::FusedConcat, // optimization pass adds it
      prim::FusionGroup, // optimization pass adds it
      prim::CudaFusionGroup, // optimization pass adds it
      prim::Load, // used in interpreter only
      prim::MMTreeReduce, // used as an optimization
      prim::MMBatchSide, // used as an optimization
      prim::Store, // used in interpreter only
      prim::profile, // used in interpreter only

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
      prim::DifferentiableGraph,
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
      prim::Function,
      prim::TupleUnpack,
      prim::TupleIndex,
      prim::TupleSlice,
      prim::ListUnpack,
      prim::PythonOp,
      prim::ConstantChunk,
      prim::BroadcastingChunk,
      prim::fork,
      prim::CreateObject,
      prim::AutogradAdd,
      prim::GetAttr,
      prim::SetAttr,
      prim::profile,
      prim::Print,
      prim::CallFunction,
      prim::CallMethod,
      aten::wait,
      prim::isinstance,
      prim::unchecked_cast,
      prim::tolist,
      prim::rpc_async,
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
    if (!aliasAnalysisHasSpecialCaseFor(s) &&
        op.aliasAnalysisKind() == AliasAnalysisKind::CONSERVATIVE) {
      AT_ERROR(
          "Missing special case in alias analysis for non-schematized"
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

const std::vector<std::shared_ptr<Operator>> getAllOperators() {
  return getRegistry().getAllOperators();
}

const std::vector<std::shared_ptr<Operator>>& getAllOperatorsFor(Symbol name) {
  return getRegistry().getOperators(name);
}

std::shared_ptr<Operator> findOperatorFor(const c10::OperatorName& full_name) {
  for (const auto& op : getRegistry().getOperators(Symbol::fromQualString(full_name.name))) {
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
  std::ostringstream out;

  out << schema.name();
  out << "(";

  bool seen_kwarg_only = false;
  for (size_t i = 0; i < schema.arguments().size(); ++i) {
    if (i > 0)
      out << ", ";
    if (schema.arguments()[i].kwarg_only() && !seen_kwarg_only) {
      out << "*, ";
      seen_kwarg_only = true;
    }
    const auto& arg = schema.arguments()[i];
    out << arg.type()->str() << " " << arg.name();
  }

  out << ") -> ";
  if (schema.returns().size() == 1) {
    out << schema.returns().at(0).type()->str();
  } else if (schema.returns().size() > 1) {
    out << "(";
    for (size_t i = 0; i < schema.returns().size(); ++i) {
      if (i > 0)
        out << ", ";
      out << schema.returns()[i].type()->str();
    }
    out << ")";
  }
  return out.str();
}

namespace details {
namespace {
  // custom ops don't do tracing/autograd in VariableType yet, we need to handle tracing here.
  // TODO This currently only handles tensors with requires_grad==False correctly.
  //      It should also handle autograd.
  Operation createOperationFromC10_withTracingHandledHere(const c10::OperatorHandle& op) {
    return [op](Stack& stack) {
      RECORD_FUNCTION(op.schema().name(), stack);
      const auto input_size = op.schema().arguments().size();
      const auto output_size = op.schema().returns().size();

      Node* node = nullptr;
      std::shared_ptr<jit::tracer::TracingState> tracer_state;

      // trace the input before unwrapping, otherwise we may lose
      // the input information
      if (jit::tracer::isTracing()) {
        tracer_state = jit::tracer::getTracingState();
        auto symbol = Symbol::fromQualString(op.schema().name());
        const auto& graph = tracer::getTracingState()->graph;
        node = graph->create(symbol, 0);
        tracer::recordSourceLocation(node);
        const auto& args = op.schema().arguments();
        int i = 0;
        for (auto iter = stack.end() - input_size; iter != stack.end();
             ++iter, ++i) {
          // TODO we need to refactor graph APIs (e.g., addInputs)
          // appropriately; after that, we can get rid of the giant if-else
          // block we will clean this tech debt together in the following PRs
          auto type = args[i].type();
          if (type->kind() == TypeKind::OptionalType) {
            if (iter->isNone()) {
              Value* none = graph->insertNode(graph->createNone())->output();
              node->addInput(none);
              continue;
            } else {
              type = type->expect<OptionalType>()->getElementType();
            }
          }
          if (type->isSubtypeOf(TensorType::get())) {
            AT_ASSERT(iter->isTensor());
            tracer::addInputs(node, args[i].name().c_str(), iter->toTensor());
          } else if (type->kind() == TypeKind::FloatType) {
            AT_ASSERT(iter->isDouble());
            tracer::addInputs(node, args[i].name().c_str(), iter->toDouble());
          } else if (type->kind() == TypeKind::IntType) {
            AT_ASSERT(iter->isInt());
            tracer::addInputs(node, args[i].name().c_str(), iter->toInt());
          } else if (type->kind() == TypeKind::BoolType) {
            AT_ASSERT(iter->isBool());
            tracer::addInputs(node, args[i].name().c_str(), iter->toBool());
          } else if (type->kind() == TypeKind::StringType) {
            AT_ASSERT(iter->isString());
            tracer::addInputs(
                node, args[i].name().c_str(), iter->toStringRef());
          } else if (type->kind() == TypeKind::NumberType) {
            tracer::addInputs(node, args[i].name().c_str(), iter->toScalar());
          } else if (type->kind() == TypeKind::ListType) {
            const auto& elem_type = type->expect<ListType>()->getElementType();
            if (elem_type->isSubtypeOf(TensorType::get())) {
              AT_ASSERT(iter->isTensorList());
              auto list = iter->toTensorVector();
              tracer::addInputs(node, args[i].name().c_str(), list);
            } else if (elem_type->kind() == TypeKind::FloatType) {
              AT_ASSERT(iter->isDoubleList());
              // NB: now, tracer doesn't support tracing double list. We add special
              // handling here, since in our case, we assume that all the doubles
              // in the list are constants
              auto value = iter->toDoubleVector();
              std::vector<Value*> info(value.size());
              for (size_t value_index = 0; value_index < value.size(); ++value_index) {
                info[value_index] = graph->insertConstant(value[value_index]);
                tracer::recordSourceLocation(info[value_index]->node());
              }
              node->addInput(
                  graph->insertNode(graph->createList(jit::FloatType::get(), info))->output());
            } else if (elem_type->kind() == TypeKind::IntType) {
              AT_ASSERT(iter->isIntList());
              tracer::addInputs(
                  node, args[i].name().c_str(), iter->toIntVector());
            } else if (elem_type->kind() == TypeKind::BoolType) {
              AT_ASSERT(iter->isBoolList());
              tracer::addInputs(
                  node, args[i].name().c_str(), iter->toBoolList().vec());
            } else {
              throw std::runtime_error(
                  "unsupported input list type: " + elem_type->str());
            }
          } else if (iter->isObject()) {
            tracer::addInputs(node, args[i].name().c_str(), iter->toObject());
          } else {
            throw std::runtime_error("unsupported input type: " + type->str());
          }
        }
        graph->insertNode(node);

        jit::tracer::setTracingState(nullptr);
      }

#ifdef USE_STATIC_DISPATCH
      {
        at::AutoNonVariableTypeMode non_var_type_mode(true);
        c10::Dispatcher::singleton().callBoxed(op, &stack);
      }
#else
      c10::Dispatcher::singleton().callBoxed(op, &stack);
#endif // USE_STATIC_DISPATCH

      if (tracer_state) {
        jit::tracer::setTracingState(std::move(tracer_state));
        int i = 0;
        for (auto iter = stack.end() - output_size; iter != stack.end();
             ++iter, ++i) {
          const auto& type = op.schema().returns()[i].type();
          if (type->isSubtypeOf(TensorType::get())) {
            AT_ASSERT(iter->isTensor());
            tracer::addOutput(node, iter->toTensor());
          } else if (type->kind() == TypeKind::ListType) {
            const auto& elem_type = type->expect<ListType>()->getElementType();
            if (elem_type->isSubtypeOf(TensorType::get())) {
              AT_ASSERT(iter->isTensorList());
              tracer::addOutput(node, iter->toTensorList());
            } else {
              throw std::runtime_error(
                  "unsupported ouptut list type: " + elem_type->str());
            }
          } else {
            throw std::runtime_error("unsupported output type: " + type->str());
          }
        }
      }

      return 0;
    };
  }

  Operation createOperationFromC10_withTracingNotHandledHere(const c10::OperatorHandle& op) {
    return [op](Stack& stack) {
      c10::Dispatcher::singleton().callBoxed(op, &stack);
      return 0;
    };
  }
}
Operation makeOperationForC10Operator(c10::OperatorHandle op) {
  if(at::is_aten_op_and_unboxing_is_already_handled_by_c10(op.schema().operator_name())) {
    // Those ops do tracing/autograd in VariableType, no need to handle it here
    return createOperationFromC10_withTracingNotHandledHere(op);
  } else if (at::is_aten_op_and_unboxing_is_not_handled_by_c10_yet(op.schema().operator_name())) {
    // register_aten_ops.cpp registers the jit unboxing wrapper for this op, no need to do anything here.
    TORCH_INTERNAL_ASSERT(false, "This is an op that still uses old style JIT registration. It should not have been registered as a c10 operator.");
  } else {
    // custom ops don't do tracing/autograd in VariableType yet, we need to handle tracing here.
    return createOperationFromC10_withTracingHandledHere(op);
  }
}

}

} // namespace jit
} // namespace torch
