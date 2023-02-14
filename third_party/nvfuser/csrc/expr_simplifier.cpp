#include <expr_simplifier.h>

#include <ir_all_nodes.h>
#include <ir_builder.h>
#include <ir_cloner.h>
#include <ir_iostream.h>
#include <ir_utils.h>
#include <lower_magic_zero.h>
#include <utils.h>

#include <cmath>
#include <functional>
#include <list>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace debug_print {

// In order to print transformations from expr simplifier, use:
//   PYTORCH_NVFUSER_DUMP="expr_simplify"
// By default (i.e. no trigger specified), enabling the above debug dump option
// will trigger printing when the expression is transformed by at least one
// simplification pass (A simplification pass is a pass that is not a flatten
// or unflatten). If you want to print even if there is no simplification pass,
// applied, use the following:
//   PYTORCH_NVFUSER_DUMP="expr_simplify(assoc_comm::flatten)"
// If you want to trigger printing only on a specified set of passes, put the
// pass names as arguments of `expr_simplify`, for example:
//   PYTORCH_NVFUSER_DUMP="expr_simplify(eliminateTrivialComputation)"

constexpr const char* kFlattenName = "assoc_comm::flatten";
constexpr const char* kUnflattenName = "assoc_comm::unflatten";

struct Record {
  const char* name;
  Val* result;
};

class NoOpLogger {
 public:
  NoOpLogger(Val*) {}
  virtual ~NoOpLogger() {}
  virtual void record(const char*, Val*) {}
};

class Logger : public NoOpLogger {
  bool shouldPrint() {
    if (current_val_->sameAs(init_val_)) {
      return false;
    }
    const auto& triggers_arg =
        getDebugDumpArguments(DebugDumpOption::ExprSimplification);
    std::vector<std::regex> triggers;
    triggers.reserve(triggers_arg.size());
    for (const auto& t : triggers_arg) {
      triggers.emplace_back(t);
    }
    if (triggers.empty()) {
      for (auto r : record_) {
        if (r.name != kFlattenName && r.name != kUnflattenName) {
          return true;
        }
      }
      return false;
    }
    for (auto r : record_) {
      auto match = [r](const std::regex& trigger) -> bool {
        return std::regex_match(r.name, trigger);
      };
      if (std::find_if(triggers.begin(), triggers.end(), match) !=
          triggers.end()) {
        return true;
      }
    }
    return false;
  }

 public:
  Logger(Val* value)
      : NoOpLogger(value), init_val_(value), current_val_(value) {}

  virtual ~Logger() override {
    if (!shouldPrint()) {
      return;
    }

    auto str = [](Val* v) {
      std::stringstream ss;
      ss << ir_utils::varName(v) << " = " << v->toInlineString();
      return ss.str();
    };

    std::string header = "Simplifying expression:\n" + str(init_val_);
    std::cout << header << std::endl;
    for (auto r : record_) {
      std::cout << r.name << ":\n" << str(r.result) << std::endl;
    }
    std::cout << std::string(std::min<size_t>(header.size(), 80), '=')
              << std::endl;
  }

  virtual void record(const char* name, Val* value) override {
    if (value->sameAs(current_val_)) {
      return;
    } else {
      record_.emplace_back(Record{name, value});
      current_val_ = value;
    }
  }

 private:
  std::vector<Record> record_;
  Val* init_val_;
  Val* current_val_;
};

std::unique_ptr<debug_print::NoOpLogger> createLogger(Val* value) {
  if (isDebugDumpEnabled(DebugDumpOption::ExprSimplification)) {
    return std::make_unique<Logger>(value);
  } else {
    return std::make_unique<NoOpLogger>(value);
  }
}

} // namespace debug_print

namespace {

// An ordered mapping of variable -> VarInfo
class VarInfoMap {
 public:
  VarInfoMap() = default;

  VarInfoMap(const std::list<VarInfo>& variables) {
    var_info_map_.reserve(variables.size());
    var_order_.reserve(variables.size());
    set_.reserve(variables.size());
    for (const auto& info : variables) {
      var_order_.emplace_back(info.variable);
      set_.emplace(info.variable);
      var_info_map_[info.variable] = info;
    }
  }

  // Get the order of variables
  const std::vector<Val*>& order() const {
    return var_order_;
  }

  // Get the info of a variable
  const VarInfo& info(Val* var) const {
    return var_info_map_.at(var);
  }

  // Check if this mapping has information about the given variable
  bool has(Val* var) const {
    return set_.count(var);
  }

  // Get the set of variables that has information
  const std::unordered_set<Val*>& set() const {
    return set_;
  }

 private:
  std::unordered_map<Val*, VarInfo> var_info_map_;
  std::vector<Val*> var_order_;
  std::unordered_set<Val*> set_;
};

bool hasSimilarType(DataType t1, DataType t2) {
  if (t1 == t2) {
    return true;
  }
  if (isIntegralType(t1) && isIntegralType(t2)) {
    return true;
  }
  if (isFloatingPointType(t1) && isFloatingPointType(t2)) {
    return true;
  }
  if (isComplexType(t1) && isComplexType(t2)) {
    return true;
  }
  return false;
}

// If `value` is a constant scalar, then evaluate the value of that constant and
// return the evaluated value. Otherwise, returns `value` itself.
Val* foldConstants(Val* value) {
  if (value->isConst()) {
    return value;
  }
  if (value->isConstScalar()) {
    if (value->isIntegralScalar() && value->isA<Int>()) {
      return IrBuilder::newConstant(
          value->evaluateInt(), *value->getDataType());
    }
    if (value->isFloatingPointScalar() && value->isA<Double>()) {
      return IrBuilder::newConstant(
          value->evaluateDouble(), *value->getDataType());
    }
    if (value->isABool() && value->isA<Bool>()) {
      return IrBuilder::newConstant(
          value->evaluateBool(), *value->getDataType());
    }
    // TODO: support complex double
  }
  return value;
}

// Get the set of variables that `value` depends on. Items in `variables` are
// considered variables, and items not in `variables` are considered constant.
// For example, if value = a + b + c + d + 3, and `variables` is {a, b, e, f},
// then this function returns {a, b}. All tensors are considered variables.
std::unordered_set<Val*> getSubexprDependency(
    Val* value,
    const std::unordered_set<Val*>& variables) {
  if (value->isA<kir::TensorIndex>()) {
    return {value};
  }
  if (variables.count(value) > 0) {
    return {value};
  }
  auto def = value->definition();
  if (def == nullptr) {
    return {};
  }
  std::unordered_set<Val*> result;
  for (auto i : def->inputs()) {
    auto deps = getSubexprDependency(i, variables);
    result.insert(deps.begin(), deps.end());
  }
  return result;
}

// Apply `rule` to `value`, if `rule` returns a new `Val*` to replace `value`,
// then return that new `Val*`, otherwise recursively goes down to its inputs.
Val* recurseDown(Val* value, std::function<Val*(Val*)> rule) {
  if (value->isOneOf<TensorView, kir::TensorIndex>()) {
    return value;
  }
  auto def = value->definition();
  if (def == nullptr) {
    return value;
  }
  auto transformed = rule(value);
  if (transformed != value) {
    return transformed;
  }
  if (def != nullptr) {
    bool changed = false;
    std::vector<Val*> new_inputs;
    new_inputs.reserve(def->inputs().size());
    for (auto v : def->inputs()) {
      new_inputs.emplace_back(recurseDown(v, rule));
      if (new_inputs.back() != v) {
        changed = true;
      }
    }

    if (!changed) {
      return value;
    }

    Val* output = IrBuilder::newScalar(*value->getDataType());
    auto create_fn = def->newObjectFunc();
    create_fn(
        def->container(), std::move(new_inputs), {output}, def->attributes());
    return output;
  }
  return value;
}

RegisterType getRegisterType(Val* value, const VarInfoMap& var_info) {
  if (auto ns = dynamic_cast<NamedScalar*>(value)) {
    if (ns->getParallelIndex() == ParallelType::TIDx ||
        ns->getParallelIndex() == ParallelType::TIDy ||
        ns->getParallelIndex() == ParallelType::TIDz) {
      return RegisterType::GeneralPurpose;
    }
  }
  if (value->isConstScalar()) {
    return RegisterType::Immediate;
  }
  if (var_info.has(value) && var_info.info(value).is_compile_time_const) {
    return RegisterType::Immediate;
  }
  if (auto def = value->definition()) {
    RegisterType result = RegisterType::Unknown;
    for (auto inp : def->inputs()) {
      auto inp_rtype = getRegisterType(inp, var_info);
      if (inp_rtype == RegisterType::GeneralPurpose) {
        return RegisterType::GeneralPurpose;
      }
      if (result == RegisterType::Unknown) {
        result = inp_rtype;
        continue;
      }
      if (result == RegisterType::Uniform ||
          inp_rtype == RegisterType::Uniform) {
        result = RegisterType::Uniform;
      }
    }
    return result;
  }
  return RegisterType::Uniform;
}

} // namespace

RegisterType getRegisterType(Val* value) {
  return getRegisterType(value, {});
}

namespace assoc_comm {

// Note: [Reordering associative and commutative operators]
//
// For binary operators that is both associative and commutative, we can freely
// change the order of operands and add/remove parenthesis without changing the
// result. For example, + is both associative and commutative, so we have:
// a + b + c := (a + b) + c = a + (b + c) = (b + a) + c = b + (a + c) = ...
// For these operators, the most convenient way for handling them is to flatten
// them. For example, for the above a + b + c, all we need to know is we are
// adding these three variables together. We don't really care whether we are
// adding a and b first, or adding a and c first, or whether we are adding a to
// c or adding c to a. `FlattenedAssocCommOp` is the class that represents this
// flattened perspective.
//
// The reordering of associative and commutative operators is mostly useful for
// index hoisting. For example, if I have a loop structure and index:
//   FOR i1
//     FOR i2
//       FOR i3
//         index = ((i3 + i2) + i1) + 256
// There is no hoisting opportunity for this index in this loop structure.
// However, if I transform the index into index = ((256 + i1) + i2) + i3,
// then I can hoist the index as
//   FOR i1
//     i4 = (256 + i1)
//     FOR i2
//       i5 = i4 + i2
//       FOR i3
//         index = i5 + i3
// This minimizes the total number of computations.

bool isAssociativeAndCommutative(BinaryOpType type) {
  return type == BinaryOpType::Add || type == BinaryOpType::Mul ||
      type == BinaryOpType::And || type == BinaryOpType::Or ||
      type == BinaryOpType::Xor || type == BinaryOpType::Max ||
      type == BinaryOpType::Min;
}

// Identity `e` is a special number that, for all x:
// x (op) e = e (op) x = x
bool isIdentity(Val* v, BinaryOpType type) {
  if (v->isConstScalar()) {
    v = foldConstants(v);
  }
  if (!v->isConst()) {
    return false;
  }
  switch (type) {
    case BinaryOpType::Add:
      return v->getInt() == 0 || v->getDouble() == 0.0;
    case BinaryOpType::Mul:
      return v->getInt() == 1 || v->getDouble() == 1.0;
    case BinaryOpType::And:
      return v->getBool() == true;
    case BinaryOpType::Or:
      return v->getBool() == false;
    case BinaryOpType::Xor:
      return v->getBool() == false;
    default:
      return false;
  }
}

// Identity `b` is a special number that, for all x:
// x (op) b = b (op) x = b
bool isBlackhole(Val* v, BinaryOpType type) {
  if (v->isConstScalar()) {
    v = foldConstants(v);
  }
  if (!v->isConst()) {
    return false;
  }
  switch (type) {
    case BinaryOpType::Mul:
      return v->getInt() == 0;
    case BinaryOpType::And:
      return v->getBool() == false || v->getInt() == 0;
    case BinaryOpType::Or:
      return v->getBool() == true;
    default:
      return false;
  }
}

// The expression type that represents the flattened ops. For example, if I have
// out = a + b + 3 + c + 5, then I will have:
//   FlattenedAssocCommOp {
//     inputs: [a, b, 3, c, 5]
//     outputs: [out]
//   }
class FlattenedAssocCommOp : public Expr {
 public:
  using Expr::Expr;

  FlattenedAssocCommOp(
      IrBuilderPasskey passkey,
      BinaryOpType op,
      Val* out,
      std::vector<Val*> terms)
      : Expr(passkey) {
    TORCH_CHECK(
        isAssociativeAndCommutative(op),
        "Can only flatten associative and commutative ops");
    addAttribute(
        IrBuilder::create<Attribute<BinaryOpType>>(passkey.ir_container_, op));
    addOutput(out);
    for (auto v : terms) {
      TORCH_CHECK(
          hasSimilarType(dtype(), *v->getDataType()),
          "Input types should be similar, but got: ",
          dtype(),
          ", and ",
          *v->getDataType());
      addInput(v);
    }
  }

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    switch (getOpType()) {
      case BinaryOpType::Add:
        return "FlattenedAdd";
      case BinaryOpType::Mul:
        return "FlattenedMul";
      case BinaryOpType::And:
        return "FlattenedAnd";
      case BinaryOpType::Or:
        return "FlattenedOr";
      case BinaryOpType::Xor:
        return "FlattenedXor";
      case BinaryOpType::Max:
        return "FlattenedMax";
      case BinaryOpType::Min:
        return "FlattenedMin";
      default:
        TORCH_INTERNAL_ASSERT(false, "Unknown operator type ", getOpType());
    }
  }

  // FlattenedAssocCommOp is unordered, so we should have
  // FlattenedAdd(a, b)->sameAs(FlattenedAdd(b, a))
  bool sameAs(const Statement* other) const override {
    if (this == other) {
      return true;
    }
    if (!other->isA<FlattenedAssocCommOp>()) {
      return false;
    }
    auto other_fop = other->as<FlattenedAssocCommOp>();
    if (getOpType() != other_fop->getOpType()) {
      return false;
    }
    // check if we can establish a 1:1 mapping between inputs() and
    // other_fop->inputs()
    std::list<Val*> other_inputs(
        other_fop->inputs().begin(), other_fop->inputs().end());
    for (const auto inp : inputs()) {
      auto it =
          std::find_if(other_inputs.begin(), other_inputs.end(), [inp](Val* v) {
            return v->sameAs(inp);
          });
      if (it == other_inputs.end()) {
        return false;
      }
      other_inputs.erase(it);
    }
    return other_inputs.empty();
  }

  std::string toString(int indent_size = 0) const override {
    std::stringstream ss;
    indent(ss, indent_size) << getOpString() << "(";
    bool needs_comma = false;
    for (auto v : inputs()) {
      if (needs_comma) {
        ss << ", ";
      }
      ss << v->toString();
      needs_comma = true;
    }
    ss << ")\n";
    return ss.str();
  }

  std::string toInlineString(int = 0) const override {
    std::stringstream ss;
    ss << getOpString() << "(";
    bool needs_comma = false;
    for (auto v : inputs()) {
      if (needs_comma) {
        ss << ", ";
      }
      ss << v->toInlineString();
      needs_comma = true;
    }
    ss << ")";
    return ss.str();
  }

  DataType dtype() const {
    return *output(0)->getDataType();
  }

  BinaryOpType getOpType() const {
    return attribute(0)->as<Attribute<BinaryOpType>>()->value;
  }

  // Get a vector of inputs, sorted as the order given by `variables`. Note that
  // the sorting key is the rightmost variable that an input depends on. For
  // example, if I have two inputs.
  // v1 = a * c
  // v2 = b
  // and variables is [a, b, c], then v2 < v1 because the rightmost depending
  // variable of v2 is b, and the rightmost depending variable of v1 is c,
  // and b < c. So in this example, this function will return [v2, v1].
  // Tensors are always considered as variables and they are always considered
  // as the rightmost.
  std::vector<Val*> sortedInputs(const VarInfoMap& var_info) {
    std::vector<Val*> sorted_inputs(inputs().begin(), inputs().end());
    std::unordered_map<Val*, std::unordered_set<Val*>> dependency;
    dependency.reserve(sorted_inputs.size());
    for (auto v : sorted_inputs) {
      dependency[v] = getSubexprDependency(v, var_info.set());
    }
    auto compare = [&](Val* v1, Val* v2) {
      // Find all variables in var_info that v1 and v2 depends on. The input (v1
      // or v2) that exclusively has the right most variable in var_info.order()
      // will be to the right of the other input.
      bool v1_is_left_of_v2 = false;
      auto deps1 = dependency.at(v1);
      auto deps2 = dependency.at(v2);
      auto hasTensorIndex = [](const auto& deps) {
        return std::any_of(deps.begin(), deps.end(), [](auto val) {
          return val->template isA<kir::TensorIndex>();
        });
      };
      if (hasTensorIndex(deps2)) {
        return true;
      }
      if (hasTensorIndex(deps1)) {
        return false;
      }
      for (auto v : var_info.order()) {
        if (deps1.count(v) > 0 && deps2.count(v) == 0) {
          v1_is_left_of_v2 = false;
        } else if (deps2.count(v) > 0 && deps1.count(v) == 0) {
          v1_is_left_of_v2 = true;
        }
      }
      return v1_is_left_of_v2;
    };
    std::sort(sorted_inputs.begin(), sorted_inputs.end(), compare);
    return sorted_inputs;
  }

  bool isTrivial() const {
    return inputs().size() == 1;
  }

  std::vector<EvaluatorValue> evaluate(
      const std::vector<EvaluatorValue>& inputs) const override {
    using namespace EvaluatorValue_functions;
    std::vector<EvaluatorValue> inputs_ = inputs;
    EvaluatorValue result;
    result = inputs_.back();
    inputs_.pop_back();
    switch (getOpType()) {
      case BinaryOpType::Add:
        for (auto i : inputs_) {
          result += i;
        }
        break;
      case BinaryOpType::Mul:
        for (auto i : inputs_) {
          result *= i;
        }
        break;
      case BinaryOpType::And:
        for (auto i : inputs_) {
          result = result && i;
        }
        break;
      case BinaryOpType::Or:
        for (auto i : inputs_) {
          result = result || i;
        }
        break;
      case BinaryOpType::Xor:
        for (auto i : inputs_) {
          result = result ^ i;
        }
        break;
      case BinaryOpType::Min:
        for (auto i : inputs_) {
          result = min(result, i);
        }
        break;
      case BinaryOpType::Max:
        for (auto i : inputs_) {
          result = max(result, i);
        }
        break;
      default:
        TORCH_INTERNAL_ASSERT(
            "Unexpected operator type encountered"
            "in EvaluatorValue::evaluate: ",
            getOpType());
    }
    return {result};
  }
};

NVFUSER_DEFINE_CLONE_AND_CREATE(FlattenedAssocCommOp)

// Recursively convert expressions like AddOp(AddOp(a, b), AddOp(c, d)) into
// FlattenedAdd(a, b, c, d). This function recursively transforms the entire
// expression, so divOp(AddOp(AddOp(a, b), AddOp(c, d)), addOp(e, f)) will
// become divOp(FlattenAdd(a, b, c, d), FlattenAdd(e, f))
Val* flatten(Val* value);

Val* flattenRule(Val* value) {
  auto def = value->definition();
  if (def == nullptr) {
    return value;
  }
  if (isProtectedWithMagicZero(value)) {
    return value;
  }
  value = foldConstants(value);
  if (value->isConst()) {
    return value;
  }

  TORCH_INTERNAL_ASSERT(
      def->outputs().size() == 1,
      "Expressions with multiple output are not supported");

  BinaryOpType op = BinaryOpType::Atan2; // Initialize with an arbitrary
                                         // non-associative non-commutative op
  bool changed = false;
  if (auto bop = dynamic_cast<BinaryOp*>(def)) {
    op = bop->getBinaryOpType();
    changed = true;
  } else if (auto fop = dynamic_cast<FlattenedAssocCommOp*>(def)) {
    op = fop->getOpType();
  }

  if (isAssociativeAndCommutative(op)) {
    // Handle associative-and-commutative op:
    // Convert binary ops into flattened op, reflatten already flattened
    // FlattenedAssocCommOp
    std::vector<Val*> inputs;

    auto append_or_merge_inputs = [&](Val* operand) {
      auto fop = dynamic_cast<FlattenedAssocCommOp*>(operand->definition());
      if (fop != nullptr && fop->getOpType() == op &&
          hasSimilarType(fop->dtype(), *value->getDataType())) {
        inputs.insert(inputs.end(), fop->inputs().begin(), fop->inputs().end());
        changed = true;
      } else {
        inputs.emplace_back(operand);
      }
    };

    for (auto inp : def->inputs()) {
      auto flattened = flatten(inp);
      if (flattened != inp) {
        changed = true;
      }
      append_or_merge_inputs(flattened);
    }

    if (!changed) {
      return value;
    }

    if (inputs.size() == 1) {
      return inputs.at(0);
    }

    auto output = IrBuilder::newScalar(*value->getDataType());
    IrBuilder::create<FlattenedAssocCommOp>(op, output, std::move(inputs));
    return output;
  }

  return value;
}

Val* flatten(Val* value) {
  return recurseDown(value, flattenRule);
}

// Recursively convert expressions like FlattenedAdd(a, b, c, d) into
// AddOp(AddOp(AddOp(a, b), c), d))
Val* unflatten(Val* value, const VarInfoMap& var_info);

Val* unflattenRule(Val* value, const VarInfoMap& var_info) {
  auto def = value->definition();
  if (def == nullptr) {
    return value;
  }
  if (isProtectedWithMagicZero(value)) {
    return value;
  }

  TORCH_INTERNAL_ASSERT(
      def->outputs().size() == 1,
      "Expressions with multiple output are not supported");

  auto fop = dynamic_cast<FlattenedAssocCommOp*>(def);

  if (fop != nullptr) {
    // Handle flattened op:
    // Convert flattened op into original binary ops
    TORCH_INTERNAL_ASSERT(fop->inputs().size() >= 2);
    auto sorted_inputs = fop->sortedInputs(var_info);
    // We need to recursively unflatten all inputs, because we might have
    // nested flattened expressions like
    // FlattenedAdd(a, b, FlattenedMul(c, d, e))
    Val* lhs = unflatten(sorted_inputs.at(0), var_info);
    int64_t next = 1;
    while (next < (int64_t)sorted_inputs.size()) {
      auto rhs = unflatten(sorted_inputs.at(next), var_info);
      if (fop->getOpType() == BinaryOpType::Add) {
        // For binary add/sub op, we need to correctly handle the promotion rule
        // ptr + int -> ptr and ptr - int -> ptr, so we use IrBuilder::addExpr
        // and IrBuilder::subExpr here.
        // Also, convert a + (-b) to a - b here for better readibility on
        // generated code
        auto uop = dynamic_cast<UnaryOp*>(rhs->definition());
        if (uop != nullptr && uop->getUnaryOpType() == UnaryOpType::Neg) {
          lhs = IrBuilder::subExpr(lhs, uop->in());
        } else {
          lhs = IrBuilder::addExpr(lhs, rhs);
        }
      } else {
        auto output = IrBuilder::newScalar(*value->getDataType());
        IrBuilder::create<BinaryOp>(fop->getOpType(), output, lhs, rhs);
        lhs = output;
      }
      next++;
    }
    return lhs;
  }
  return value;
}

Val* unflatten(Val* value, const VarInfoMap& var_info) {
  return recurseDown(
      value, [&var_info](Val* val) { return unflattenRule(val, var_info); });
}

} // namespace assoc_comm

namespace {

using FOp = assoc_comm::FlattenedAssocCommOp;

FOp* toFlattenedAdd(Expr* expr) {
  auto fop = dynamic_cast<FOp*>(expr);
  if (!fop) {
    return nullptr;
  }
  if (fop->getOpType() == BinaryOpType::Add) {
    return fop;
  }
  return nullptr;
}

bool isFlattenedAdd(Val* x) {
  return toFlattenedAdd(x->definition()) != nullptr;
}

FOp* toFlattenedMul(Expr* expr) {
  auto fop = dynamic_cast<FOp*>(expr);
  if (!fop) {
    return nullptr;
  }
  if (fop->getOpType() == BinaryOpType::Mul) {
    return fop;
  }
  return nullptr;
}

bool isFlattenedMul(Val* x) {
  return toFlattenedMul(x->definition()) != nullptr;
}

BinaryOp* toDivModOp(Expr* expr) {
  auto bop = dynamic_cast<BinaryOp*>(expr);
  if (!bop) {
    return nullptr;
  }
  if (bop->getBinaryOpType() == BinaryOpType::Div ||
      bop->getBinaryOpType() == BinaryOpType::Mod) {
    // TODO: Add CeilDiv as well? Need mathematiclly prove its rules first
    return bop;
  }
  return nullptr;
}

// Classify terms of a FlattenedMul as (constant, symbolic), for example:
// a * 3 * b * 5 --> (15, {a, b})
// a * b --> (1, {a, b})
// 3 * 5 --> (15, {})
// If the given Val `x` is not a flattened mul, then return (1, {x})
std::pair<int64_t, std::list<Val*>> getConstAndSymbolicFactors(Val* x) {
  std::vector<Val*> factors;
  if (auto fop = toFlattenedMul(x->definition())) {
    factors = fop->inputs();
  } else {
    factors.emplace_back(x);
  }
  int64_t const_factor = 1;
  std::list<Val*> symbolic_factors;
  for (auto f : factors) {
    f = foldConstants(f);
    if (f->getInt().has_value()) {
      const_factor *= *f->getInt();
    } else {
      symbolic_factors.emplace_back(f);
    }
  }
  return {const_factor, symbolic_factors};
}

Val* productOfFactors(
    int64_t const_factor,
    std::vector<Val*> symbolic_factors,
    DataType dtype) {
  if (const_factor != 1) {
    symbolic_factors.emplace_back(IrBuilder::newConstant(const_factor, dtype));
  }
  if (symbolic_factors.size() == 1) {
    return symbolic_factors.at(0);
  }
  if (symbolic_factors.empty()) {
    return IrBuilder::newConstant(1, dtype);
  }
  auto output = IrBuilder::newScalar(dtype);
  IrBuilder::create<FOp>(
      BinaryOpType::Mul, output, std::move(symbolic_factors));
  return output;
}

} // namespace

namespace sym_algebra {

// Common utilities for symbolic algebra.

// Rewrite x in the form x = x1 * x2 * x3 * ...
Val* factorize(Val* x);

// Given that x = x1 * x2 * x3 * x4 * ..., y = x2 * x4 * ..., where x is a
// multiple of y, evaluate x/y as x/y = x1 * x3 * ... Both x and y should have
// already been factorized in the form of products, otherwise, this function
// might not be able to get the desired result. Returns nullptr if not
// divisible. Note that this function does symbolic term cancellation, it does
// not require y to be non-zero.
Val* divideFactorized(Val* x, Val* y) {
  auto x_factors = getConstAndSymbolicFactors(x);
  auto y_factors = getConstAndSymbolicFactors(y);

  int64_t quoient_const_factor;
  std::vector<Val*> quoient_symbolic_factors;

  if (x_factors.first % y_factors.first != 0) {
    // not divisible
    return nullptr;
  } else {
    quoient_const_factor = x_factors.first / y_factors.first;
  }

  for (auto yf : y_factors.second) {
    auto it = std::find_if(
        x_factors.second.begin(), x_factors.second.end(), [yf](Val* v) {
          return v->sameAs(yf);
        });
    if (it == x_factors.second.end()) {
      // not divisible
      return nullptr;
    }
    x_factors.second.erase(it);
  }
  quoient_symbolic_factors.insert(
      quoient_symbolic_factors.end(),
      x_factors.second.begin(),
      x_factors.second.end());
  return productOfFactors(
      quoient_const_factor,
      std::move(quoient_symbolic_factors),
      *x->getDataType());
}

// Symbolic gcd, for example: greatestCommonDivisor({6*a*b, 9*b*c}) -> 3*b
Val* greatestCommonDivisor(const std::vector<Val*>& inputs) {
  // The gcd of the constant part. Because gcd(0, a) = gcd(a, 0) = a, it is
  // great to use 0 as initial value because it does not need special handling.
  int64_t common_const_factor = 0;
  // The gcd of the symbolic part. nullptr serve as 0, empty vector serve as 1.
  std::unique_ptr<std::vector<Val*>> common_symbolic_factors = nullptr;

  for (auto inp : inputs) {
    auto factors = getConstAndSymbolicFactors(inp);
    common_const_factor = std::gcd(common_const_factor, factors.first);
    std::vector<Val*> new_common_symbolic_factors;
    if (common_symbolic_factors == nullptr) {
      // gcd(0, x) -> x
      new_common_symbolic_factors.insert(
          new_common_symbolic_factors.end(),
          factors.second.begin(),
          factors.second.end());
    } else {
      for (auto f : (*common_symbolic_factors)) {
        auto it = std::find_if(
            factors.second.begin(), factors.second.end(), [f](Val* v) {
              return v->sameAs(f);
            });
        if (it != factors.second.end()) {
          new_common_symbolic_factors.emplace_back(f);
          factors.second.erase(it);
        }
      }
    }
    common_symbolic_factors = std::make_unique<std::vector<Val*>>(
        std::move(new_common_symbolic_factors));
  }

  TORCH_INTERNAL_ASSERT(common_const_factor != 0);
  TORCH_INTERNAL_ASSERT(common_symbolic_factors != nullptr);
  return productOfFactors(
      common_const_factor,
      std::move(*common_symbolic_factors),
      *inputs[0]->getDataType());
}

namespace {

Val* factorizeFlattenedMul(Val* x) {
  auto fop = toFlattenedMul(x->definition());
  TORCH_INTERNAL_ASSERT(fop != nullptr);
  // Recursively factorize all its inputs, and combine their terms
  int64_t const_factor = 1;
  std::vector<Val*> symbolic_factors;
  bool changed = false;
  for (auto inp : fop->inputs()) {
    auto factorized_inp = factorize(inp);
    auto factors = getConstAndSymbolicFactors(factorized_inp);
    const_factor *= factors.first;
    symbolic_factors.insert(
        symbolic_factors.end(), factors.second.begin(), factors.second.end());
    if (factors.second != std::list<Val*>{inp}) {
      changed = true;
    }
  }

  if (!changed) {
    return x;
  }
  return productOfFactors(
      const_factor, std::move(symbolic_factors), *x->getDataType());
}

Val* factorizeFlattenedAdd(Val* x) {
  // Warning: This implementation can only factorize out common divisor. It can
  // not factorize FlattenedAdd(x * x, 2 * x, 1) as FlattenedMul(x + 1, x + 1).
  // But I believe factorizing out common divisor is sufficient for index
  // simplification.
  auto fop = toFlattenedAdd(x->definition());
  TORCH_INTERNAL_ASSERT(fop != nullptr);
  std::vector<Val*> factorized_inputs;
  for (auto inp : fop->inputs()) {
    factorized_inputs.emplace_back(factorize(inp));
  }
  // Find common factors
  auto gcd = greatestCommonDivisor(factorized_inputs);
  if (assoc_comm::isIdentity(gcd, BinaryOpType::Mul)) {
    return x;
  }
  // divide by common factors
  std::vector<Val*> quotient_inputs;
  for (auto inp : factorized_inputs) {
    auto quotient = divideFactorized(inp, gcd);
    TORCH_INTERNAL_ASSERT(quotient != nullptr);
    quotient_inputs.emplace_back(quotient);
  }
  auto quotient = IrBuilder::newScalar(*x->getDataType());
  IrBuilder::create<FOp>(
      BinaryOpType::Add, quotient, std::move(quotient_inputs));
  auto product = IrBuilder::newScalar(*x->getDataType());
  IrBuilder::create<FOp>(
      BinaryOpType::Mul, product, std::vector<Val*>{quotient, gcd});
  // Quotient might contain nested FlattenedAdd, for example, if we have:
  //   FlattenedAdd(a * FlattenedAdd(b, c), a * FlattenedAdd(d, e))
  // then the gcd will be a, and the quotient will be:
  //   FlattenedAdd(FlattenedAdd(b, c), FlattenedAdd(d, e))
  // So we need to reflatten to get rid of this nested FlattenedAdd.
  return assoc_comm::flatten(product);
}

} // namespace

// Rewrite x in the form x = x1 * x2 * x3 * ...
Val* factorize(Val* x) {
  if (x->isConstScalar()) {
    return foldConstants(x);
  }
  if (isProtectedWithMagicZero(x)) {
    return x;
  }
  if (isFlattenedMul(x)) {
    return factorizeFlattenedMul(x);
  }
  if (isFlattenedAdd(x)) {
    return factorizeFlattenedAdd(x);
  }
  // TODO: handle other operators, for example, rule M, O
  return x;
}

} // namespace sym_algebra

namespace prove {

// Prove properties of values. Note that functions in this namespace return
// boolean values. If true is returned, then this means the property is
// successfully proved. If false is returned, then this means that we are unable
// to prove the property. Be warned that a false being returned does not
// necessarily means the opposite of the property holds. For example, if you get
// isNonZero(x) == false, you shouldn't think that isZero(x) == true, because
// isNonZero(x) == false could mean:
// - x is actually non-zero, but we are not smart enough to prove it
// - x can be either zero or non-zero, it is just a symbolic number that depends
// - x is zero

bool isPositive(Val* value, const VarInfoMap& var_info);

bool isNonNegative(Val* value, const VarInfoMap& var_info) {
  value = foldConstants(value);
  if (value->getInt().has_value() && *value->getInt() >= 0) {
    return true;
  }
  if (value->getDouble().has_value() && *value->getDouble() >= 0.0) {
    return true;
  }
  if (isPositive(value, var_info)) {
    return true;
  }
  if (auto ns = dynamic_cast<NamedScalar*>(value)) {
    if (ns->getParallelDim().has_value() ||
        ns->getParallelIndex().has_value() || ns->isTensorSize() ||
        ns->isTensorStride()) {
      return true;
    }
  }
  value = maybeUnwrapMagicZero(value);
  if (var_info.has(value)) {
    return isNonNegative(var_info.info(value).start, var_info) &&
        isNonNegative(var_info.info(value).step, var_info);
  }
  if (auto fop = dynamic_cast<FOp*>(value->definition())) {
    auto op = fop->getOpType();
    if (op == BinaryOpType::Add || op == BinaryOpType::Mul) {
      for (auto inp : fop->inputs()) {
        if (!isNonNegative(inp, var_info)) {
          return false;
        }
      }
      return true;
    }
  } else if (auto bop = dynamic_cast<BinaryOp*>(value->definition())) {
    auto op = bop->getBinaryOpType();
    if (op == BinaryOpType::Mod || op == BinaryOpType::Div ||
        op == BinaryOpType::CeilDiv) {
      return isNonNegative(bop->lhs(), var_info) &&
          isPositive(bop->rhs(), var_info);
    }
  }
  return false;
}

bool isPositive(Val* value, const VarInfoMap& var_info) {
  value = foldConstants(value);
  if (value->getInt().has_value() && *value->getInt() > 0) {
    return true;
  }
  if (value->getDouble().has_value() && *value->getDouble() > 0.0) {
    return true;
  }
  if (auto ns = dynamic_cast<NamedScalar*>(value)) {
    // TODO: currently we are implicitly assuming tensor sizes to be non-zero.
    // For example, if I have T0[2, 0] and I do T0->merge(0), I get T0[2*0]. And
    // during indexing, we will generate index like: (i / 0, i % 0). Strictly
    // speaking, this is wrong, but we are not handling this edge case. We are
    // just hoping that we will be lucky and this will work. For expression
    // simplification, I am not considering this edge case either, because all
    // simplifications involving div and mod requires the divisor to be
    // non-zero, and I don't want this edge case to block the simplification of
    // normal cases.
    if (ns->getParallelDim().has_value() || ns->isTensorSize()) {
      return true;
    }
  }
  if (auto fop = dynamic_cast<FOp*>(value->definition())) {
    auto op = fop->getOpType();
    if (op == BinaryOpType::Add) {
      bool has_positive = false;
      for (auto inp : fop->inputs()) {
        if (isPositive(inp, var_info)) {
          has_positive = true;
        } else if (!isNonNegative(inp, var_info)) {
          return false;
        }
      }
      return has_positive;
    } else if (op == BinaryOpType::Mul) {
      for (auto inp : fop->inputs()) {
        if (!isPositive(inp, var_info)) {
          return false;
        }
      }
      return true;
    }
  }
  return false;
}

bool isZero(Val* value) {
  value = foldConstants(value);
  if (value->getBool() == false || value->isZeroInt() ||
      value->getDouble() == 0.0) {
    return true;
  }
  return false;
}

bool isNonZero(Val* value, const VarInfoMap& var_info) {
  value = foldConstants(value);
  if (value->getInt().has_value() && *value->getInt() != 0) {
    return true;
  }
  if (value->getDouble().has_value() && *value->getDouble() != 0.0) {
    return true;
  }
  if (isPositive(value, var_info)) {
    return true;
  }
  if (auto fop = toFlattenedMul(value->definition())) {
    for (auto inp : fop->inputs()) {
      if (!isNonZero(inp, var_info)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

// Tries to prove that x is a multiple of y, that is, there exist an integer `k`
// such that x = k*y
bool isMultipleOf(Val* x, Val* y) {
  auto lhs = sym_algebra::factorize(x);
  auto rhs = sym_algebra::factorize(y);
  return sym_algebra::divideFactorized(lhs, rhs) != nullptr;
}

bool hasCompatibleSign(Val* x, Val* y, const VarInfoMap& var_info) {
  return isNonNegative(x, var_info) && isNonNegative(y, var_info);
}

} // namespace prove

namespace rules {

// Do simplifications like:
// 1 * a -> a
// 0 * a -> 0
// true && a -> a
// false && a -> false
// true || a -> true
// false || a -> a
// a % 1 -> 0
// a / 1 -> a
// x - x -> 0
// b && b -> b (assuming no side effect on b)
// b || b -> b (assuming no side effect on b)
// -(-x) -> x
// !(!x) -> x
// ...
Val* eliminateTrivialComputation(Val* value, const VarInfoMap& var_info) {
  auto folded = foldConstants(value);
  if (folded != value) {
    return folded;
  }
  if (auto fop = dynamic_cast<FOp*>(value->definition())) {
    if (fop->isTrivial()) {
      return fop->input(0);
    }
    auto op = fop->getOpType();
    { // 0 * a -> 0, 1 * a -> a
      std::vector<Val*> new_inputs;
      Val* const_term = nullptr;
      bool changed = false;
      for (auto inp : fop->inputs()) {
        if (inp->isConstScalar()) {
          if (const_term == nullptr) {
            const_term = inp;
          } else {
            auto out = IrBuilder::newScalar(*const_term->getDataType());
            IrBuilder::create<BinaryOp>(op, out, const_term, inp);
            const_term = out;
            changed = true;
          }
        } else {
          new_inputs.emplace_back(inp);
        }
      }
      if (const_term != nullptr) {
        auto folded_const = foldConstants(const_term);
        if (folded_const != const_term) {
          changed = true;
          const_term = folded_const;
        }
        if (assoc_comm::isBlackhole(const_term, op)) {
          // 0 * a -> 0
          return const_term;
        }
        if (assoc_comm::isIdentity(const_term, op)) {
          // 1 * a -> a
          const_term = nullptr;
          changed = true;
        }
      }
      if (changed) {
        if (const_term != nullptr) {
          new_inputs.emplace_back(const_term);
        }
        if (new_inputs.size() == 1) {
          return new_inputs.at(0);
        }
        auto output = IrBuilder::newScalar(*value->getDataType());
        IrBuilder::create<FOp>(op, output, std::move(new_inputs));
        return output;
      }
    }
    { // b && b -> b, b || b -> b
      if (op == BinaryOpType::And || op == BinaryOpType::Or) {
        std::vector<Val*> dedup_input;
        for (auto v : fop->inputs()) {
          bool found_dup = false;
          for (auto v2 : dedup_input) {
            if (v->sameAs(v2)) {
              found_dup = true;
              break;
            }
          }
          if (!found_dup) {
            dedup_input.emplace_back(v);
          }
        }
        if (dedup_input.size() < fop->inputs().size()) {
          if (dedup_input.size() == 1) {
            return dedup_input.at(0);
          } else {
            auto output = IrBuilder::newScalar(*value->getDataType());
            IrBuilder::create<FOp>(op, output, std::move(dedup_input));
            return output;
          }
        }
      }
    }
  } else if (auto bop = dynamic_cast<BinaryOp*>(value->definition())) {
    if (bop->getBinaryOpType() == BinaryOpType::Mod) {
      // a % 1 -> 0
      auto rhs = foldConstants(bop->rhs());
      if (rhs->isOneInt()) {
        return IrBuilder::newConstant(0, *value->getDataType());
      }
    } else if (
        bop->getBinaryOpType() == BinaryOpType::Div ||
        bop->getBinaryOpType() == BinaryOpType::CeilDiv) {
      // a / 1 -> a
      // 0 / a -> 0
      auto lhs = foldConstants(bop->lhs());
      auto rhs = foldConstants(bop->rhs());
      bool divisor_is_1 = (rhs->getInt() == 1 || rhs->getDouble() == 1.0);
      if (divisor_is_1 ||
          (prove::isNonZero(rhs, var_info) && lhs->getInt() == 0)) {
        return lhs;
      }
    } else if (bop->getBinaryOpType() == BinaryOpType::Sub) {
      auto lhs = foldConstants(bop->lhs());
      auto rhs = foldConstants(bop->rhs());
      if (lhs->sameAs(rhs)) {
        return IrBuilder::newConstant(0, *value->getDataType());
      }
    }
  } else if (auto uop = dynamic_cast<UnaryOp*>(value->definition())) {
    // -(-x) -> x, !(!x) -> x
    auto optype = uop->getUnaryOpType();
    if (optype == UnaryOpType::Neg || optype == UnaryOpType::Not) {
      auto uop_in = dynamic_cast<UnaryOp*>(uop->in()->definition());
      if (uop_in != nullptr) {
        auto optype_in = uop_in->getUnaryOpType();
        if (optype == optype_in) {
          return uop_in->in();
        }
      }
    }
  }
  return value;
}

// If x can be proved to be non-negative, then replace x >= 0 as true, replace
// x < 0 as false
// If x can be proved to be positive, then replace x >= 0 and x > 0 as true,
// replace x <= 0 and x < 0 as false
// If x can be proved to be nonzero, then replace x != 0 as true, replace x == 0
// as false
// if x->sameAs(y), then replace x == y as true, replace x != y as false
Val* eliminateTrivialPredicate(Val* value, const VarInfoMap& var_info) {
  if (!value->isABool()) {
    return value;
  }
  auto bop = dynamic_cast<BinaryOp*>(value->definition());
  if (!bop) {
    return value;
  }
  auto op = bop->getBinaryOpType();
  if (bop->lhs()->sameAs(bop->rhs())) {
    if (op == BinaryOpType::Eq) {
      return value->fusion()->trueVal();
    } else if (op == BinaryOpType::NE) {
      return value->fusion()->falseVal();
    }
  }
  if (prove::isZero(bop->rhs())) {
    if (op == BinaryOpType::GE && prove::isNonNegative(bop->lhs(), var_info)) {
      return value->fusion()->trueVal();
    } else if (
        op == BinaryOpType::GT && prove::isPositive(bop->lhs(), var_info)) {
      return value->fusion()->trueVal();
    } else if (
        op == BinaryOpType::NE && prove::isNonZero(bop->lhs(), var_info)) {
      return value->fusion()->trueVal();
    } else if (
        op == BinaryOpType::LT && prove::isNonNegative(bop->lhs(), var_info)) {
      return value->fusion()->falseVal();
    } else if (
        op == BinaryOpType::LE && prove::isPositive(bop->lhs(), var_info)) {
      return value->fusion()->falseVal();
    } else if (
        op == BinaryOpType::Eq && prove::isNonZero(bop->lhs(), var_info)) {
      return value->fusion()->falseVal();
    }
  } else if (prove::isZero(bop->lhs())) {
    if (op == BinaryOpType::LE && prove::isNonNegative(bop->rhs(), var_info)) {
      return value->fusion()->trueVal();
    } else if (
        op == BinaryOpType::LT && prove::isPositive(bop->rhs(), var_info)) {
      return value->fusion()->trueVal();
    } else if (
        op == BinaryOpType::NE && prove::isNonZero(bop->rhs(), var_info)) {
      return value->fusion()->trueVal();
    } else if (
        op == BinaryOpType::GT && prove::isNonNegative(bop->rhs(), var_info)) {
      return value->fusion()->falseVal();
    } else if (
        op == BinaryOpType::GE && prove::isPositive(bop->rhs(), var_info)) {
      return value->fusion()->falseVal();
    } else if (
        op == BinaryOpType::Eq && prove::isNonZero(bop->rhs(), var_info)) {
      return value->fusion()->falseVal();
    }
  }
  return value;
}

// Apply rule L to replace x % y with 0 if x can be proved to be a multiple of y
// Also, according to rule M, if x can be factorized as x = k * y, then x / y
// can be simplified as x / y = (k * y) / y = k * (y / y) = k
Val* simplifyDivisibleDivMod(Val* value, const VarInfoMap& var_info) {
  auto bop = dynamic_cast<BinaryOp*>(value->definition());
  if (!bop) {
    return value;
  }
  if (!prove::isNonZero(bop->rhs(), var_info)) {
    return value;
  }
  if (bop->getBinaryOpType() == BinaryOpType::Mod) {
    if (prove::isMultipleOf(bop->lhs(), bop->rhs())) {
      return IrBuilder::newConstant(0, *value->getDataType());
    }
  } else if (bop->getBinaryOpType() == BinaryOpType::Div) {
    auto lhs = sym_algebra::factorize(bop->lhs());
    auto rhs = sym_algebra::factorize(bop->rhs());
    auto quotient = sym_algebra::divideFactorized(lhs, rhs);
    if (quotient != nullptr) {
      return quotient;
    }
  }
  return value;
}

// Simplify div and mod by canceling common terms
//
// For div, use rule N): a / (b * c) = (a / b) / c to simplify division:
// Let y = gcd(x, y) * y' and x = gcd(x, y) * x'
// then we can simplify x / y as:
// x / y = x / (gcd(x, y) * y') = (x / gcd(x, y)) / y' = x' / y'
//
// For mod, use rule
// O) If d divides a and b, then a % b = ((a / d) % (b / d)) * d
// Let y = gcd(x, y) * y' and x = gcd(x, y) * x'
// If gcd is nonzero, then we can simplify x % y as:
// x' % y' * gcd(x, y)
Val* cancelDivMod(Val* value, const VarInfoMap& var_info) {
  auto divmod = toDivModOp(value->definition());
  if (!divmod) {
    return value;
  }
  auto op = divmod->getBinaryOpType();
  if (op != BinaryOpType::Div && op != BinaryOpType::Mod) {
    return value;
  }
  auto lhs = sym_algebra::factorize(divmod->lhs());
  auto rhs = sym_algebra::factorize(divmod->rhs());
  auto gcd = sym_algebra::greatestCommonDivisor({lhs, rhs});
  if (gcd->isOneInt() || gcd->getDouble() == 1.0 ||
      !prove::isNonZero(gcd, var_info)) {
    return value;
  }
  auto numerator = sym_algebra::divideFactorized(lhs, gcd);
  auto denominator = sym_algebra::divideFactorized(rhs, gcd);
  if (op == BinaryOpType::Div) {
    return IrBuilder::divExpr(numerator, denominator);
  } else {
    TORCH_INTERNAL_ASSERT(op == BinaryOpType::Mod);
    return assoc_comm::flatten(
        IrBuilder::mulExpr(IrBuilder::modExpr(numerator, denominator), gcd));
  }
}

// Use the following rule to simplify div and mod:
// J) Distributivity of % over +:
//    If compatible_sign(a, b), then (a + b) % c = (a % c + b % c) % c
// Q) If compatible_sign(a, b) and -|c| < a % c + b % c < |c|, then
//    (a+b)/c = a/c + b/c
// In this pass we distribute div and mod for a special case:
// If compatible_sign(a, b), and a is a multiple of c, then:
//  (a+b)/c = a/c + b/c
//  (a + b) % c = b % c
Val* distributeDivisibleDivMod(Val* value, const VarInfoMap& var_info) {
  auto divmod = toDivModOp(value->definition());
  if (!divmod) {
    return value;
  }
  auto lhs = divmod->lhs();
  auto rhs = divmod->rhs();
  if (!prove::isNonZero(rhs, var_info)) {
    return value;
  }
  auto fop = toFlattenedAdd(lhs->definition());
  if (!fop) {
    return value;
  }
  for (auto i : c10::irange(fop->inputs().size())) {
    Val* divisible_term = fop->input(i);
    if (!prove::isMultipleOf(divisible_term, rhs)) {
      continue;
    }
    std::vector<Val*> other_terms;
    other_terms.reserve(fop->inputs().size() - 1);
    for (auto j : c10::irange(fop->inputs().size())) {
      if (j == i) {
        continue;
      }
      other_terms.emplace_back(fop->input(j));
    }
    Val* sum_of_other_terms = nullptr;
    if (other_terms.size() == 1) {
      sum_of_other_terms = other_terms.at(0);
    } else {
      sum_of_other_terms = IrBuilder::newScalar(*value->getDataType());
      IrBuilder::create<FOp>(
          BinaryOpType::Add, sum_of_other_terms, std::move(other_terms));
    }
    if (prove::hasCompatibleSign(
            divisible_term, sum_of_other_terms, var_info)) {
      std::vector<Val*> new_inputs;
      auto term1 = IrBuilder::newScalar(*value->getDataType());
      IrBuilder::create<BinaryOp>(
          divmod->getBinaryOpType(), term1, divisible_term, rhs);
      new_inputs.emplace_back(simplifyDivisibleDivMod(term1, var_info));
      new_inputs.emplace_back(IrBuilder::newScalar(*value->getDataType()));
      IrBuilder::create<BinaryOp>(
          divmod->getBinaryOpType(), new_inputs[1], sum_of_other_terms, rhs);
      auto output = IrBuilder::newScalar(*value->getDataType());
      IrBuilder::create<FOp>(BinaryOpType::Add, output, std::move(new_inputs));
      return output;
    }
  }
  return value;
}

// a * (b + c) -> a * b + a * c
Val* distributeMul(Val* value, const VarInfoMap& var_info) {
  auto fop = toFlattenedMul(value->definition());
  if (!fop) {
    return value;
  }
  Val* flattened_add = nullptr;
  std::vector<Val*> other_terms;
  for (auto inp : fop->inputs()) {
    if (flattened_add == nullptr && isFlattenedAdd(inp)) {
      flattened_add = inp;
    } else {
      other_terms.emplace_back(inp);
    }
  }
  if (flattened_add == nullptr) {
    return value;
  }
  auto fadd_op = toFlattenedAdd(flattened_add->definition());
  std::vector<Val*> add_terms;
  for (auto inp : fadd_op->inputs()) {
    std::vector<Val*> inputs = other_terms;
    inputs.emplace_back(inp);
    add_terms.emplace_back(IrBuilder::newScalar(*value->getDataType()));
    IrBuilder::create<FOp>(
        BinaryOpType::Mul, add_terms.back(), std::move(inputs));
  }
  auto output = IrBuilder::newScalar(*value->getDataType());
  IrBuilder::create<FOp>(BinaryOpType::Add, output, std::move(add_terms));
  return output;
}

// This pass optimizes register usage of predicate evaluation, especially for
// unrolled for loop.
// For example, if I have an unrolled for loop like below:
//   base = ...
//   for i in 4
//     for j in 4
//       if base + i* stride0 + j*stride1 < T.size:
//         expr
// Then the above for loop uses 4 * 4 = 16 general purposed registers to
// evaluate the lhs of the predicate.
// However, if I transform the predicate as:
//   base = ...
//   lhs = base - T.size
//   for i in 4
//     for j in 4
//       if lhs < -i* stride0 - j*stride1:
//         expr
// Then it only takes 1 general purposed register for lhs, and the rhs is a
// compile time constant for nvRTC, so it can use the immediate fields of the
// instruction. This optimization results in a great save on registers.
// See also: https://github.com/csarofeen/pytorch/pull/1976
Val* reducePredicateRegisterUsage(Val* value, const VarInfoMap& var_info) {
  auto bop = dynamic_cast<BinaryOp*>(value->definition());
  if (!value->isABool() || bop == nullptr) {
    return value;
  }
  auto op_type = bop->getBinaryOpType();
  if (!isLogicalOp(op_type)) {
    return value;
  }
  auto ltype = *bop->lhs()->getDataType();
  auto rtype = *bop->rhs()->getDataType();
  if (!hasSimilarType(ltype, rtype)) {
    return value;
  }
  // Redistribute terms. We always put general purposed registers to the
  // left, and immediates to the right. If there is no immediates, then
  // we put uniform registers to the right, otherwise we put uniform
  // registers to the left.
  bool moved = false;
  std::vector<Val*> new_lhs;
  std::vector<Val*> new_rhs;
  std::vector<Val*> lhs_uniform;
  std::vector<Val*> rhs_uniform;
  auto neg = [&](Val* x) {
    moved = true;
    return IrBuilder::negExpr(x);
  };
  // Redistribute lhs_terms into new_lhs, new_rhs, and lhs_uniform, can only
  // be called once.
  auto redist_lhs = [&](const std::vector<Val*>& lhs_terms) {
    for (auto inp : lhs_terms) {
      if (prove::isZero(inp)) {
        continue;
      }
      switch (getRegisterType(inp, var_info)) {
        case RegisterType::GeneralPurpose:
          new_lhs.emplace_back(inp);
          break;
        case RegisterType::Uniform:
          lhs_uniform.emplace_back(inp);
          break;
        case RegisterType::Immediate:
          new_rhs.emplace_back(neg(inp));
          break;
        default:
          TORCH_INTERNAL_ASSERT(false, "Unknown register type");
      }
    }
  };
  // Redistribute rhs_terms into new_lhs, new_rhs, and rhs_uniform, can only
  // be called once.
  auto redist_rhs = [&](const std::vector<Val*>& rhs_terms) {
    for (auto inp : rhs_terms) {
      if (prove::isZero(inp)) {
        continue;
      }
      switch (getRegisterType(inp, var_info)) {
        case RegisterType::GeneralPurpose:
          new_lhs.emplace_back(neg(inp));
          break;
        case RegisterType::Uniform:
          rhs_uniform.emplace_back(inp);
          break;
        case RegisterType::Immediate:
          new_rhs.emplace_back(inp);
          break;
        default:
          TORCH_INTERNAL_ASSERT(false, "Unknown register type");
      }
    }
  };
  if (auto lhs_fop = toFlattenedAdd(bop->lhs()->definition())) {
    redist_lhs(lhs_fop->inputs());
  } else {
    redist_lhs({bop->lhs()});
  }
  if (auto rhs_fop = toFlattenedAdd(bop->rhs()->definition())) {
    redist_rhs(rhs_fop->inputs());
  } else {
    redist_rhs({bop->rhs()});
  }
  if (new_rhs.empty()) {
    new_rhs = rhs_uniform;
    for (auto inp : lhs_uniform) {
      new_rhs.emplace_back(neg(inp));
    }
  } else {
    new_lhs.insert(new_lhs.end(), lhs_uniform.begin(), lhs_uniform.end());
    for (auto inp : rhs_uniform) {
      new_lhs.emplace_back(neg(inp));
    }
  }
  if (!moved) {
    return value;
  }
  Val* lhs = nullptr;
  Val* rhs = nullptr;
  if (new_lhs.size() == 0) {
    lhs = IrBuilder::newConstant(0, ltype);
  } else if (new_lhs.size() == 1) {
    lhs = new_lhs.at(0);
  } else {
    lhs = IrBuilder::newScalar(ltype);
    IrBuilder::create<FOp>(BinaryOpType::Add, lhs, std::move(new_lhs));
  }
  if (new_rhs.size() == 0) {
    rhs = IrBuilder::newConstant(0, rtype);
  } else if (new_rhs.size() == 1) {
    rhs = new_rhs.at(0);
  } else {
    rhs = IrBuilder::newScalar(rtype);
    IrBuilder::create<FOp>(BinaryOpType::Add, rhs, std::move(new_rhs));
  }
  auto output = IrBuilder::newScalar(DataType::Bool);
  IrBuilder::create<BinaryOp>(op_type, output, lhs, rhs);
  return output;
}

// Merge values using the fundamental division-with-remainder property:
// a = a / b * b + a % b
// This pass do merges of the following patterns:
// Pattern 1: a / b * b + a % b -> a
// Pattern 2: a / b * (b*c) + a % b * c -> a * c
Val* fundamentalDivisionWithRemainderProperty(
    Val* value,
    const VarInfoMap& var_info) {
  auto fadd = toFlattenedAdd(value->definition());
  if (!fadd) {
    return value;
  }
  // Get all patterns like: (a op b) * c, for example, if I have
  // (a / b) * (e / f), the this funciton will return:
  // {(a, b, e/f), (e, f, a/b)}
  auto get_a_op_b_mul_c = [](BinaryOpType op, Val* x) {
    std::vector<std::tuple<Val*, Val*, Val*>> result;
    auto fmul = toFlattenedMul(x->definition());
    if (fmul == nullptr) {
      return result;
    }
    for (auto j : c10::irange(fmul->inputs().size())) {
      auto vmul = fmul->input(j);
      if (!isIntegralType(*vmul->getDataType())) {
        continue;
      }
      auto bop = dynamic_cast<BinaryOp*>(vmul->definition());
      if (bop == nullptr) {
        continue;
      }
      if (bop->getBinaryOpType() == op) {
        auto a = bop->lhs();
        auto b = bop->rhs();
        std::vector<Val*> other_terms;
        for (auto k : c10::irange(fmul->inputs().size())) {
          if (j == k) {
            continue;
          }
          other_terms.emplace_back(fmul->input(k));
        }
        Val* c;
        if (other_terms.size() == 0) {
          continue;
        } else if (other_terms.size() == 1) {
          c = other_terms.at(0);
        } else {
          c = IrBuilder::newScalar(*vmul->getDataType());
          IrBuilder::create<FOp>(BinaryOpType::Mul, c, std::move(other_terms));
        }
        if (!isIntegralType(*a->getDataType()) ||
            !isIntegralType(*b->getDataType()) ||
            !isIntegralType(*c->getDataType())) {
          continue;
        }
        result.push_back(std::make_tuple(a, b, c));
      }
    }
    return result;
  };
  // Find a / b * b or a / b * (b*c)
  std::vector<std::tuple<size_t, Val*, Val*, Val*>> divmuls;
  for (auto i : c10::irange(fadd->inputs().size())) {
    auto vadd = fadd->input(i);
    if (!isIntegralType(*vadd->getDataType())) {
      continue;
    }
    for (auto& [a, b, bc] : get_a_op_b_mul_c(BinaryOpType::Div, vadd)) {
      divmuls.push_back(std::make_tuple(i, a, b, bc));
    }
  }
  // Find a % b or a % b * c
  std::vector<std::tuple<size_t, Val*, Val*, Val*>> modmuls;
  for (auto i : c10::irange(fadd->inputs().size())) {
    auto vadd = fadd->input(i);
    if (!isIntegralType(*vadd->getDataType())) {
      continue;
    }
    auto bop = dynamic_cast<BinaryOp*>(vadd->definition());
    if (bop != nullptr && bop->getBinaryOpType() == BinaryOpType::Mod) {
      if (!isIntegralType(*bop->lhs()->getDataType()) ||
          !isIntegralType(*bop->rhs()->getDataType())) {
        continue;
      }
      modmuls.push_back(std::make_tuple(
          i,
          bop->lhs(),
          bop->rhs(),
          IrBuilder::newConstant(1, *vadd->getDataType())));
    }
    for (auto& [a, b, c] : get_a_op_b_mul_c(BinaryOpType::Mod, vadd)) {
      modmuls.push_back(std::make_tuple(i, a, b, c));
    }
  }
  // Find matching divmul and modmul
  for (auto& [i, a1, b1, bc] : divmuls) {
    for (auto& [j, a2, b2, c] : modmuls) {
      if (i == j) {
        continue;
      }
      if (!a1->sameAs(a2) || !b1->sameAs(b2)) {
        continue;
      }
      if (!prove::isNonZero(b1, var_info)) {
        continue;
      }
      auto factorized_b = sym_algebra::factorize(b1);
      auto factorized_bc = sym_algebra::factorize(bc);
      auto quotient =
          sym_algebra::divideFactorized(factorized_bc, factorized_b);
      if (quotient != nullptr && quotient->sameAs(c)) {
        // Found match!
        // Simplify [1] + [2] + ... + [i] + ... + [j] + ...
        // As: [1] + [2] + a * c ... + ...  + ...
        Val* ac = IrBuilder::newScalar(*value->getDataType());
        IrBuilder::create<FOp>(BinaryOpType::Mul, ac, std::vector<Val*>{a1, c});
        std::vector<Val*> terms{ac};
        for (auto k : c10::irange(fadd->inputs().size())) {
          if (k == i || k == j) {
            continue;
          }
          terms.emplace_back(fadd->input(k));
        }
        if (terms.size() == 1) {
          return terms.at(0);
        } else {
          Val* result = IrBuilder::newScalar(*value->getDataType());
          IrBuilder::create<FOp>(BinaryOpType::Add, result, terms);
          return result;
        }
      }
    }
  }
  return value;
}

} // namespace rules

#define RUN_PASS(pass_name)                                      \
  if (disabled_passes == nullptr ||                              \
      (!disabled_passes->empty() &&                              \
       disabled_passes->count(#pass_name) == 0)) {               \
    simplified = recurseDown(simplified, [&var_info](Val* val) { \
      return rules::pass_name(val, var_info);                    \
    });                                                          \
    logger->record(#pass_name, simplified);                      \
  }

// Requires that all the passes before the barrier to be converged before
// procceeding to the passes after the barrier.
#define PASS_BARRIER                \
  if (old_simplified != simplified) \
  continue

Val* simplifyExpr(Val* value, const std::list<VarInfo>& variables) {
  FusionGuard fg(value->fusion());
  const VarInfoMap var_info(variables);
  auto logger = debug_print::createLogger(value);

  // nullptr -> disable nothing
  // empty set -> disable everything
  // {"abc", "def"} -> disable passes abc and def
  std::unique_ptr<std::unordered_set<std::string>> disabled_passes = nullptr;
  if (isOptionDisabled(DisableOption::ExprSimplify)) {
    const auto& v = getDisableOptionArguments(DisableOption::ExprSimplify);
    disabled_passes =
        std::make_unique<std::unordered_set<std::string>>(v.begin(), v.end());
  }

  Val* simplified = value;
  Val* old_simplified = nullptr;
  while (old_simplified != simplified) {
    old_simplified = simplified;

    // Passes other than assoc_comm::flatten assumes that all
    // associative-and-commutative binary ops (such as + *) are flattened. So
    // that they don't need to worry about things like
    // (a + b) + c vs a + (b + c)
    // So the first step before running all other passes is always flattening
    //
    // Note that, some passes might create nested flattened ops, something like
    // FlattenedAdd(FlattenedAdd(...), ...), so we should rerun flatten at the
    // beginning of each round instead of flattening before the while loop.
    simplified = assoc_comm::flatten(simplified);
    logger->record(debug_print::kFlattenName, simplified);

    RUN_PASS(eliminateTrivialComputation);
    RUN_PASS(eliminateTrivialPredicate);
    RUN_PASS(simplifyDivisibleDivMod);
    RUN_PASS(cancelDivMod);
    RUN_PASS(fundamentalDivisionWithRemainderProperty);
    PASS_BARRIER;
    RUN_PASS(distributeDivisibleDivMod);
    PASS_BARRIER;
    RUN_PASS(distributeMul);
    PASS_BARRIER;
    RUN_PASS(reducePredicateRegisterUsage);
  }

  auto unflattened = assoc_comm::unflatten(simplified, variables);
  logger->record(debug_print::kUnflattenName, unflattened);
  return unflattened;
}

#undef RUN_PASS

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
