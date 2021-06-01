
namespace torch {
namespace jit {

using FunctionTable = std::unordered_map<std::string, Function&>;
using ValueTable = std::unordered_map<std::string, SugaredValuePtr>;
using TypeTable = std::unordered_map<std::string, TypePtr>;
using AttributeMap = std::unordered_map<std::string, Const>;
using ListAttributeMap = std::unordered_map<std::string, std::vector<Const>>;

/* ======================================================== */
/*                      Program Status                      */
/* ======================================================== */

enum NoneStatus { ALWAYS, MAYBE, NEVER };
NoneStatus canBeNone(Value* v) {
  if (v->node()->mustBeNone()) {
    return ALWAYS;
  }
  if (v->type()->kind() == OptionalType::Kind) {
    return MAYBE;
  }
  return NEVER;
}

// Information for each def being emitted.
// Defs can be nested to support closures so we need a stack of this information
// Currently records information about the functions return type.
struct DefContext {
  TypePtr declared_return_type_; // nullptr if not annotated
  TypePtr merged_return_type_; // nullptr if a Return has not been seen yet
};

enum class LoopStatus { NOT_IN_LOOP, IN_LOOP, IN_UNROLLED_LOOP };

struct WithLoopStatus {
  WithLoopStatus(LoopStatus* prev, LoopStatus new_status) {
    prev_value_ = *prev;
    prev_ptr_ = prev;
    *prev = new_status;
  }
  ~WithLoopStatus() {
    *prev_ptr_ = prev_value_;
  }

 private:
  LoopStatus* prev_ptr_;
  LoopStatus prev_value_;
};

/* ==================================================================== */
/*                      `__setstate__` Information                      */
/* ==================================================================== */

// see [setstate type]
static TypePtr getTypeForSetStateArg(const Def& def, const Self* self) {
  TORCH_CHECK(self, "Expected __setstate__ to have a `self` argument");
  auto getstate = self->getClassType()->findMethod("__getstate__");
  if (!getstate) {
    throw ErrorReport(def.range())
        << "`__setstate__` defined but not `__getstate__`. "
        << "You must have both defined on a ScriptModule "
        << "to customize serialization.\n"
        << "Did you forget to use `@torch.jit.export`?";
  }
  getstate->ensure_defined();
  return self->getClassType()
      ->getMethod("__getstate__")
      .getSchema()
      .returns()
      .at(0)
      .type();
}

// see [setstate type]
static bool shouldDeriveSetStateType(
    const Def& def,
    const FunctionSchema& schema) {
  const bool noTypeAnnotations = std::all_of(
      schema.arguments().begin(),
      schema.arguments().end(),
      [](const Argument& arg) { return arg.is_inferred_type(); });

  bool shouldInfer = def.name().name() == "__setstate__" && noTypeAnnotations;
  if (!shouldInfer) {
    return false;
  }

  // Do some additional basic validation that the __setstate__ func is
  // well-formed
  TORCH_INTERNAL_ASSERT(def.name().name() == "__setstate__");
  const auto numDeclParams = def.decl().params().size();
  if (numDeclParams != 2) {
    throw ErrorReport(def.range())
        << "Expected 2 arguments for `__setstate__`, got: " << numDeclParams;
  }
  return true;
}

/* ======================================================== */
/*                      Getters                             */
/*         (Given X, return the corresponding Y)            */
/* ======================================================== */

NodeKind getNodeKind(int kind, int ninputs) {
  switch (kind) {
    case '+':
      return aten::add;
    case '-':
      return aten::sub;
    case TK_UNARY_MINUS:
      return aten::neg;
    case '*':
      return aten::mul;
    case TK_POW:
      return aten::pow;
    case '@':
      return aten::matmul;
    case TK_STARRED:
      return prim::Starred;
    case '/':
      return aten::div;
    case '%':
      return aten::remainder;
    case TK_NE:
      return aten::ne;
    case TK_EQ:
      return aten::eq;
    case '<':
      return aten::lt;
    case '>':
      return aten::gt;
    case TK_LE:
      return aten::le;
    case TK_GE:
      return aten::ge;
    case TK_AND:
      return aten::__and__;
    case TK_OR:
      return aten::__or__;
    case TK_IS:
      return aten::__is__;
    case TK_ISNOT:
      return aten::__isnot__;
    case TK_NOT:
      return aten::__not__;
    case TK_FLOOR_DIV:
      return aten::floordiv;
    case TK_LSHIFT:
      return aten::__lshift__;
    case TK_RSHIFT:
      return aten::__rshift__;
    case '&':
      return aten::__and__;
    case '|':
      return aten::__or__;
    case '^':
      return aten::__xor__;
    case TK_IN:
      return aten::__contains__;
    default:
      throw std::runtime_error("unknown kind " + c10::to_string(kind));
  }
}

NodeKind reverseComparision(NodeKind kind) {
  if (kind == aten::lt) {
    return aten::gt;
  } else if (kind == aten::le) {
    return aten::ge;
  } else if (kind == aten::gt) {
    return aten::lt;
  } else if (kind == aten::ge) {
    return aten::le;
  }
  throw std::runtime_error(
      "reverseComparision: unsupported NodeKind. File a bug");
}

std::string getOperatorOverload(int kind, int ninputs) {
  switch (kind) {
    case '+':
      return "__add__";
    case '-':
      return "__sub__";
    case TK_UNARY_MINUS:
      return "__neg__";
    case '~':
      return "__invert__";
    case '*':
      return "__mul__";
    case TK_POW:
      return "__pow__";
    case '/':
      return "__truediv__";
    case '%':
      return "__mod__";
    case TK_NE:
      return "__ne__";
    case TK_EQ:
      return "__eq__";
    case '<':
      return "__lt__";
    case '>':
      return "__gt__";
    case TK_LE:
      return "__le__";
    case TK_GE:
      return "__ge__";
    case '&':
      return "__and__";
    case '|':
      return "__or__";
    case '^':
      return "__xor__";
    case TK_IN:
      return "__contains__";
    case TK_LSHIFT:
      return "__lshift__";
    case TK_RSHIFT:
      return "__rshift__";
    default:
      throw std::runtime_error("unknown kind " + c10::to_string(kind));
  }
}

static Value* asSimple(const SugaredValuePtr& value) {
  if (SimpleValue* sv = dynamic_cast<SimpleValue*>(value.get())) {
    return sv->getValue();
  }
  return nullptr;
}

static std::shared_ptr<MagicMethod> makeMagic(
    const std::string& name,
    SugaredValuePtr base) {
  return std::make_shared<MagicMethod>(name, base);
}

// Get the appropriate builtin op for this augmented assignment
// If the RHS is a tensor, return the corresponding ATen in-place op
// If it's a list of scalars, then return the corresponding list augment op
Symbol getAugOp(const AugAssign& stmt, const TypePtr& type) {
  bool use_inplace_op = type->isSubtypeOf(TensorType::get()) ||
      type->kind() == TypeKind::ListType;
  switch (stmt.aug_op()) {
    case '+':
      return use_inplace_op ? aten::add_ : aten::add;
    case '-':
      return use_inplace_op ? aten::sub_ : aten::sub;
    case '/':
      return use_inplace_op ? aten::div_ : aten::div;
    case '*':
      return use_inplace_op ? aten::mul_ : aten::mul;
    case '%':
      return use_inplace_op ? aten::fmod_ : aten::fmod;
    case '|':
      return use_inplace_op ? aten::bitwise_or : aten::__or__;
    case '&':
      return use_inplace_op ? aten::bitwise_and : aten::__and__;
    case '^':
      return use_inplace_op ? aten::bitwise_xor : aten::__xor__;
    case TK_LSHIFT:
      // NOLINTNEXTLINE(bugprone-branch-clone)
      return use_inplace_op ? aten::__lshift__ : aten::__lshift__;
    case TK_RSHIFT:
      return use_inplace_op ? aten::__irshift__ : aten::__rshift__;
    case TK_POW:
      return aten::pow;
    default:
      throw ErrorReport(stmt)
          << "Unknown augmented assignment: " << kindToString(stmt.aug_op());
  }
}

// Get a pair of <in place magic method name, out of place magic method name>
// since the out of place method is called if the in place method is not
// present
std::pair<std::string, std::string> getAugMagicMethod(const AugAssign& stmt) {
  switch (stmt.aug_op()) {
    case '+':
      return std::make_pair(std::string("__iadd__"), std::string("__add__"));
    case '-':
      return std::make_pair(std::string("__isub__"), std::string("__sub__"));
    case '/':
      return std::make_pair(
          std::string("__itruediv__"), std::string("__truediv__"));
    case '*':
      return std::make_pair(std::string("__imul__"), std::string("__mul__"));
    case '%':
      return std::make_pair(std::string("__imod__"), std::string("__mod__"));
    default:
      throw ErrorReport(stmt)
          << "Unknown augmented assignment: " << kindToString(stmt.aug_op());
  }
}

// we consider _N where N is a number, to be a non-meaningful name
// and do not record it as a unique name. This allows python printing to
// be able to export and import more consistently named graphs
bool meaningfulName(const std::string& name) {
  if (name.size() == 0)
    return false;
  if (name[0] == '$')
    return false;
  if (name[0] != '_')
    return true;
  for (size_t i = 1; i < name.size(); ++i) {
    if (!isdigit(name[i]))
      return true;
  }
  return false;
}

/* ======================================================== */
/*                      Verification                        */
/* ======================================================== */

void checkApplyNumInputs(Apply& apply, size_t expected_inputs) {
  const SourceRange& loc = apply.range();
  if (apply.inputs().size() != expected_inputs) {
    throw ErrorReport(loc) << Var(apply.callee()).name().name()
                           << " expected exactly " << expected_inputs
                           << " arguments but found " << apply.inputs().size();
  }
  if (apply.attributes().size() > 0) {
    throw ErrorReport(loc) << Var(apply.callee()).name().name()
                           << " takes no keyword arguments";
  }
}

inline bool isSupportedListElementType(const TypePtr& type) {
  return type->isSubtypeOf(TensorType::get()) ||
      type->isSubtypeOf(NumberType::get());
}

// Validate that the `lhs` Expr's in an assignment statement are valid. That
// is:
//
// 1) All lhs Expr's are either Var, Tuple or Starred nodes
// 2) There is at most one Starred node in the lhs Expr
// 3) A Starred node can only appear when there is another non-Starred lhs
//    Expr. Concretely this means that `*abc = func()` is illegal. Unpacking
//    all outputs into a tuple is covered by `abc = func()`.
bool validateAssignLhsExpr(const List<Expr>& lhs, const SourceRange& r) {
  size_t num_normal_assign = 0;
  size_t num_starred = 0;
  for (const auto& assignee : lhs) {
    if (assignee.kind() == TK_VAR || assignee.kind() == TK_SUBSCRIPT ||
        assignee.kind() == TK_TUPLE_LITERAL || assignee.kind() == '.') {
      num_normal_assign++;
    } else if (assignee.kind() == TK_STARRED) {
      num_starred++;
    } else {
      throw ErrorReport(assignee) << "lhs of assignment must be a variable, "
                                  << "subscript, or starred expression";
    }
  }

  if (num_starred > 1) {
    throw ErrorReport(r) << "Only one starred expression is allowed on the lhs";
  }

  if (num_starred > 0 && num_normal_assign == 0) {
    throw ErrorReport(r) << "A Starred expression may only appear on the "
                         << "lhs within the presence of another non-starred"
                         << " expression";
  }

  return num_starred;
}

/* ================================================ */
/*                      Misc                        */
/* ================================================ */

template <class T, class Hash>
static Value* materializeConstant(
    T val,
    Graph& graph,
    const SourceRange& r,
    std::unordered_map<T, Value*, Hash>& map) {
  auto existing_constant = map.find(val);
  if (existing_constant != map.end()) {
    return existing_constant->second;
  }

  WithInsertPoint guard(graph.block()->nodes().front());
  auto new_constant = graph.insertConstant(val, r);
  map[val] = new_constant;

  return new_constant;
}

int64_t getAdjTupleIndex(
    const SourceRange& loc,
    const TupleTypePtr& tuple_type,
    int64_t input_index,
    bool allow_out_of_bounds) {
  // set index to be positive to simplify logic in runtime
  int64_t adj_index = input_index;
  int64_t tuple_len = tuple_type->elements().size();
  if (input_index < 0) {
    adj_index = tuple_len + input_index;
  }
  if (!allow_out_of_bounds && (adj_index >= tuple_len || adj_index < 0)) {
    throw ErrorReport(loc) << "Tuple index out of range. Tuple is length "
                           << tuple_len << " and index is " << input_index;
  }
  return adj_index;
}

} // namespace jit
} // namespace torch
