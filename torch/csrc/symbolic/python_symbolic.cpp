#include <torch/csrc/symbolic/python_symbolic.h>

#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/symbolic/Expr.h>
#include <torch/csrc/symbolic/NativeShapeEnv.h>
#include <torch/csrc/symbolic/NativeSymNodeImpl.h>
#include <torch/csrc/symbolic/PyFallback.h>
#include <torch/csrc/symbolic/ValueRanges.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_symnode.h>

#include <c10/core/impl/TorchDispatchModeTLS.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <optional>
#include <unordered_map>
#include <vector>

namespace torch::symbolic {

namespace py = pybind11;

namespace {

// Mirrors whether torch._ops' process-global pre-dispatch PROXY slot is set.
std::atomic<bool> pre_dispatch_proxy_set{false};

constexpr Kind kFunctionKinds[] = {
    Kind::Mod,
    Kind::PythonMod,
    Kind::FloorDiv,
    Kind::CleanDiv,
    Kind::Max,
    Kind::Min,
    Kind::PowByNatural,
    Kind::FloatPow,
    Kind::FloatTrueDiv,
    Kind::IntTrueDiv,
    Kind::CeilToInt,
    Kind::FloorToInt,
    Kind::TruncToInt,
    Kind::RoundToInt,
    Kind::RoundDecimal,
    Kind::ToFloat,
    Kind::TruncToFloat,
    Kind::IsNonOverlappingAndDenseIndicator};

// Python-side state for an ExprArena: the sympy Symbol objects that native
// symbols came from and a conversion cache (conversion is pure per Expr).
struct PyArena {
  PyArena() {
    py::module_ sympy = py::module_::import("sympy");
    py::module_ numbers = py::module_::import("torch.utils._sympy.numbers");
    Integer = sympy.attr("Integer");
    Rational = sympy.attr("Rational");
    Float = sympy.attr("Float");
    Symbol = sympy.attr("Symbol");
    Dummy = sympy.attr("Dummy");
    Add = sympy.attr("Add");
    Mul = sympy.attr("Mul");
    Pow = sympy.attr("Pow");
    IntInfinity = numbers.attr("IntInfinity");
    NegativeIntInfinity = numbers.attr("NegativeIntInfinity");
    int_oo = numbers.attr("int_oo");
    true_ = sympy.attr("true");
    false_ = sympy.attr("false");
    Not = sympy.attr("Not");
    And = sympy.attr("And");
    Or = sympy.attr("Or");
    lattice_new = py::module_::import("builtins")
                      .attr("super")(
                          py::module_::import("sympy.core.operations")
                              .attr("AssocOp"),
                          And)
                      .attr("__new__");
    for (const char* name : {"Eq", "Ne", "Lt", "Le", "Gt", "Ge"}) {
      relationals.push_back(sympy.attr(name));
    }
    py::module_ functions = py::module_::import("torch.utils._sympy.functions");
    for (auto k : kFunctionKinds) {
      function_classes.push_back(functions.attr(function_name(k)));
    }
  }

  const Expr* from_sympy(py::handle obj);
  py::object to_sympy(const Expr* e);

  c10::intrusive_ptr<ExprArena> arena = c10::make_intrusive<ExprArena>();
  py::dict symbol_index;
  std::vector<const Expr*> symbol_exprs;
  std::vector<py::object> symbol_objects;
  std::unordered_map<uint32_t, py::object> sympy_cache;

  py::object Integer, Rational, Float, Symbol, Dummy, Add, Mul, Pow,
      IntInfinity, NegativeIntInfinity, int_oo, true_, false_, Not, And, Or;
  // super(AssocOp, cls).__new__, which LatticeOp.__new__ calls with the final
  // ordered args.
  py::object lattice_new;
  // Indexed by Kind - Kind::Eq.
  std::vector<py::object> relationals;
  // Parallel to kFunctionKinds.
  std::vector<py::object> function_classes;
};

struct PyExpr {
  std::shared_ptr<PyArena> owner;
  const Expr* expr;
};

// NativeShapeEnv::binding().
struct EnvBinding {
  std::shared_ptr<PyArena> arena;
  // A weakref to the ShapeEnv, or None. The ShapeEnv owns the native env.
  py::object shape_env;
  // The ShapeEnv while the env has live nodes, else None.
  py::object live_shape_env = py::none();
};

struct PyShapeEnv {
  PyShapeEnv(std::shared_ptr<PyArena> owner, const py::object& shape_env)
      : owner(std::move(owner)),
        env(c10::make_intrusive<NativeShapeEnv>(this->owner->arena)) {
    auto* binding = new EnvBinding{
        this->owner,
        shape_env.is_none() ? py::object(py::none())
                            : py::object(py::weakref(shape_env))};
    // Native nodes may drop the env without the GIL.
    env->set_binding(std::shared_ptr<EnvBinding>(binding, [](EnvBinding* b) {
      if (!Py_IsInitialized()) {
        return;
      }
      py::gil_scoped_acquire gil;
      delete b;
    }));
  }

  std::shared_ptr<PyArena> owner;
  c10::intrusive_ptr<NativeShapeEnv> env;
};

std::optional<Fact> fact_from_name(const std::string& name) {
  for (size_t i = 0; i < kNumFacts; ++i) {
    if (name == fact_name(static_cast<Fact>(i))) {
      return static_cast<Fact>(i);
    }
  }
  return std::nullopt;
}

int64_t to_int64(py::handle obj) {
  int overflow = 0;
  long long v = PyLong_AsLongLongAndOverflow(obj.ptr(), &overflow);
  if (overflow != 0) {
    throw NativeUnsupported("integer overflow");
  }
  if (v == -1 && PyErr_Occurred()) {
    throw py::error_already_set();
  }
  return v;
}

// A None, int or bool hint; nullopt for any other, or an int that does not
// fit.
std::optional<Hint> hint_from_py(py::handle hint) {
  if (PyBool_Check(hint.ptr())) {
    return Hint(hint.ptr() == Py_True);
  }
  if (PyLong_CheckExact(hint.ptr())) {
    int overflow = 0;
    int64_t v = PyLong_AsLongLongAndOverflow(hint.ptr(), &overflow);
    if (overflow != 0) {
      return std::nullopt;
    }
    return Hint(v);
  }
  if (hint.is_none()) {
    return Hint();
  }
  return std::nullopt;
}

const Expr* PyArena::from_sympy(py::handle obj) {
  if (PyLong_CheckExact(obj.ptr()) || py::isinstance(obj, Integer)) {
    py::int_ v(py::reinterpret_borrow<py::object>(obj));
    return arena->integer(to_int64(v));
  }
  if (py::isinstance(obj, Rational)) {
    return arena->rational(to_int64(obj.attr("p")), to_int64(obj.attr("q")));
  }
  if (py::isinstance(obj, Float)) {
    if (obj.attr("_prec").cast<int64_t>() != 53) {
      throw NativeUnsupported("Float of precision other than 53");
    }
    auto v = obj.cast<double>();
    // float() rounds Floats outside the doubles.
    if (!Float(v).equal(obj)) {
      throw NativeUnsupported("Float that is not a double");
    }
    return arena->float_number(v);
  }
  if (py::isinstance(obj, IntInfinity)) {
    return arena->int_oo();
  }
  if (py::isinstance(obj, NegativeIntInfinity)) {
    return arena->neg_int_oo();
  }
  // Exact type: Dummy/Wild compare by more than name and assumptions.
  if (Py_TYPE(obj.ptr()) == reinterpret_cast<PyTypeObject*>(Symbol.ptr())) {
    if (symbol_index.contains(obj)) {
      return symbol_exprs.at(symbol_index[obj].cast<size_t>());
    }
    // sympy Symbol equality covers every assumption, so any assumption we do
    // not track could merge distinct symbols.
    Facts facts;
    facts.fill(Tri::Unknown);
    py::dict assumptions = obj.attr("assumptions0");
    for (auto [k, v] : assumptions) {
      auto name = k.cast<std::string>();
      auto f = fact_from_name(name);
      if (!f) {
        throw NativeUnsupported("untracked assumption " + name);
      }
      facts[static_cast<size_t>(*f)] = v.cast<bool>() ? Tri::True : Tri::False;
    }
    const Expr* e = arena->symbol(obj.attr("name").cast<std::string>(), facts);
    auto idx = static_cast<size_t>(e->p);
    if (idx >= symbol_objects.size()) {
      symbol_objects.resize(idx + 1);
      symbol_exprs.resize(idx + 1);
    }
    symbol_objects[idx] = py::reinterpret_borrow<py::object>(obj);
    symbol_exprs[idx] = e;
    symbol_index[obj] = idx;
    return e;
  }
  bool is_add = py::isinstance(obj, Add);
  if (is_add || py::isinstance(obj, Mul)) {
    std::vector<const Expr*> args;
    for (py::handle a : obj.attr("args")) {
      args.push_back(from_sympy(a));
    }
    // Only evaluate=False builds this; native construction would distribute.
    if (!is_add && args.size() == 2 && args[0]->is_number() &&
        args[1]->kind == Kind::Add) {
      throw NativeUnsupported("unevaluated Mul(c, Add)");
    }
    return is_add ? arena->add(args) : arena->mul(args);
  }
  if (py::isinstance(obj, Pow)) {
    py::tuple args = obj.attr("args");
    return arena->pow(from_sympy(args[0]), from_sympy(args[1]));
  }
  if (obj.is(true_) || obj.is(false_)) {
    return arena->boolean(obj.is(true_));
  }
  for (size_t i = 0; i < relationals.size(); ++i) {
    if (Py_TYPE(obj.ptr()) ==
        reinterpret_cast<PyTypeObject*>(relationals[i].ptr())) {
      py::tuple args = obj.attr("args");
      auto kind = static_cast<Kind>(static_cast<size_t>(Kind::Eq) + i);
      return arena->rel(
          kind, from_sympy(args[0]), from_sympy(args[1]), /*evaluate=*/false);
    }
  }
  if (py::isinstance(obj, Not)) {
    const Expr* a = from_sympy(py::tuple(obj.attr("args"))[0]);
    const Expr* r = arena->logical_not(a);
    if (r->kind != Kind::Not || r->args[0] != a) {
      throw NativeUnsupported("unevaluated Not");
    }
    return r;
  }
  for (size_t i = 0; i < function_classes.size(); ++i) {
    // Exact type: CleanDiv subclasses FloorDiv.
    if (Py_TYPE(obj.ptr()) ==
        reinterpret_cast<PyTypeObject*>(function_classes[i].ptr())) {
      std::vector<const Expr*> args;
      for (py::handle a : obj.attr("args")) {
        args.push_back(from_sympy(a));
      }
      Kind kind = kFunctionKinds[i];
      // A Max/Min is taken as is: _collapse_arguments leaves unevaluated
      // ones in the args of evaluated ones.
      const Expr* r = kind == Kind::Max || kind == Kind::Min
          ? arena->minmax(kind, args, false)
          : arena->function(kind, args);
      if (r->kind != kind ||
          !std::equal(
              r->args.begin(), r->args.end(), args.begin(), args.end())) {
        throw NativeUnsupported("unevaluated function");
      }
      return r;
    }
  }
  for (auto kind : {Kind::And, Kind::Or}) {
    py::handle cls = kind == Kind::And ? And : Or;
    if (Py_TYPE(obj.ptr()) == reinterpret_cast<PyTypeObject*>(cls.ptr())) {
      std::vector<const Expr*> args;
      for (py::handle a : obj.attr("args")) {
        args.push_back(from_sympy(a));
      }
      return arena->lattice_from_args(kind, args);
    }
  }
  throw NativeUnsupported(
      "unsupported sympy type " +
      py::str(py::type::handle_of(obj).attr("__name__")).cast<std::string>());
}

py::object PyArena::to_sympy(const Expr* e) {
  auto it = sympy_cache.find(e->id);
  if (it != sympy_cache.end()) {
    return it->second;
  }
  py::object r;
  switch (e->kind) {
    case Kind::Integer:
      r = Integer(e->p);
      break;
    case Kind::Rational:
      r = Rational(e->p, e->q);
      break;
    case Kind::Float:
      r = Float(e->float_value());
      break;
    case Kind::IntInfinity:
      r = int_oo;
      break;
    case Kind::NegativeIntInfinity:
      r = int_oo.attr("__neg__")();
      break;
    case Kind::Symbol: {
      const SymbolInfo& info = arena->symbol_info(e);
      if (info.dummy_index == 0) {
        r = symbol_objects.at(e->p);
        break;
      }
      py::dict assumptions;
      for (size_t i = 0; i < kNumFacts; ++i) {
        auto f = static_cast<Fact>(i);
        if (info.facts.get(f) != Tri::Unknown) {
          assumptions[fact_name(f)] = info.facts.get(f) == Tri::True;
        }
      }
      r = Dummy(info.name, **assumptions);
      break;
    }
    case Kind::Add:
    case Kind::Mul:
    case Kind::Pow: {
      py::tuple args(e->args.size());
      for (size_t i = 0; i < e->args.size(); ++i) {
        args[i] = to_sympy(e->args[i]);
      }
      r =
          (e->kind == Kind::Add       ? Add
               : e->kind == Kind::Mul ? Mul
                                      : Pow)(*args);
      break;
    }
    case Kind::BooleanTrue:
      r = true_;
      break;
    case Kind::BooleanFalse:
      r = false_;
      break;
    case Kind::Eq:
    case Kind::Ne:
    case Kind::Lt:
    case Kind::Le:
    case Kind::Gt:
    case Kind::Ge: {
      py::object cls = relationals.at(
          static_cast<size_t>(e->kind) - static_cast<size_t>(Kind::Eq));
      r =
          cls(to_sympy(e->args[0]),
              to_sympy(e->args[1]),
              py::arg("evaluate") = false);
      break;
    }
    case Kind::Not:
      r = Not(to_sympy(e->args[0]));
      break;
    case Kind::And:
    case Kind::Or: {
      py::tuple args(e->args.size());
      for (size_t i = 0; i < e->args.size(); ++i) {
        args[i] = to_sympy(e->args[i]);
      }
      // Rebuilding with Or(*args) could re-filter: Or's filter is not
      // idempotent.
      py::object cls = e->kind == Kind::And ? And : Or;
      r = lattice_new(cls, *args);
      r.attr("_argset") = py::frozenset(args);
      break;
    }
    case Kind::Mod:
    case Kind::PythonMod:
    case Kind::FloorDiv:
    case Kind::CleanDiv:
    case Kind::Max:
    case Kind::Min:
    case Kind::PowByNatural:
    case Kind::FloatPow:
    case Kind::FloatTrueDiv:
    case Kind::IntTrueDiv:
    case Kind::CeilToInt:
    case Kind::FloorToInt:
    case Kind::TruncToInt:
    case Kind::RoundToInt:
    case Kind::RoundDecimal:
    case Kind::ToFloat:
    case Kind::TruncToFloat:
    case Kind::IsNonOverlappingAndDenseIndicator: {
      py::tuple args(e->args.size());
      for (size_t i = 0; i < e->args.size(); ++i) {
        args[i] = to_sympy(e->args[i]);
      }
      auto* it = std::find(
          std::begin(kFunctionKinds), std::end(kFunctionKinds), e->kind);
      r = function_classes.at(it - std::begin(kFunctionKinds))(
          *args, py::arg("evaluate") = false);
      break;
    }
  }
  sympy_cache.emplace(e->id, r);
  return r;
}

const char* kind_name(Kind k) {
  switch (k) {
    case Kind::Integer:
      return "Integer";
    case Kind::Rational:
      return "Rational";
    case Kind::Float:
      return "Float";
    case Kind::IntInfinity:
      return "IntInfinity";
    case Kind::NegativeIntInfinity:
      return "NegativeIntInfinity";
    case Kind::Symbol:
      return "Symbol";
    case Kind::Pow:
      return "Pow";
    case Kind::Mul:
      return "Mul";
    case Kind::Add:
      return "Add";
    case Kind::BooleanTrue:
      return "BooleanTrue";
    case Kind::BooleanFalse:
      return "BooleanFalse";
    case Kind::Eq:
      return "Eq";
    case Kind::Ne:
      return "Ne";
    case Kind::Lt:
      return "Lt";
    case Kind::Le:
      return "Le";
    case Kind::Gt:
      return "Gt";
    case Kind::Ge:
      return "Ge";
    case Kind::Not:
      return "Not";
    case Kind::And:
      return "And";
    case Kind::Or:
      return "Or";
    case Kind::Mod:
    case Kind::PythonMod:
    case Kind::FloorDiv:
    case Kind::CleanDiv:
    case Kind::Max:
    case Kind::Min:
    case Kind::PowByNatural:
    case Kind::FloatPow:
    case Kind::FloatTrueDiv:
    case Kind::IntTrueDiv:
    case Kind::CeilToInt:
    case Kind::FloorToInt:
    case Kind::TruncToInt:
    case Kind::RoundToInt:
    case Kind::RoundDecimal:
    case Kind::ToFloat:
    case Kind::TruncToFloat:
    case Kind::IsNonOverlappingAndDenseIndicator:
      return function_name(k);
  }
  return "?";
}

Kind relational_kind(const std::string& op) {
  for (auto k : {Kind::Eq, Kind::Ne, Kind::Lt, Kind::Le, Kind::Gt, Kind::Ge}) {
    if (op == kind_name(k)) {
      return k;
    }
  }
  TORCH_CHECK(false, "unknown relational ", op);
}

Kind function_kind(const std::string& name) {
  for (auto k : kFunctionKinds) {
    if (name == function_name(k)) {
      return k;
    }
  }
  TORCH_CHECK(false, "unknown function ", name);
}

const Expr* unwrap(const std::shared_ptr<PyArena>& self, const PyExpr& e) {
  TORCH_CHECK(e.owner == self, "expression belongs to a different arena");
  return e.expr;
}

py::object to_py(Tri t) {
  if (t == Tri::Unknown) {
    return py::none();
  }
  return py::bool_(t == Tri::True);
}

py::set fact_set(uint32_t true_mask, uint32_t false_mask) {
  py::set r;
  for (size_t i = 0; i < kNumFacts; ++i) {
    for (bool v : {false, true}) {
      if (((v ? true_mask : false_mask) >> i) & 1) {
        r.add(py::make_tuple(fact_name(static_cast<Fact>(i)), v));
      }
    }
  }
  return r;
}

// _assume_rules as the Python structures sympy uses, for testing the tables.
py::tuple assume_rules_to_py() {
  AssumeRules rules = assume_rules();
  py::dict implications;
  py::dict triggers;
  py::dict prereq;
  for (size_t i = 0; i < kNumFacts; ++i) {
    const char* name = fact_name(static_cast<Fact>(i));
    for (bool v : {false, true}) {
      const Implication& imp = rules.full_implications[i][v];
      implications[py::make_tuple(name, v)] =
          fact_set(imp.true_mask, imp.false_mask);
      py::set t;
      for (size_t b = 0; b < rules.beta_rules.size(); ++b) {
        if ((rules.beta_triggers[i][v] >> b) & 1) {
          t.add(py::int_(b));
        }
      }
      triggers[py::make_tuple(name, v)] = t;
    }
    py::set pre;
    for (size_t j = 0; j < kNumFacts; ++j) {
      if ((rules.prereq[i] >> j) & 1) {
        pre.add(py::str(fact_name(static_cast<Fact>(j))));
      }
    }
    prereq[py::str(name)] = pre;
  }
  py::list beta;
  for (const BetaRule& r : rules.beta_rules) {
    beta.append(
        py::make_tuple(
            fact_set(r.cond_true, r.cond_false),
            py::make_tuple(fact_name(r.fact), r.value)));
  }
  return py::make_tuple(implications, beta, triggers, prereq);
}

std::vector<const Expr*> unwrap_all(
    const std::shared_ptr<PyArena>& self,
    const std::vector<PyExpr>& es) {
  std::vector<const Expr*> r;
  r.reserve(es.size());
  for (const auto& e : es) {
    r.push_back(unwrap(self, e));
  }
  return r;
}

py::object sort_key_to_py(PyArena& arena, const SortKey& k) {
  switch (k.type) {
    case SortKey::Type::Int:
      return py::int_(k.i);
    case SortKey::Type::Str:
      return py::str(k.s);
    case SortKey::Type::Num:
      return arena.to_sympy(k.num);
    case SortKey::Type::Tuple:
      break;
  }
  py::tuple t(k.items.size());
  for (size_t i = 0; i < k.items.size(); ++i) {
    t[i] = sort_key_to_py(arena, *k.items[i]);
  }
  return t;
}

// A torch.fx.experimental._config entry, read the way ConfigModule.__getattr__
// resolves an entry without alias or justknob, but without running Python
// code. Everything but `hide` and the user override is fixed at creation.
struct ConfigEntry {
  ConfigEntry(py::handle entry, py::handle unset)
      : entry(py::reinterpret_borrow<py::object>(entry)),
        user_override(entry.attr("user_override")),
        forced(entry.attr("env_value_force")) {
    plain = entry.attr("alias").is_none() && entry.attr("justknob").is_none();
    fallback = entry.attr("env_value_default");
    if (fallback.is(unset)) {
      fallback = entry.attr("default");
    }
  }

  // Aliased, justknob and hidden entries count as set.
  bool falsy(PyObject* unset, PyObject* hide_str) const {
    if (!plain) {
      return false;
    }
    py::object hidden = py::reinterpret_steal<py::object>(
        PyObject_GetAttr(entry.ptr(), hide_str));
    if (!hidden) {
      throw py::error_already_set();
    }
    int r = PyObject_IsTrue(hidden.ptr());
    if (r != 0) {
      if (r < 0) {
        throw py::error_already_set();
      }
      return false;
    }
    PyObject* value = forced.ptr();
    py::object override_value;
    if (value == unset) {
      if (PyContextVar_Get(user_override.ptr(), nullptr, &value) != 0) {
        throw py::error_already_set();
      }
      override_value = py::reinterpret_steal<py::object>(value);
    }
    if (value == unset) {
      value = fallback.ptr();
    }
    r = PyObject_IsTrue(value);
    if (r < 0) {
      throw py::error_already_set();
    }
    return r == 0;
  }

  py::object entry;
  py::object user_override;
  py::object forced;
  py::object fallback;
  bool plain;
};

EnvBinding& binding_of(NativeShapeEnv& env) {
  auto* b = static_cast<EnvBinding*>(env.binding());
  TORCH_CHECK(b != nullptr, "native env without a Python binding");
  return *b;
}

py::object pytype_to_py(PyType t) {
  return py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(
      t == PyType::Int ? &PyLong_Type : &PyBool_Type));
}

// Puts the native mod choices where binary_magic_impl would have cached them.
// Runs before the env leaves pristine and before _symop_cache is copied.
void flush_mod_memo(PyShapeEnv& self) {
  const py::object& ref = binding_of(*self.env).shape_env;
  if (!self.env->pristine() || ref.is_none()) {
    return;
  }
  py::object shape_env = ref();
  if (shape_env.is_none()) {
    return;
  }
  PyArena& a = *self.owner;
  py::object cache = shape_env.attr("_symop_cache");
  py::object version = shape_env.attr("_replacements_version_counter");
  for (const auto& [key, out] : self.env->mod_memo()) {
    cache.attr("setdefault")(
        py::make_tuple(
            "mod", a.to_sympy(key.first), a.to_sympy(key.second), version),
        py::make_tuple(a.to_sympy(out), pytype_to_py(PyType::Int), false));
  }
}

py::object node_expr(const NativeSymNodeImpl& node) {
  NativeShapeEnv& env = *node.env();
  auto lock = lock_env(env);
  return binding_of(env).arena->to_sympy(node.expr());
}

c10::SymNode node_from_py(py::handle obj) {
  if (py::isinstance<c10::SymNodeImpl>(obj)) {
    return py::cast<c10::SymNode>(obj);
  }
  return c10::make_intrusive<impl::PythonSymNodeImpl>(
      py::reinterpret_borrow<py::object>(obj));
}

// sympy's node.expr.<attr>, computed by `native` from the native expr unless
// replacements could change node.expr.
template <typename F>
bool expr_query(const NativeSymNodeImpl& node, const char* attr, F native) {
  {
    NativeShapeEnv& env = *node.env();
    auto lock = lock_env(env);
    if (env.replacements_empty()) {
      return native(env.arena(), node.expr());
    }
  }
  return node_to_py(materialize(node)).attr("expr").attr(attr).cast<bool>();
}

std::vector<c10::SymNode> nodes_from_py(const py::sequence& nodes) {
  std::vector<c10::SymNode> r;
  r.reserve(nodes.size());
  for (py::handle n : nodes) {
    r.push_back(node_from_py(n));
  }
  return r;
}

py::list nodes_to_py(c10::ArrayRef<c10::SymNode> nodes) {
  py::list r(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    r[i] = node_to_py(nodes[i]);
  }
  return r;
}

py::handle python_symnode_class() {
  static py::handle cls =
      py::object(
          py::module_::import("torch.fx.experimental.sym_node").attr("SymNode"))
          .release();
  return cls;
}

} // namespace

py::object node_to_py(const c10::SymNode& n) {
  if (auto* p = dynamic_cast<impl::PythonSymNodeImpl*>(n.get())) {
    return py::reinterpret_borrow<py::object>(p->getPyObj());
  }
  return py::cast(n);
}

bool native_config_is_default() {
  std::optional<py::gil_scoped_acquire> gil;
  if (py::detail::get_thread_state_unchecked() == nullptr) {
    gil.emplace();
  }
  struct Entries {
    py::object unset;
    py::object hide_str;
    ConfigEntry backed;
    ConfigEntry aggressive;
  };
  static const auto* entries = [] {
    py::dict config =
        py::module_::import("torch.fx.experimental._config").attr("_config");
    py::object unset = py::module_::import("torch.utils._config_module")
                           .attr("_UNSET_SENTINEL");
    return new Entries{
        unset,
        py::reinterpret_steal<py::object>(PyUnicode_InternFromString("hide")),
        ConfigEntry(config["backed_size_oblivious"], unset),
        ConfigEntry(config["aggressive_guard_free_semantics"], unset)};
  }();
  PyObject* unset = entries->unset.ptr();
  PyObject* hide_str = entries->hide_str.ptr();
  return entries->backed.falsy(unset, hide_str) &&
      entries->aggressive.falsy(unset, hide_str);
}

void live_nodes_changed(NativeShapeEnv& env) {
  if (!Py_IsInitialized()) {
    return;
  }
  py::gil_scoped_acquire gil;
  auto* b = static_cast<EnvBinding*>(env.binding());
  if (b == nullptr || b->shape_env.is_none()) {
    return;
  }
  // Re-read under the GIL: concurrent transitions converge on the last one.
  if (env.live_nodes() == 0) {
    b->live_shape_env = py::none();
  } else if (b->live_shape_env.is_none()) {
    b->live_shape_env = b->shape_env();
  }
}

std::unique_lock<std::mutex> lock_env(NativeShapeEnv& env) {
  std::unique_lock<std::mutex> lock(env.mutex(), std::try_to_lock);
  if (!lock.owns_lock()) {
    if (PyGILState_Check()) {
      py::gil_scoped_release no_gil;
      lock.lock();
    } else {
      lock.lock();
    }
  }
  return lock;
}

c10::SymNode materialize(const NativeSymNodeImpl& node) {
  py::gil_scoped_acquire gil;
  EnvBinding& binding = binding_of(*node.env());
  py::object shape_env =
      binding.shape_env.is_none() ? py::none() : binding.shape_env();
  TORCH_CHECK(!shape_env.is_none(), "the ShapeEnv of a native SymNode is gone");
  py::module_ sym_node = py::module_::import("torch.fx.experimental.sym_node");
  py::object hint = std::holds_alternative<std::monostate>(node.hint())
      ? py::object(sym_node.attr("_NO_HINT"))
      : hint_to_py(node.hint());
  py::object constant = hint_to_py(node.constant());
  py::object r = sym_node.attr("SymNode")(
      node_expr(node),
      shape_env,
      pytype_to_py(node.pytype()),
      hint,
      py::arg("constant") = constant,
      py::arg("fx_node") = constant,
      py::arg("optimized_summation") = node.optimized_summation());
  return c10::make_intrusive<impl::PythonSymNodeImpl>(std::move(r));
}

bool proxy_mode() {
  return pre_dispatch_proxy_set.load(std::memory_order_relaxed) ||
      c10::impl::TorchDispatchModeTLS::get_mode(
          c10::impl::TorchDispatchModeKey::PROXY)
          .has_value();
}

c10::SymNode python_impl(const char* method, c10::ArrayRef<c10::SymNode> args) {
  py::gil_scoped_acquire gil;
  py::tuple py_args(nodes_to_py(args));
  return node_from_py(python_symnode_class().attr(method)(*py_args));
}

namespace {

// proxy_tensor._sym_register with thunkify off: _compute_proxy, then
// set_proxy_slot. Calls _sym_register for a lazy thunk, a Symbol result,
// set_proxy_slot debug logging, or an operand not in symnode_tracker.
void sym_register(
    const py::object& proxy_tensor,
    py::handle tracer,
    const py::object& op,
    const py::tuple& wrapped,
    const py::object& sym,
    const NativeSymNodeImpl& out) {
  static const auto* statics = new std::array<py::object, 5>{
      proxy_tensor.attr("log"),
      py::module_::import("logging").attr("DEBUG"),
      proxy_tensor.attr("py_sym_types"),
      proxy_tensor.attr("fx").attr("Proxy"),
      proxy_tensor.attr("Thunk")};
  const auto& [log, debug, sym_types, proxy_type, thunk_type] = *statics;
  auto python = [&] {
    proxy_tensor.attr("_sym_register")(tracer, op, wrapped, sym);
  };
  if (out.expr()->kind == Kind::Symbol ||
      tracer.attr("enable_thunkify").cast<bool>() ||
      log.attr("isEnabledFor")(debug).cast<bool>()) {
    return python();
  }
  py::dict nodes = tracer.attr("symnode_tracker").attr("sym_node_dict");
  py::tuple n_args(wrapped.size());
  for (size_t i = 0; i < wrapped.size(); ++i) {
    py::handle a = wrapped[i];
    int is_sym = PyObject_IsInstance(a.ptr(), sym_types.ptr());
    if (is_sym < 0) {
      throw py::error_already_set();
    }
    if (is_sym == 0) {
      n_args[i] = a;
      continue;
    }
    PyObject* thunk =
        PyDict_GetItemWithError(nodes.ptr(), a.attr("node").ptr());
    if (thunk == nullptr) {
      if (PyErr_Occurred()) {
        throw py::error_already_set();
      }
      return python();
    }
    n_args[i] = py::handle(thunk).attr("force")().attr("node");
  }
  py::object n_out =
      tracer.attr("create_node")("call_function", op, n_args, py::dict());
  py::object p_out = proxy_type(n_out, tracer);
  n_out.attr("meta")["val"] = sym;
  py::object thunk = thunk_type(py::none());
  thunk.attr("r") = p_out;
  py::object key = sym.attr("node");
  if (!nodes.contains(key)) {
    nodes[key] = thunk;
  }
}

} // namespace

c10::SymNode proxy_dispatch(
    const char* method,
    const char* op_name,
    c10::ArrayRef<c10::SymNode> args,
    bool mul,
    c10::function_ref<c10::SymNode()> compute) {
  py::gil_scoped_acquire gil;
  static const auto* modules = new std::array<py::object, 3>{
      py::module_::import("torch._logging._internal"),
      py::module_::import("torch.fx.experimental.sym_node"),
      py::module_::import("torch.fx.experimental.proxy_tensor")};
  const auto& [logging, sym_node, proxy_tensor] = *modules;
  if (logging.attr("GET_DTRACE_STRUCTURED").cast<bool>()) {
    return python_impl(method, args);
  }
  py::tuple wrapped(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    auto* n = static_cast<NativeSymNodeImpl*>(args[i].get());
    if (!std::holds_alternative<std::monostate>(n->constant())) {
      wrapped[i] = hint_to_py(n->constant());
    } else {
      wrapped[i] = make_sym_object(
          n->is_int() ? get_symint_class() : get_symbool_class(),
          node_to_py(args[i]));
    }
  }
  py::object op = sym_node.attr("METHOD_TO_OPERATOR")[op_name];
  py::object r;
  static const auto* proxy_mode_type =
      new py::object(proxy_tensor.attr("ProxyTorchDispatchMode"));
  static PyObject* const tracer_str = PyUnicode_InternFromString("tracer");
  auto mode = pre_dispatch_proxy_set.load(std::memory_order_relaxed)
      ? std::nullopt
      : c10::impl::TorchDispatchModeTLS::get_mode(
            c10::impl::TorchDispatchModeKey::PROXY);
  PyObject* mode_obj =
      mode.has_value() ? (*mode)->ptr(getPyInterpreter()) : nullptr;
  if (mode_obj != nullptr &&
      Py_TYPE(mode_obj) ==
          reinterpret_cast<PyTypeObject*>(proxy_mode_type->ptr()) &&
      sym_glue_native()) {
    auto is_one = [](const c10::SymNode& n) {
      const Hint& c = static_cast<NativeSymNodeImpl*>(n.get())->constant();
      const int64_t* i = std::get_if<int64_t>(&c);
      const bool* b = std::get_if<bool>(&c);
      return (i != nullptr && *i == 1) || (b != nullptr && *b);
    };
    if (mul && is_one(args[1])) {
      r = wrapped[0];
    } else if (mul && is_one(args[0])) {
      r = wrapped[1];
    } else if (c10::SymNode out = compute()) {
      py::object sym = make_sym_object(
          out->is_int() ? get_symint_class() : get_symbool_class(),
          node_to_py(out));
      sym_register(
          proxy_tensor,
          py::handle(mode_obj).attr(tracer_str),
          op,
          wrapped,
          sym,
          *static_cast<NativeSymNodeImpl*>(out.get()));
      return out;
    }
  }
  if (!r) {
    r = proxy_tensor.attr("handle_sym_dispatch")(op, wrapped, py::dict());
  }
  if (is_symint(r) || is_symfloat(r) || is_symbool(r)) {
    return node_from_py(r.attr("node"));
  }
  PyObject* t = reinterpret_cast<PyObject*>(Py_TYPE(r.ptr()));
  const char* wrap = t == reinterpret_cast<PyObject*>(&PyBool_Type) ? "wrap_bool"
      : t == reinterpret_cast<PyObject*>(&PyLong_Type)             ? "wrap_int"
      : t == reinterpret_cast<PyObject*>(&PyFloat_Type)            ? "wrap_float"
                                                                   : nullptr;
  if (wrap == nullptr) {
    return node_from_py(py::handle(Py_NotImplemented));
  }
  return node_from_py(node_to_py(args[0]).attr(wrap)(r));
}

c10::SymNode python_impl(
    const char* method,
    const c10::SymNode& self,
    c10::ArrayRef<c10::SymNode> sizes,
    c10::ArrayRef<c10::SymNode> strides) {
  py::gil_scoped_acquire gil;
  return node_from_py(python_symnode_class().attr(method)(
      node_to_py(self), nodes_to_py(sizes), nodes_to_py(strides)));
}

c10::SymNode python_impl(
    const char* method,
    const c10::SymNode& self,
    c10::ArrayRef<c10::SymNode> args) {
  py::gil_scoped_acquire gil;
  return node_from_py(
      python_symnode_class().attr(method)(node_to_py(self), nodes_to_py(args)));
}

void initSymbolicBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module_>();
  auto sm = m.def_submodule("_symbolic", "native symbolic expressions");
  py::register_exception<NativeUnsupported>(sm, "NativeUnsupported");
  initGlueBindings(sm);
  sm.def("_assume_rules", &assume_rules_to_py);
  sm.def("_native_config_is_default", &native_config_is_default);
  sm.def("_native_queries_pending", &NativeShapeEnv::queries_pending);
  sm.def("_set_suppress_guards", [](bool v) { suppress_guards_tls() = v; });
  sm.def("_suppress_guards", [] { return suppress_guards_tls(); });
  sm.def("_set_pre_dispatch_proxy", [](bool v) { pre_dispatch_proxy_set = v; });
  sm.def("_roundtrip_symbool", [](const c10::SymBool& v) { return v; });
  sm.def("_roundtrip_symfloat", [](const c10::SymFloat& v) { return v; });

  py::class_<PyExpr>(sm, "_Expr")
      .def(
          "__eq__",
          [](const PyExpr& a, const PyExpr& b) {
            return a.owner == b.owner && a.expr == b.expr;
          })
      .def(
          "__hash__",
          [](const PyExpr& a) { return std::hash<const Expr*>()(a.expr); })
      .def_property_readonly(
          "kind", [](const PyExpr& a) { return kind_name(a.expr->kind); })
      .def_property_readonly("id", [](const PyExpr& a) { return a.expr->id; });

  using Self = std::shared_ptr<PyArena>;
  auto wrap = [](const Self& self, const Expr* e) { return PyExpr{self, e}; };
  py::class_<PyArena, Self> arena_cls(sm, "_Arena");
  arena_cls.def(py::init<>())
      .def("__len__", [](const Self& self) { return self->arena->size(); })
      .def(
          "from_sympy",
          [wrap](const Self& self, py::handle obj) {
            return wrap(self, self->from_sympy(obj));
          })
      .def(
          "to_sympy",
          [](const Self& self, const PyExpr& e) {
            return self->to_sympy(unwrap(self, e));
          })
      .def(
          "integer",
          [wrap](const Self& self, py::handle v) {
            return wrap(self, self->arena->integer(to_int64(v)));
          })
      .def(
          "rational",
          [wrap](const Self& self, py::handle p, py::handle q) {
            return wrap(self, self->arena->rational(to_int64(p), to_int64(q)));
          })
      .def(
          "add",
          [wrap](const Self& self, const std::vector<PyExpr>& args) {
            return wrap(self, self->arena->add(unwrap_all(self, args)));
          })
      .def(
          "mul",
          [wrap](const Self& self, const std::vector<PyExpr>& args) {
            return wrap(self, self->arena->mul(unwrap_all(self, args)));
          })
      .def(
          "pow",
          [wrap](const Self& self, const PyExpr& b, const PyExpr& e) {
            return wrap(
                self, self->arena->pow(unwrap(self, b), unwrap(self, e)));
          })
      .def(
          "neg",
          [wrap](const Self& self, const PyExpr& a) {
            return wrap(self, self->arena->neg(unwrap(self, a)));
          })
      .def(
          "sub",
          [wrap](const Self& self, const PyExpr& a, const PyExpr& b) {
            return wrap(
                self, self->arena->sub(unwrap(self, a), unwrap(self, b)));
          })
      .def(
          "boolean",
          [wrap](const Self& self, bool v) {
            return wrap(self, self->arena->boolean(v));
          })
      .def(
          "rel",
          [wrap](
              const Self& self,
              const std::string& op,
              const PyExpr& lhs,
              const PyExpr& rhs,
              bool evaluate) {
            return wrap(
                self,
                self->arena->rel(
                    relational_kind(op),
                    unwrap(self, lhs),
                    unwrap(self, rhs),
                    evaluate));
          },
          py::arg("op"),
          py::arg("lhs"),
          py::arg("rhs"),
          py::arg("evaluate") = true)
      .def(
          "function",
          [wrap](
              const Self& self,
              const std::string& name,
              const std::vector<PyExpr>& args) {
            return wrap(
                self,
                self->arena->function(
                    function_kind(name), unwrap_all(self, args)));
          })
      .def(
          "ceildiv",
          [wrap](const Self& self, const PyExpr& a, const PyExpr& b) {
            return wrap(
                self, self->arena->ceildiv(unwrap(self, a), unwrap(self, b)));
          })
      .def(
          "lshift",
          [wrap](const Self& self, const PyExpr& a, const PyExpr& b) {
            return wrap(
                self, self->arena->lshift(unwrap(self, a), unwrap(self, b)));
          })
      .def(
          "rshift",
          [wrap](const Self& self, const PyExpr& a, const PyExpr& b) {
            return wrap(
                self, self->arena->rshift(unwrap(self, a), unwrap(self, b)));
          })
      .def(
          "logical_not",
          [wrap](const Self& self, const PyExpr& a) {
            return wrap(self, self->arena->logical_not(unwrap(self, a)));
          })
      .def(
          "logical_and",
          [wrap](const Self& self, const std::vector<PyExpr>& args) {
            return wrap(self, self->arena->logical_and(unwrap_all(self, args)));
          })
      .def(
          "logical_or",
          [wrap](const Self& self, const std::vector<PyExpr>& args) {
            return wrap(self, self->arena->logical_or(unwrap_all(self, args)));
          })
      .def(
          "is_eq",
          [](const Self& self, const PyExpr& a, const PyExpr& b) {
            return to_py(self->arena->is_eq(unwrap(self, a), unwrap(self, b)));
          })
      .def(
          "is_ge",
          [](const Self& self, const PyExpr& a, const PyExpr& b) {
            return to_py(self->arena->is_ge(unwrap(self, a), unwrap(self, b)));
          })
      .def(
          "args",
          [wrap](const Self& self, const PyExpr& e) {
            std::vector<PyExpr> r;
            for (const Expr* a : unwrap(self, e)->args) {
              r.push_back(wrap(self, a));
            }
            return r;
          })
      .def(
          "value_range",
          [wrap](
              const Self& self,
              const PyExpr& e,
              const std::vector<std::tuple<PyExpr, PyExpr, PyExpr>>& ranges) {
            RangeMap m;
            for (const auto& [sym, lower, upper] : ranges) {
              m.insert_or_assign(
                  unwrap(self, sym),
                  ValueRanges(unwrap(self, lower), unwrap(self, upper)));
            }
            ValueRanges r = value_range_interp(*self->arena, unwrap(self, e), m);
            return std::make_pair(wrap(self, r.lower), wrap(self, r.upper));
          })
      .def(
          "bound_sympy",
          [wrap](
              const Self& self,
              const PyExpr& e,
              const std::vector<std::tuple<PyExpr, PyExpr, PyExpr>>& ranges,
              const std::vector<std::tuple<PyExpr, PyExpr, PyExpr>>&
                  context_ranges) {
            auto to_map = [&](const auto& rs) {
              RangeMap m;
              for (const auto& [sym, lower, upper] : rs) {
                m.insert_or_assign(
                    unwrap(self, sym),
                    ValueRanges(unwrap(self, lower), unwrap(self, upper)));
              }
              return m;
            };
            RangeMap m = to_map(ranges);
            RangeMap context = to_map(context_ranges);
            ValueRanges r =
                bound_sympy(*self->arena, unwrap(self, e), m, &context);
            return std::make_pair(wrap(self, r.lower), wrap(self, r.upper));
          },
          py::arg("e"),
          py::arg("ranges"),
          py::arg("context_ranges") =
              std::vector<std::tuple<PyExpr, PyExpr, PyExpr>>{})
      .def(
          "compare",
          [](const Self& self, const PyExpr& a, const PyExpr& b) {
            return self->arena->compare(unwrap(self, a), unwrap(self, b));
          })
      .def(
          "sort_key",
          [](const Self& self, const PyExpr& e) {
            return sort_key_to_py(
                *self, *self->arena->sort_key(unwrap(self, e)));
          })
      .def(
          "ordered",
          [wrap](const Self& self, const std::vector<PyExpr>& seq) {
            std::vector<PyExpr> r;
            for (const Expr* e : self->arena->ordered(unwrap_all(self, seq))) {
              r.push_back(wrap(self, e));
            }
            return r;
          })
      .def(
          "could_extract_minus_sign",
          [](const Self& self, const PyExpr& e) {
            return self->arena->could_extract_minus_sign(unwrap(self, e));
          })
      .def(
          "sstr",
          [](const Self& self, const PyExpr& e) {
            return self->arena->str(unwrap(self, e));
          })
      .def(
          "ask",
          [](const Self& self, const PyExpr& e, const std::string& fact) {
            auto f = fact_from_name(fact);
            TORCH_CHECK(f, "unknown assumption ", fact);
            return to_py(self->arena->ask(unwrap(self, e), *f));
          })
      .def(
          "diff",
          [wrap](const Self& self, const PyExpr& e, const PyExpr& x) {
            const Expr* s = unwrap(self, x);
            TORCH_CHECK(s->kind == Kind::Symbol, "diff needs a symbol");
            return wrap(self, self->arena->diff(unwrap(self, e), s));
          })
      .def(
          "as_numer_denom",
          [wrap](const Self& self, const PyExpr& e) {
            auto [n, d] = self->arena->as_numer_denom(unwrap(self, e));
            return std::make_pair(wrap(self, n), wrap(self, d));
          })
      .def(
          "is_polynomial",
          [](const Self& self, const PyExpr& e) {
            return self->arena->is_polynomial(unwrap(self, e));
          })
      .def(
          "monotonic_sign",
          [wrap](const Self& self, const PyExpr& e) -> std::optional<PyExpr> {
            const Expr* r = self->arena->monotonic_sign(unwrap(self, e));
            if (r == nullptr) {
              return std::nullopt;
            }
            return wrap(self, r);
          });
  arena_cls.def(
      "xreplace",
      [wrap](
          const Self& self,
          const PyExpr& e,
          const std::vector<std::pair<PyExpr, PyExpr>>& rule) {
        std::vector<std::pair<const Expr*, const Expr*>> reps;
        for (const auto& [old, rep] : rule) {
          reps.emplace_back(unwrap(self, old), unwrap(self, rep));
        }
        return wrap(self, self->arena->xreplace(unwrap(self, e), reps));
      });
  auto wrap_opt = [wrap](const Self& self, const Expr* e) {
    return e ? std::optional<PyExpr>(wrap(self, e)) : std::nullopt;
  };
  py::class_<PyShapeEnv>(sm, "NativeShapeEnv")
      .def(
          py::init<Self, const py::object&>(),
          py::arg("arena"),
          py::arg("shape_env") = py::none())
      .def_property_readonly(
          "arena", [](const PyShapeEnv& self) { return self.owner; })
      .def_property_readonly(
          "pristine", [](const PyShapeEnv& self) { return self.env->pristine(); })
      .def(
          "add_symbol",
          [](PyShapeEnv& self,
             const PyExpr& sym,
             std::optional<int64_t> hint,
             const PyExpr& lower,
             const PyExpr& upper,
             bool size_like) {
            auto lock = lock_env(*self.env);
            self.env->add_symbol(
                unwrap(self.owner, sym),
                hint,
                ValueRanges(unwrap(self.owner, lower), unwrap(self.owner, upper)),
                size_like);
          })
      .def(
          "mirror_symbol",
          [](PyShapeEnv& self,
             py::handle sym,
             py::handle hint,
             py::handle lower,
             py::handle upper) {
            // Takes sympy objects. A symbol that is not representable stays
            // unmirrored, so every query mentioning it delegates.
            PyArena& a = *self.owner;
            auto lock = lock_env(*self.env);
            try {
              self.env->add_symbol(
                  a.from_sympy(sym),
                  to_int64(hint),
                  ValueRanges(a.from_sympy(lower), a.from_sympy(upper)),
                  /*size_like=*/false);
            } catch (const NativeUnsupported&) {
            }
          })
      .def(
          "mirrored",
          [](PyShapeEnv& self, py::handle sym) -> py::object {
            std::optional<std::tuple<std::optional<int64_t>, ValueRanges, bool>>
                m;
            auto lock = lock_env(*self.env);
            try {
              m = self.env->mirrored(self.owner->from_sympy(sym));
            } catch (const NativeUnsupported&) {
            }
            if (!m) {
              return py::none();
            }
            auto& [hint, range, size_like] = *m;
            return py::make_tuple(
                hint,
                self.owner->to_sympy(range.lower),
                self.owner->to_sympy(range.upper),
                size_like);
          })
      .def_property_readonly(
          "replacements_empty",
          [](const PyShapeEnv& self) { return self.env->replacements_empty(); })
      // A copied or unpickled ShapeEnv gets no native env.
      .def(
          "__deepcopy__",
          [](const PyShapeEnv&, const py::dict&) { return py::none(); })
      .def(
          "__reduce__",
          [](const PyShapeEnv&) {
            return py::make_tuple(
                py::type::of(py::none()), py::tuple());
          })
      .def(
          "update_range",
          [](PyShapeEnv& self,
             const PyExpr& sym,
             const PyExpr& lower,
             const PyExpr& upper) {
            auto lock = lock_env(*self.env);
            flush_mod_memo(self);
            self.env->update_range(
                unwrap(self.owner, sym),
                ValueRanges(unwrap(self.owner, lower), unwrap(self.owner, upper)));
          })
      .def(
          "mark_not_pristine",
          [](PyShapeEnv& self) {
            auto lock = lock_env(*self.env);
            flush_mod_memo(self);
            self.env->mark_not_pristine();
          })
      .def(
          "mark_replacements",
          [](PyShapeEnv& self) {
            auto lock = lock_env(*self.env);
            flush_mod_memo(self);
            self.env->mark_replacements();
          })
      .def(
          "flush_mod_memo",
          [](PyShapeEnv& self) {
            auto lock = lock_env(*self.env);
            flush_mod_memo(self);
          })
      .def(
          "simplify",
          [wrap](PyShapeEnv& self, const PyExpr& e) {
            auto lock = lock_env(*self.env);
            return wrap(self.owner, self.env->simplify(unwrap(self.owner, e)));
          })
      .def(
          "maybe_evaluate_static",
          [wrap_opt](PyShapeEnv& self, const PyExpr& e) {
            auto lock = lock_env(*self.env);
            return wrap_opt(
                self.owner,
                self.env->maybe_evaluate_static(unwrap(self.owner, e)));
          })
      .def(
          "static_eval",
          [wrap_opt](PyShapeEnv& self, const PyExpr& e) {
            auto lock = lock_env(*self.env);
            auto r = self.env->static_eval(unwrap(self.owner, e));
            return std::make_pair(
                r.has_value(), wrap_opt(self.owner, r.value_or(nullptr)));
          })
      .def(
          "evaluate_expr",
          [wrap_opt](
              PyShapeEnv& self,
              const PyExpr& e,
              py::handle hint,
              std::optional<bool> fallback_value) {
            // Only None, int and bool hints are ported.
            auto h = hint_from_py(hint);
            if (!h || !native_config_is_default()) {
              return std::optional<PyExpr>();
            }
            auto lock = lock_env(*self.env);
            auto r = self.env->evaluate_expr(
                unwrap(self.owner, e), *h, fallback_value);
            return wrap_opt(self.owner, r.value_or(nullptr));
          },
          py::arg("e"),
          py::arg("hint") = py::none(),
          py::arg("fallback_value") = py::none())
      .def(
          "make_node",
          [](PyShapeEnv& self,
             py::handle expr,
             py::handle pytype,
             py::handle hint) {
            bool is_bool = pytype.is(pytype_to_py(PyType::Bool));
            TORCH_CHECK(
                is_bool || pytype.is(pytype_to_py(PyType::Int)),
                "native nodes are int or bool");
            TORCH_CHECK(
                hint.is_none() ||
                    (is_bool ? PyBool_Check(hint.ptr())
                             : PyLong_CheckExact(hint.ptr())),
                "hint must be None or of the pytype");
            auto h = hint_from_py(hint);
            if (!h) {
              throw NativeUnsupported("hint overflows int64");
            }
            auto lock = lock_env(*self.env);
            const Expr* e = self.owner->from_sympy(expr);
            if (e->has_float) {
              throw NativeUnsupported("Float in a native node");
            }
            return c10::make_intrusive<NativeSymNodeImpl>(
                self.env, e, is_bool ? PyType::Bool : PyType::Int, *h);
          })
      .def(
          "take_queries",
          [](PyShapeEnv& self) {
            // (seq, expr, evaluate, hint, fallback_value, suppress_guards,
            // result) with sympy exprs.
            PyArena& a = *self.owner;
            py::list out;
            auto lock = lock_env(*self.env);
            for (const auto& [seq, q, result] : self.env->take_queries()) {
              out.append(py::make_tuple(
                  seq,
                  a.to_sympy(q.expr),
                  q.evaluate,
                  hint_to_py(q.hint),
                  q.fallback_value,
                  q.suppress_guards,
                  result ? a.to_sympy(result) : py::object(py::none())));
            }
            return out;
          });
  for (auto [name, fn] :
       {std::pair{"as_ordered_terms", &ExprArena::as_ordered_terms},
        std::pair{"as_ordered_factors", &ExprArena::as_ordered_factors}}) {
    arena_cls.def(name, [wrap, fn](const Self& self, const PyExpr& e) {
      std::vector<PyExpr> r;
      for (const Expr* a : (self->arena.get()->*fn)(unwrap(self, e))) {
        r.push_back(wrap(self, a));
      }
      return r;
    });
  }
  for (auto [name, fn] :
       {std::pair{"canonical", &ExprArena::canonical},
        std::pair{"reversed", &ExprArena::reversed},
        std::pair{"reversedsign", &ExprArena::reversedsign},
        std::pair{"negated", &ExprArena::negated},
        std::pair{"weak", &ExprArena::weak},
        std::pair{"strict", &ExprArena::strict},
        std::pair{"safe_expand", &ExprArena::safe_expand},
        std::pair{"canonicalize_bool_expr", &ExprArena::canonicalize_bool_expr}}) {
    arena_cls.def(name, [wrap, fn](const Self& self, const PyExpr& r) {
      return wrap(self, (self->arena.get()->*fn)(unwrap(self, r)));
    });
  }

  // Unlike the _SymNode methods it overrides, these take and return Python
  // SymNodes as well.
  py::class_<
      NativeSymNodeImpl,
      c10::SymNodeImpl,
      c10::intrusive_ptr<NativeSymNodeImpl>>
      node_cls(sm, "_NativeSymNode");
  node_cls
      .def_property_readonly(
          "_expr", [](const NativeSymNodeImpl& n) { return node_expr(n); })
      .def_property_readonly(
          "expr",
          [](const NativeSymNodeImpl& n) -> py::object {
            bool replaced = false;
            {
              auto lock = lock_env(*n.env());
              replaced = !n.env()->replacements_empty();
            }
            if (!replaced) {
              return node_expr(n);
            }
            return node_to_py(materialize(n)).attr("expr");
          })
      .def_property_readonly(
          "_expr_is_number",
          [](const NativeSymNodeImpl& n) {
            // Boolean kinds are not Exprs (Basic.is_number is False), and no
            // numeric kind has a Boolean arg.
            return expr_query(
                n, "is_number", [](const ExprArena& a, const Expr* e) {
                  return !e->is_boolean() && a.free_symbols(e).empty();
                });
          })
      .def_property_readonly(
          "_expr_is_Boolean",
          [](const NativeSymNodeImpl& n) {
            // BooleanAtom and BooleanFunction; Relational is not.
            return expr_query(
                n, "is_Boolean", [](const ExprArena& /*a*/, const Expr* e) {
                  return e->is_boolean() && !e->is_relational();
                });
          })
      .def_property_readonly(
          "hint",
          [](const NativeSymNodeImpl& n) { return hint_to_py(n.hint()); })
      .def_property_readonly(
          "_hint",
          [](const NativeSymNodeImpl& n) { return hint_to_py(n.hint()); })
      .def_property_readonly(
          "shape_env",
          [](const NativeSymNodeImpl& n) -> py::object {
            const py::object& ref = binding_of(*n.env()).shape_env;
            return ref.is_none() ? py::none() : ref();
          })
      // Native envs exist only without translation validation.
      .def_property_readonly(
          "fx_node", [](const NativeSymNodeImpl& /*n*/) { return false; })
      .def_property_readonly(
          "constant",
          [](const NativeSymNodeImpl& n) { return hint_to_py(n.constant()); })
      .def_property_readonly(
          "pytype",
          [](const NativeSymNodeImpl& n) { return pytype_to_py(n.pytype()); })
      .def_property_readonly(
          "_optimized_summation", &NativeSymNodeImpl::optimized_summation)
      .def("maybe_as_int", &NativeSymNodeImpl::maybe_as_int)
      .def("str", &NativeSymNodeImpl::str)
      .def("statically_known_true", &NativeSymNodeImpl::statically_known_true)
      .def(
          "wrap_float",
          [](NativeSymNodeImpl& self, double v) {
            return node_to_py(self.wrap_float(v));
          })
      // Copies and pickles are Python SymNodes. _SymNode.__deepcopy__ clones.
      .def(
          "__deepcopy__",
          [](const NativeSymNodeImpl& self, py::handle memo) {
            return py::module_::import("copy").attr("deepcopy")(
                node_to_py(materialize(self)), memo);
          })
      .def(
          "__reduce_ex__",
          [](const NativeSymNodeImpl& self, py::handle /*protocol*/) {
            return py::make_tuple(
                py::module_::import("copy").attr("copy"),
                py::make_tuple(node_to_py(materialize(self))));
          })
      // Every other SymNode method runs its Python impl with the native node
      // as self; those read only the attributes bound here.
      .def("__getattr__", [](py::handle self, const std::string& name) {
        py::handle cls = python_symnode_class();
        py::object attr = cls.attr("__dict__").attr("get")(name);
        bool dunder = name.size() > 4 && name.rfind("__", 0) == 0 &&
            name.compare(name.size() - 2, 2, "__") == 0;
        if (attr.is_none() || dunder) {
          // @allow-raw-throw: getattr and hasattr need an AttributeError
          throw py::attribute_error(
              "'_NativeSymNode' object has no attribute '" + name + "'");
        }
        if (py::hasattr(attr, "__get__")) {
          return attr.attr("__get__")(self, cls);
        }
        return attr;
      })
      .def("__repr__", [](py::handle self) {
        return python_symnode_class().attr("__repr__")(self);
      });
  for (auto [name, fn] :
       {std::pair{"add", &c10::SymNodeImpl::add},
        std::pair{"sub", &c10::SymNodeImpl::sub},
        std::pair{"mul", &c10::SymNodeImpl::mul},
        std::pair{"truediv", &c10::SymNodeImpl::truediv},
        std::pair{"float_truediv", &c10::SymNodeImpl::float_truediv},
        std::pair{"int_truediv", &c10::SymNodeImpl::int_truediv},
        std::pair{"pow", &c10::SymNodeImpl::pow},
        std::pair{"float_pow", &c10::SymNodeImpl::float_pow},
        std::pair{"pow_by_natural", &c10::SymNodeImpl::pow_by_natural},
        std::pair{"floordiv", &c10::SymNodeImpl::floordiv},
        std::pair{"int_floordiv", &c10::SymNodeImpl::int_floordiv},
        std::pair{"mod", &c10::SymNodeImpl::mod},
        std::pair{"eq", &c10::SymNodeImpl::eq},
        std::pair{"ne", &c10::SymNodeImpl::ne},
        std::pair{"gt", &c10::SymNodeImpl::gt},
        std::pair{"lt", &c10::SymNodeImpl::lt},
        std::pair{"le", &c10::SymNodeImpl::le},
        std::pair{"ge", &c10::SymNodeImpl::ge},
        std::pair{"sym_min", &c10::SymNodeImpl::sym_min},
        std::pair{"sym_max", &c10::SymNodeImpl::sym_max},
        std::pair{"sym_and", &c10::SymNodeImpl::sym_and},
        std::pair{"sym_or", &c10::SymNodeImpl::sym_or},
        std::pair{"and_", &c10::SymNodeImpl::sym_and},
        std::pair{"or_", &c10::SymNodeImpl::sym_or}}) {
    node_cls.def(
        name,
        [fn](
            NativeSymNodeImpl& self, py::handle other) {
          return node_to_py((self.*fn)(node_from_py(other)));
        });
  }
  for (auto [name, fn] :
       {std::pair{"neg", &c10::SymNodeImpl::neg},
        std::pair{"sym_not", &c10::SymNodeImpl::sym_not},
        std::pair{"ceil", &c10::SymNodeImpl::ceil},
        std::pair{"floor", &c10::SymNodeImpl::floor},
        std::pair{"sym_float", &c10::SymNodeImpl::sym_float}}) {
    node_cls.def(name, [fn](NativeSymNodeImpl& self) {
      return node_to_py((self.*fn)());
    });
  }
  for (auto [name, fn] :
       {std::pair{"is_contiguous", &c10::SymNodeImpl::is_contiguous},
        std::pair{
            "is_channels_last_contiguous_2d",
            &c10::SymNodeImpl::is_channels_last_contiguous_2d},
        std::pair{
            "is_channels_last_contiguous_3d",
            &c10::SymNodeImpl::is_channels_last_contiguous_3d},
        std::pair{
            "is_channels_last_strides_2d",
            &c10::SymNodeImpl::is_channels_last_strides_2d},
        std::pair{
            "is_channels_last_strides_3d",
            &c10::SymNodeImpl::is_channels_last_strides_3d},
        std::pair{
            "is_non_overlapping_and_dense",
            &c10::SymNodeImpl::is_non_overlapping_and_dense}}) {
    node_cls.def(
        name,
        [fn](
            NativeSymNodeImpl& self,
            const py::sequence& sizes,
            const py::sequence& strides) {
          return node_to_py(
              (self.*fn)(nodes_from_py(sizes), nodes_from_py(strides)));
        });
  }
  node_cls.def(
      "is_non_overlapping_and_dense_indicator",
      [](NativeSymNodeImpl& self,
         const py::sequence& sizes,
         const py::sequence& strides) {
        return node_to_py(self.is_non_overlapping_and_dense_indicator(
            nodes_from_py(sizes), nodes_from_py(strides)));
      });
  node_cls.def(
      "sym_sum", [](NativeSymNodeImpl& self, const py::sequence& args) {
        return node_to_py(self.sym_sum(nodes_from_py(args)));
      });
  node_cls.def(
      "sym_ite",
      [](NativeSymNodeImpl& self, py::handle then_val, py::handle else_val) {
        return node_to_py(
            self.sym_ite(node_from_py(then_val), node_from_py(else_val)));
      });
}

} // namespace torch::symbolic
