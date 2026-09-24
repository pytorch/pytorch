#include <torch/csrc/symbolic/python_symbolic.h>

#include <torch/csrc/symbolic/Expr.h>
#include <torch/csrc/symbolic/NativeShapeEnv.h>
#include <torch/csrc/symbolic/NativeSymNodeImpl.h>
#include <torch/csrc/symbolic/PyFallback.h>
#include <torch/csrc/symbolic/ValueRanges.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_symnode.h>

#include <algorithm>
#include <array>
#include <optional>
#include <unordered_map>
#include <vector>

namespace torch::symbolic {

namespace py = pybind11;

namespace {

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

  py::object Integer, Rational, Symbol, Dummy, Add, Mul, Pow, IntInfinity,
      NegativeIntInfinity, int_oo, true_, false_, Not, And, Or;
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

py::object hint_to_py(const Hint& h) {
  return std::visit(
      [](auto v) -> py::object {
        if constexpr (std::is_same_v<decltype(v), std::monostate>) {
          return py::none();
        } else {
          return py::cast(v);
        }
      },
      h);
}

const Expr* PyArena::from_sympy(py::handle obj) {
  if (PyLong_CheckExact(obj.ptr()) || py::isinstance(obj, Integer)) {
    py::int_ v(py::reinterpret_borrow<py::object>(obj));
    return arena->integer(to_int64(v));
  }
  if (py::isinstance(obj, Rational)) {
    return arena->rational(to_int64(obj.attr("p")), to_int64(obj.attr("q")));
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

// Whether a torch.fx.experimental._config entry is falsy, read the way
// ConfigModule.__getattr__ resolves an entry without alias or justknob, but
// without running Python code. Other entries count as set.
bool config_entry_unset(py::handle entry, py::handle unset) {
  if (!entry.attr("alias").is_none() || !entry.attr("justknob").is_none() ||
      py::bool_(entry.attr("hide"))) {
    return false;
  }
  py::object v = entry.attr("env_value_force");
  if (v.is(unset)) {
    PyObject* override_value = nullptr;
    if (PyContextVar_Get(
            entry.attr("user_override").ptr(), nullptr, &override_value) != 0) {
      throw py::error_already_set();
    }
    v = py::reinterpret_steal<py::object>(override_value);
  }
  if (v.is(unset)) {
    v = entry.attr("env_value_default");
  }
  if (v.is(unset)) {
    v = entry.attr("default");
  }
  return !py::bool_(v);
}

// The config that ShapeEnv reads on every evaluation and that a native env
// requires unset: backed_size_oblivious and aggressive_guard_free_semantics.
bool native_config_is_default() {
  static const auto* entries = [] {
    py::dict config =
        py::module_::import("torch.fx.experimental._config").attr("_config");
    return new std::array<py::object, 3>{
        config["backed_size_oblivious"],
        config["aggressive_guard_free_semantics"],
        py::module_::import("torch.utils._config_module")
            .attr("_UNSET_SENTINEL")};
  }();
  const auto& [backed, aggressive, unset] = *entries;
  return config_entry_unset(backed, unset) &&
      config_entry_unset(aggressive, unset);
}

EnvBinding& binding_of(NativeShapeEnv& env) {
  auto* b = static_cast<EnvBinding*>(env.binding());
  TORCH_CHECK(b != nullptr, "native env without a Python binding");
  return *b;
}

py::object pytype_to_py(PyType t) {
  return py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(
      t == PyType::Int ? &PyLong_Type : &PyBool_Type));
}

py::object node_expr(const NativeSymNodeImpl& node) {
  NativeShapeEnv& env = *node.env();
  auto lock = lock_env(env);
  return binding_of(env).arena->to_sympy(node.expr());
}

} // namespace

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

void initSymbolicBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module_>();
  auto sm = m.def_submodule("_symbolic", "native symbolic expressions");
  py::register_exception<NativeUnsupported>(sm, "NativeUnsupported");
  sm.def("_assume_rules", &assume_rules_to_py);
  sm.def("_native_config_is_default", &native_config_is_default);
  sm.def("_native_queries_pending", &NativeShapeEnv::queries_pending);
  sm.def("_set_suppress_guards", [](bool v) { suppress_guards_tls() = v; });
  sm.def("_suppress_guards", [] { return suppress_guards_tls(); });

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
            self.env->update_range(
                unwrap(self.owner, sym),
                ValueRanges(unwrap(self.owner, lower), unwrap(self.owner, upper)));
          })
      .def(
          "mark_not_pristine",
          [](PyShapeEnv& self) {
            auto lock = lock_env(*self.env);
            self.env->mark_not_pristine();
          })
      .def(
          "mark_replacements",
          [](PyShapeEnv& self) {
            auto lock = lock_env(*self.env);
            self.env->mark_replacements();
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
            auto h = hint_from_py(hint);
            TORCH_CHECK(
                h && (hint.is_none() || PyBool_Check(hint.ptr()) == is_bool),
                "hint must be None or of the pytype");
            auto lock = lock_env(*self.env);
            return c10::make_intrusive<NativeSymNodeImpl>(
                self.env,
                self.owner->from_sympy(expr),
                is_bool ? PyType::Bool : PyType::Int,
                *h);
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
  auto node_from_py = [](py::handle obj) -> c10::SymNode {
    if (py::isinstance<c10::SymNodeImpl>(obj)) {
      return py::cast<c10::SymNode>(obj);
    }
    return c10::make_intrusive<impl::PythonSymNodeImpl>(
        py::reinterpret_borrow<py::object>(obj));
  };
  auto node_to_py = [](const c10::SymNode& n) -> py::object {
    if (auto* p = dynamic_cast<impl::PythonSymNodeImpl*>(n.get())) {
      return py::reinterpret_borrow<py::object>(p->getPyObj());
    }
    return py::cast(n);
  };
  py::class_<
      NativeSymNodeImpl,
      c10::SymNodeImpl,
      c10::intrusive_ptr<NativeSymNodeImpl>>
      node_cls(sm, "_NativeSymNode");
  node_cls
      .def_property_readonly(
          "_expr", [](const NativeSymNodeImpl& n) { return node_expr(n); })
      .def_property_readonly(
          "hint",
          [](const NativeSymNodeImpl& n) { return hint_to_py(n.hint()); })
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
      .def("wrap_float", [node_to_py](NativeSymNodeImpl& self, double v) {
        return node_to_py(self.wrap_float(v));
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
        [fn, node_from_py, node_to_py](
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
    node_cls.def(name, [fn, node_to_py](NativeSymNodeImpl& self) {
      return node_to_py((self.*fn)());
    });
  }
}

} // namespace torch::symbolic
