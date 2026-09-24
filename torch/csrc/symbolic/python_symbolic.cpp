#include <torch/csrc/symbolic/python_symbolic.h>

#include <torch/csrc/symbolic/Expr.h>
#include <torch/csrc/utils/pybind.h>

#include <algorithm>
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
    Kind::CleanDiv};

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
      const Expr* r = arena->function(kFunctionKinds[i], args);
      if (r->kind != kFunctionKinds[i] ||
          !std::equal(r->args.begin(), r->args.end(), args.begin(), args.end())) {
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
    case Kind::CleanDiv: {
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

} // namespace

void initSymbolicBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module_>();
  auto sm = m.def_submodule("_symbolic", "native symbolic expressions");
  py::register_exception<NativeUnsupported>(sm, "NativeUnsupported");
  sm.def("_assume_rules", &assume_rules_to_py);

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
        std::pair{"strict", &ExprArena::strict}}) {
    arena_cls.def(name, [wrap, fn](const Self& self, const PyExpr& r) {
      return wrap(self, (self->arena.get()->*fn)(unwrap(self, r)));
    });
  }
}

} // namespace torch::symbolic
