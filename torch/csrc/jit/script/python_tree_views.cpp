#include <torch/csrc/jit/script/python_tree_views.h>

#include <torch/csrc/jit/script/tree_views.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

namespace py = pybind11;

namespace torch {
namespace jit {
namespace script {

struct SourceRangeFactory {
  SourceRangeFactory(
      std::string text,
      std::string filename,
      size_t file_lineno,
      size_t leading_whitespace_chars)
      : source_(std::make_shared<Source>(
            std::move(text),
            std::move(filename),
            file_lineno)),
        leading_whitespace_chars_(leading_whitespace_chars) {}
  SourceRange create(int line, int start_col, int end_col) {
    size_t start_byte_offset, end_byte_offset;
    std::tie(start_byte_offset, end_byte_offset) =
        line_col_to_byte_offs(
            line,
            start_col + leading_whitespace_chars_,
            end_col + leading_whitespace_chars_);
    return SourceRange(source_, start_byte_offset, end_byte_offset);
  }

  std::tuple<size_t, size_t> line_col_to_byte_offs(
      int line,
      int start_col,
      int end_col) {
    // Python has a weird convention where col_offset points to the column
    // *before* the token starts.
    start_col++;
    end_col++;
    // Also, lines are counted from 1.
    line--;
    auto line_start = source_->offset_for_line(line);
    return std::make_tuple<size_t, size_t>(
        line_start + start_col, line_start + end_col);
  }

  std::shared_ptr<Source> source_;
  std::vector<size_t> line_len_prefix_sum_;
  size_t leading_whitespace_chars_;
};

template <typename T>
List<T> wrap_list(const SourceRange& fallback_pos, std::vector<T>&& vec) {
  if (vec.empty())
    return List<T>::create(fallback_pos, std::move(vec));
  return List<T>::create(vec.front().range(), std::move(vec));
}

template <typename T>
Maybe<T> wrap_maybe(const SourceRange& fallback_pos, T* val) {
  return val ? Maybe<T>::create(val->range(), *val)
             : Maybe<T>::create(fallback_pos);
}

void initTreeViewBindings(PyObject* module) {
  auto _C = py::handle(module).cast<py::module>();
  auto m = _C.def_submodule("_jit_tree_views");

  py::class_<SourceRange>(m, "SourceRange")
      .def(
          "highlight",
          [](const SourceRange& self) {
            std::ostringstream stream;
            self.highlight(stream);
            return stream.str();
          })
      .def_property_readonly("start", &SourceRange::start)
      .def_property_readonly("end", &SourceRange::end);
  py::class_<SourceRangeFactory>(m, "SourceRangeFactory")
      .def(py::init<std::string&&, std::string&&, size_t, size_t>())
      .def("make_range", &SourceRangeFactory::create)
      .def(
          "make_raw_range",
          [](const SourceRangeFactory& self, size_t start, size_t end) {
            return SourceRange(self.source_, start, end);
          })
      .def_property_readonly("source", [](const SourceRangeFactory& self) {
        return self.source_->text();
      });

  py::class_<TreeView>(m, "TreeView")
      .def("range", &TreeView::range)
      .def(
          "__str__",
          [](const TreeView& tree) {
            std::ostringstream stream;
            stream << tree.get();
            return stream.str();
          })
      .def("dump", [](const TreeView& tree) { tree.dump(); });

  py::class_<Ident, TreeView>(m, "Ident")
      .def(py::init(&Ident::create))
      .def_property_readonly(
          "name", [](const Ident& self) { return self.name(); });

  py::class_<Param, TreeView>(m, "Param")
      .def(py::init([](const Expr& type, const Ident& name, bool kwarg_only) {
        return Param::create(
            name.range(),
            name,
            Maybe<Expr>::create(type.range(), type),
            Maybe<Expr>::create(name.range()),
            kwarg_only);
      }))
      .def(py::init(
          [](const Maybe<Expr>& type, const Ident& name, bool kwarg_only) {
            return Param::create(
                name.range(),
                name,
                type,
                Maybe<Expr>::create(name.range()),
                kwarg_only);
          }));
  py::class_<Attribute, TreeView>(m, "Attribute")
      .def(py::init([](const Ident& name, const Expr& value) {
        return Attribute::create(name.range(), name, value);
      }));
  m.def("TrueLiteral", [](const SourceRange& range) {
    return Expr(Compound::create(TK_TRUE, range, {}));
  });
  m.def("FalseLiteral", [](const SourceRange& range) {
    return Expr(Compound::create(TK_FALSE, range, {}));
  });
  m.def("NoneLiteral", [](const SourceRange& range) {
    return Expr(Compound::create(TK_NONE, range, {}));
  });

  py::class_<Stmt, TreeView>(m, "Stmt"); // NOLINT(bugprone-unused-raii)
  py::class_<Expr, TreeView>(m, "Expr"); // NOLINT(bugprone-unused-raii)
  py::class_<Def, TreeView>(m, "Def").def(
      py::init([](const Ident& name, Decl decl, std::vector<Stmt> body) {
        const auto& r = name.range();
        return Def::create(r, name, decl, wrap_list(r, std::move(body)));
      }));
  py::class_<ClassDef, TreeView>(m, "ClassDef")
      .def(py::init([](const Ident& name, std::vector<Def> body) {
        const auto& r = name.range();
        return ClassDef::create(r, name, wrap_list(r, std::move(body)));
      }));
  py::class_<Decl, TreeView>(m, "Decl").def(py::init(
      [](const SourceRange& r, std::vector<Param> params, Expr* return_type) {
        return Decl::create(
            r, wrap_list(r, std::move(params)), wrap_maybe(r, return_type));
      }));

  py::class_<Assign, Stmt>(m, "Assign")
      .def(py::init([](const Expr& lhs, const Expr& rhs) {
        return Assign::create(
            lhs.range(), lhs, rhs, Maybe<Expr>::create(lhs.range()));
      }))
      .def(py::init([](const Expr& lhs, const Expr& rhs, Expr* type) {
        return Assign::create(
            lhs.range(), lhs, rhs, wrap_maybe(lhs.range(), type));
      }));
  py::class_<AugAssign, Stmt>(m, "AugAssign")
      .def(py::init([](const Expr& lhs, std::string kind_str, const Expr& rhs) {
        const auto& r = lhs.range();
        auto kind =
            AugAssignKind(Compound::create(stringToKind(kind_str), r, {}));
        return AugAssign::create(r, lhs, kind, rhs);
      }));
  py::class_<Return, Stmt>(m, "Return")
      .def(py::init([](const SourceRange& range, Expr* value) {
        return Return::create(
            range, value ? *value : Expr(Compound::create(TK_NONE, range, {})));
      }));
  py::class_<Raise, Stmt>(m, "Raise")
      .def(py::init([](const SourceRange& range, Expr* expr) {
        return Raise::create(range, wrap_maybe(range, expr));
      }));
  py::class_<Assert, Stmt>(m, "Assert")
      .def(py::init([](const SourceRange& range, const Expr& test, Expr* msg) {
        return Assert::create(range, test, wrap_maybe(range, msg));
      }));
  py::class_<Pass, Stmt>(m, "Pass").def(
      py::init([](const SourceRange& range) { return Pass::create(range); }));
  py::class_<Dots, Expr>(m, "Dots").def(
      py::init([](const SourceRange& range) { return Dots::create(range); }));
  py::class_<If, Stmt>(m, "If").def(
      py::init([](const SourceRange& range,
                  const Expr& cond,
                  std::vector<Stmt> true_branch,
                  std::vector<Stmt> false_branch) {
        return If::create(
            range,
            cond,
            wrap_list(range, std::move(true_branch)),
            wrap_list(range, std::move(false_branch)));
      }));
  py::class_<While, Stmt>(m, "While")
      .def(py::init([](const SourceRange& range,
                       const Expr& cond,
                       std::vector<Stmt> body) {
        return While::create(range, cond, wrap_list(range, std::move(body)));
      }));
  py::class_<For, Stmt>(m, "For").def(py::init([](const SourceRange range,
                                                  std::vector<Expr>& targets,
                                                  std::vector<Expr>& itrs,
                                                  std::vector<Stmt> body) {
    return For::create(
        range,
        wrap_list(range, std::move(targets)),
        wrap_list(range, std::move(itrs)),
        wrap_list(range, std::move(body)));
  }));
  py::class_<ExprStmt, Stmt>(m, "ExprStmt").def(py::init([](const Expr& expr) {
    return ExprStmt::create(expr.range(), expr);
  }));

  py::class_<Var, Expr>(m, "Var")
      .def(py::init(
          [](const Ident& name) { return Var::create(name.range(), name); }))
      .def_property_readonly("name", [](const Var& var) { return var.name(); });
  py::class_<BinOp, Expr>(m, "BinOp")
      .def(py::init([](std::string kind, const Expr& lhs, const Expr& rhs) {
        return BinOp::create(lhs.range(), stringToKind(kind), lhs, rhs);
      }));
  // NB: we take range here, because unary ops precede their exprs, so we need
  // to include them
  py::class_<UnaryOp, Expr>(m, "UnaryOp")
      .def(py::init(
          [](const SourceRange& range, std::string kind, const Expr& expr) {
            auto resolved_kind = stringToKind(kind);
            resolved_kind =
                resolved_kind == '-' ? TK_UNARY_MINUS : resolved_kind;
            return UnaryOp::create(range, resolved_kind, expr);
          }));
  py::class_<Const, Expr>(m, "Const")
      .def(py::init([](const SourceRange& range, std::string value) {
        return Const::create(range, value);
      }));
  py::class_<StringLiteral, Expr>(m, "StringLiteral")
      .def(py::init([](const SourceRange& range, std::string value) {
        return StringLiteral::create(range, value);
      }));
  py::class_<Apply, Expr>(m, "Apply")
      .def(py::init([](const Expr& expr,
                       std::vector<Expr> args,
                       std::vector<Attribute> kwargs) {
        const auto& r = expr.range();
        return Apply::create(
            expr.range(),
            expr,
            wrap_list(r, std::move(args)),
            wrap_list(r, std::move(kwargs)));
      }));
  py::class_<Select, Expr>(m, "Select")
      .def(py::init([](const Expr& expr, const Ident& field) {
        return Select::create(expr.range(), expr, field);
      }));
  py::class_<TernaryIf, Expr>(m, "TernaryIf")
      .def(py::init(
          [](const Expr& cond, const Expr& true_expr, const Expr& false_expr) {
            return TernaryIf::create(cond.range(), cond, true_expr, false_expr);
          }));
  py::class_<ListComp, Expr>(m, "ListComp")
      .def(py::init([](const SourceRange& range,
                       const Expr& elt,
                       const Expr& target,
                       const Expr& iter) {
        return ListComp::create(range, elt, target, iter);
      }));
  py::class_<ListLiteral, Expr>(m, "ListLiteral")
      .def(py::init([](const SourceRange& range, std::vector<Expr> args) {
        return ListLiteral::create(range, wrap_list(range, std::move(args)));
      }));
  py::class_<TupleLiteral, Expr>(m, "TupleLiteral")
      .def(py::init([](const SourceRange& range, std::vector<Expr> args) {
        return TupleLiteral::create(range, wrap_list(range, std::move(args)));
      }));
  py::class_<DictLiteral, Expr>(m, "DictLiteral")
      .def(py::init([](const SourceRange& range,
                       std::vector<Expr> keys,
                       std::vector<Expr> values) {
        return DictLiteral::create(
            range,
            wrap_list(range, std::move(keys)),
            wrap_list(range, std::move(values)));
      }));
  py::class_<Subscript, Expr>(m, "Subscript")
      .def(py::init([](const Expr& base, std::vector<Expr> subscript_exprs) {
        return Subscript::create(
            base.range(),
            base,
            wrap_list(base.range(), std::move(subscript_exprs)));
      }));
  py::class_<SliceExpr, Expr>(m, "SliceExpr")
      .def(py::init([](const SourceRange& range, Expr* lower, Expr* upper, Expr* step) {
        return SliceExpr::create(
            range, wrap_maybe(range, lower), wrap_maybe(range, upper), wrap_maybe(range, step));
      }));
  py::class_<Starred, Expr>(m, "Starred")
      .def(py::init([](const SourceRange& range, Expr expr) {
        return Starred::create(range, expr);
      }));
  py::class_<Maybe<Expr>, TreeView>(m, "EmptyTypeAnnotation")
      .def(py::init(
          [](const SourceRange& range) { return Maybe<Expr>::create(range); }));
}

} // namespace script
} // namespace jit
} // namespace torch
