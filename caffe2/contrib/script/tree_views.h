#pragma once
#include "error_report.h"
#include "tree.h"

namespace caffe2 {
namespace script {

// TreeView provides a statically-typed way to access the members of a TreeRef
// instead of using TK_MATCH

struct TreeView {
  explicit TreeView(const TreeRef& tree_) : tree_(tree_) {}
  TreeRef tree() const {
    return tree_;
  }
  const SourceRange& range() const {
    return tree_->range();
  }
  operator TreeRef() const {
    return tree_;
  }

 protected:
  TreeRef tree_;
};

template <typename T>
struct ListViewIterator {
  ListViewIterator(TreeList::const_iterator it) : it(it) {}
  bool operator!=(const ListViewIterator& rhs) const {
    return it != rhs.it;
  }
  T operator*() const {
    return T(*it);
  }
  void operator++() {
    ++it;
  }
  void operator--() {
    --it;
  }

 private:
  TreeList::const_iterator it;
};

template <typename T>
struct ListView : public TreeView {
  ListView(const TreeRef& tree) : TreeView(tree) {
    tree->match(TK_LIST);
  }
  typedef ListViewIterator<T> iterator;
  typedef ListViewIterator<T> const_iterator;
  iterator begin() const {
    return iterator(tree_->trees().begin());
  }
  iterator end() const {
    return iterator(tree_->trees().end());
  }
  T operator[](size_t i) const {
    return T(tree_->trees().at(i));
  }
  TreeRef map(std::function<TreeRef(const T&)> fn) {
    return tree_->map([&](TreeRef v) { return fn(T(v)); });
  }
  size_t size() const {
    return tree_->trees().size();
  }
};

template <typename T>
struct OptionView : public TreeView {
  explicit OptionView(const TreeRef& tree) : TreeView(tree) {
    C2S_ASSERT(tree, tree->kind() == TK_OPTION);
  }
  bool present() const {
    return tree_->trees().size() > 0;
  }
  T get() const {
    C2S_ASSERT(tree_, present());
    return T(tree_->trees()[0]);
  }
  TreeRef map(std::function<TreeRef(const T&)> fn) {
    return tree_->map([&](TreeRef v) { return fn(T(v)); });
  }
};

struct Ident : public TreeView {
  // each subclass of TreeView provides:
  // 1. a constructor that takes a TreeRef, and matches it to the right type.
  explicit Ident(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_IDENT, name_);
  }
  // 2. accessors that get underlying information out of the object
  // in this case, we return the name of the identifier, and handle the
  // converstion to a string in the method
  const std::string& name() const {
    return name_->stringValue();
  }

  // 3. a static method 'create' that creates the underlying TreeRef object
  // for every TreeRef kind that has a TreeView, the parser always uses
  // (e.g.) Ident::create rather than Compound::Create, this means that
  // changes to the structure of Ident are always made right here rather
  // than both in the parser and in this code
  static TreeRef create(const SourceRange& range, const std::string& name) {
    return Compound::create(TK_IDENT, range, {String::create(name)});
  }

 private:
  TreeRef name_;
};

struct Attribute : public TreeView {
  explicit Attribute(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_ATTRIBUTE, name_, value_);
  }
  Ident name() const {
    return Ident(name_);
  }
  TreeRef value() const {
    return value_;
  }
  static TreeRef create(const SourceRange& range, TreeRef name, TreeRef value) {
    return Compound::create(TK_ATTRIBUTE, range, {name, value});
  }

 private:
  TreeRef name_;
  TreeRef value_;
};

struct Apply : public TreeView {
  explicit Apply(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_APPLY, name_, inputs_, attributes_);
  }

  Ident name() const {
    return Ident(name_);
  }
  ListView<TreeRef> inputs() const {
    return ListView<TreeRef>(inputs_);
  }
  ListView<Attribute> attributes() const {
    return ListView<Attribute>(attributes_);
  }

  static TreeRef create(
      const SourceRange& range,
      TreeRef name,
      TreeRef inputs,
      TreeRef attributes) {
    return Compound::create(TK_APPLY, range, {name, inputs, attributes});
  }

 private:
  TreeRef name_;
  TreeRef inputs_;
  TreeRef attributes_;
};

struct TensorType : public TreeView {
  explicit TensorType(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_TENSOR_TYPE, scalar_type_, dims_);
  }
  static TreeRef
  create(const SourceRange& range, TreeRef scalar_type_, TreeRef dims_) {
    return Compound::create(TK_TENSOR_TYPE, range, {scalar_type_, dims_});
  }
  int scalarType() const {
    if (scalar_type_->kind() == TK_IDENT)
      throw ErrorReport(tree_)
          << " TensorType has a symbolic ident " << Ident(scalar_type_).name()
          << " rather than a concrete type";
    return scalar_type_->kind();
  }
  ListView<Ident> dims() const {
    return ListView<Ident>(dims_);
  }

 private:
  TreeRef scalar_type_;
  TreeRef dims_;
};

struct Param : public TreeView {
  explicit Param(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_PARAM, ident_, type_);
  }
  static TreeRef create(const SourceRange& range, TreeRef ident, TreeRef type) {
    return Compound::create(TK_PARAM, range, {ident, type});
  }
  // when the type of a field is statically know the accessors return
  // the wrapped type. for instance here we know ident_ is an identifier
  // so the accessor returns an Ident
  // this means that clients can do p.ident().name() to get the name of the
  // parameter.
  Ident ident() const {
    return Ident(ident_);
  }
  // may be TensorType or TK_INFERRED
  TreeRef type() const {
    return type_;
  }
  bool typeIsInferred() const {
    return type_->kind() == TK_INFERRED;
  }
  // helper for when you know the type is not inferred.
  TensorType tensorType() const {
    return TensorType(type_);
  }

 private:
  TreeRef ident_;
  TreeRef type_;
};

struct Assign : public TreeView {
  explicit Assign(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_ASSIGN, idents_, reduction_, rhs_);
  }
  static TreeRef create(
      const SourceRange& range,
      TreeRef idents,
      TreeRef reduction,
      TreeRef rhs) {
    return Compound::create(TK_ASSIGN, range, {idents, reduction, rhs});
  }
  // when the type of a field is statically know the accessors return
  // the wrapped type. for instance here we know ident_ is an identifier
  // so the accessor returns an Ident
  // this means that clients can do p.ident().name() to get the name of the
  // parameter.
  ListView<Ident> idents() const {
    return ListView<Ident>(idents_);
  }
  const std::string& reduction() const {
    return reduction_->stringValue();
  }
  TreeRef rhs() const {
    return rhs_;
  }

 private:
  TreeRef idents_;
  TreeRef reduction_;
  TreeRef rhs_;
};

struct Def : public TreeView {
  explicit Def(const TreeRef& tree) : TreeView(tree) {
    tree->match(TK_DEF, name_, paramlist, retlist, stmts_list);
  }
  Ident name() const {
    return Ident(name_);
  }
  // ListView helps turn TK_LISTs into vectors of TreeViews
  // so that we can, e.g., return lists of parameters
  ListView<Param> params() const {
    return ListView<Param>(paramlist);
  }
  ListView<Param> returns() const {
    return ListView<Param>(retlist);
  }
  ListView<TreeRef> statements() const {
    return ListView<TreeRef>(stmts_list);
  }
  static TreeRef create(
      const SourceRange& range,
      TreeRef name,
      TreeRef paramlist,
      TreeRef retlist,
      TreeRef stmts_list) {
    return Compound::create(
        TK_DEF, range, {name, paramlist, retlist, stmts_list});
  }

 private:
  TreeRef name_;
  TreeRef paramlist;
  TreeRef retlist;
  TreeRef stmts_list;
};

struct Select : public TreeView {
  explicit Select(const TreeRef& tree) : TreeView(tree) {
    tree_->match('.', name_, index_);
  }
  Ident name() const {
    return Ident(name_);
  }
  int index() const {
    return index_->doubleValue();
  }
  static TreeRef create(const SourceRange& range, TreeRef name, TreeRef index) {
    return Compound::create('.', range, {name, index});
  }

 private:
  TreeRef name_;
  TreeRef index_;
};

struct If : public TreeView {
  explicit If(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_IF, cond_, true_branch_, false_branch_);
  }
  const TreeRef& cond() const {
    return cond_;
  }
  ListView<TreeRef> trueBranch() const {
    return ListView<TreeRef>(true_branch_);
  }
  ListView<TreeRef> falseBranch() const {
    return ListView<TreeRef>(false_branch_);
  }

  static TreeRef create(
      const SourceRange& range,
      TreeRef cond_,
      TreeRef true_branch_,
      TreeRef false_branch_) {
    return Compound::create(TK_IF, range, {cond_, true_branch_, false_branch_});
  }

 private:
  TreeRef cond_;
  TreeRef true_branch_;
  TreeRef false_branch_;
};

struct While : public TreeView {
  explicit While(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_WHILE, cond_, body_);
  }
  const TreeRef& cond() const {
    return cond_;
  }
  ListView<TreeRef> body() const {
    return ListView<TreeRef>(body_);
  }

  static TreeRef
  create(const SourceRange& range, TreeRef cond_, TreeRef body_) {
    return Compound::create(TK_WHILE, range, {cond_, body_});
  }

 private:
  TreeRef cond_;
  TreeRef body_;
};

} // namespace script
} // namespace caffe2
