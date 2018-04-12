#include "VariableType.h"

// ${generated_comment}

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/autograd/generated/Functions.h"
#include "torch/csrc/autograd/functions/tensor.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/jit/tracer.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/jit/tensor_conversions.h"
#include "torch/csrc/utils/variadic.h"

#include <initializer_list>
#include <iostream>
#include <functional>
#include <cstddef>

#ifdef _MSC_VER
#ifdef Type
#undef Type
#endif
#endif

using namespace at;
using namespace torch::autograd::generated;

namespace torch { namespace autograd {
// Helper methods for working with Attributes (torch/csrc/jit/attributes.h)

// The overloaded accessors are convenient for the generated code (since we
// don't want to make the codegen do the dispatch manually)
static void setattr(jit::Node* n, jit::Symbol name, int64_t v)             { n->i_(name, v); }
static void setattr(jit::Node* n, jit::Symbol name, const at::Scalar& v)   { n->t_(name, v.toTensor()); }
static void setattr(jit::Node* n, jit::Symbol name, SparseTensor s)        { n->t_(name, s.tref); }
static void setattr(jit::Node* n, jit::Symbol name, const at::IntList& v)  { n->is_(name, v); }
static void setattr(jit::Node* n, jit::Symbol name, bool v)                { n->i_(name, v); }
static void setattr(jit::Node* n, jit::Symbol name, double v)              { n->f_(name, v); }
static void setattr(jit::Node* n, jit::Symbol name, std::string v)         { n->s_(name, v); }
template<std::size_t N>
static void setattr(jit::Node* n, jit::Symbol name, std::array<bool, N> v) { n->is_(name, std::vector<int64_t>(v.begin(), v.end())); }

template<typename T>
static jit::Value* createConstant(jit::Node* n, T value) {
  return n->owningGraph()->createConstant(jit::as_tensor(value))->insertBefore(n)->output();
}

template<typename T>
static void genericInsertInput(jit::Node* n, size_t idx, T value) {
  n->insertInput(idx, createConstant(n, t));
}

void failPosAttr() {
  throw std::runtime_error("unsupported type in setposattr. File a bug report!");
}

static void setposattr(jit::Node* n, size_t idx, const char *name, int64_t v)             { genericInsertInput(n, idx, v); }
static void setposattr(jit::Node* n, size_t idx, const char *name, const at::Scalar& v)   { genericInsertInput(n, idx, v); }
static void setposattr(jit::Node* n, size_t idx, const char *name, SparseTensor s)        { failPosAttr(); }
static void setposattr(jit::Node* n, size_t idx, const char *name, const at::IntList& v)  {
  using ArgumentStash = jit::tracer::ArgumentStash;
  if (ArgumentStash::hasIntList(name)) {
    auto info = ArgumentStash::popIntList(name);
    for (size_t i = 0; i < info.size(); ++i) {
      if (info[i] != nullptr) continue;
      info[i] = createConstant(n, v[i]);
    }
    jit::TensorType expected_type {at::kLong, -1, {}};
    for (jit::Value* v : info) {
      if (*v->type() != expected_type) {
        throw std::runtime_error(
          "Type mismatch in setposattr for IntList. Check that your program "
          "is valid without tracing, and please file a bug report if it is.");
      }
    }
    jit::WithInsertPoint insert_point{n};
    auto symbolic_info = fmap<jit::SymbolicVariable>(info);
    auto size = jit::SymbolicVariable::stack(symbolic_info, 0);
    n->insertInput(idx, size);
  } else {
    return genericInsertInput(n, idx, v);
  }
}
static void setposattr(jit::Node* n, size_t idx, const char *name, bool v)                { genericInsertInput(n, idx, v); }
static void setposattr(jit::Node* n, size_t idx, const char *name, double v)              { genericInsertInput(n, idx, v); }
template<std::size_t N>
static void setposattr(jit::Node* n, size_t idx, const char *name, std::array<bool, N> v) { failPosAttr(); }

VariableType::VariableType(Context* context, Type* baseType)
  : Type(context, /*is_variable_or_undefined=*/true)
  , baseType(baseType) {
  str = std::string("Variable[") + baseType->toString() + "]";
}

ScalarType VariableType::scalarType() const {
  return baseType->scalarType();
}
Backend VariableType::backend() const {
  return baseType->backend();
}
bool VariableType::is_cuda() const { return baseType->is_cuda(); }
bool VariableType::is_sparse() const { return baseType->is_sparse(); }
bool VariableType::is_distributed() const { return baseType->is_distributed(); }

std::unique_ptr<Storage> VariableType::storage() const {
  return baseType->storage();
}
std::unique_ptr<Storage> VariableType::storage(size_t size) const {
  return baseType->storage(size);
}
std::unique_ptr<Storage> VariableType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
  return baseType->storageFromBlob(data, size, deleter);
}
std::unique_ptr<Storage> VariableType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  return baseType->unsafeStorageFromTH(th_pointer, retain);
}
std::unique_ptr<Storage> VariableType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
  return baseType->storageWithAllocator(size, std::move(allocator));
}
Tensor VariableType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  return make_variable(baseType->unsafeTensorFromTH(th_pointer, retain), /*requires_grad=*/false);
}
std::unique_ptr<Generator> VariableType::generator() const {
  return baseType->generator();
}

const char * VariableType::toString() const {
  return str.c_str();
}
size_t VariableType::elementSizeInBytes() const {
  return baseType->elementSizeInBytes();
}
Type & VariableType::toBackend(Backend b) const {
  return *getType(baseType->toBackend(b));
}
Type & VariableType::toScalarType(ScalarType s) const {
  return *getType(baseType->toScalarType(s));
}
TypeID VariableType::ID() const {
  throw std::runtime_error("VariableType::ID() not implemented");
}

const char * VariableType::typeString() {
  return "VariableType";
}

struct VariableTypeRegistry {
  static constexpr int MaxTypes = static_cast<int>(at::TypeID::NumOptions);

  VariableTypeRegistry();

  std::vector<VariableType> types_vec;
  at::Type* types[MaxTypes];
};

VariableTypeRegistry::VariableTypeRegistry() {
  auto& context = at::globalContext();
  types_vec.reserve(MaxTypes);
  memset(types, 0, MaxTypes * sizeof(at::Type*));
  for (int p = 0; p < static_cast<int>(Backend::NumOptions); ++p) {
    for (int s = 0; s < static_cast<int>(ScalarType::NumOptions); s++) {
      auto baseType = context.type_registry[p][s].get();
      if (baseType && baseType->backend() != Backend::Undefined) {
        auto id = static_cast<int>(baseType->ID());
        types_vec.emplace_back(&context, baseType);
        types[id] = &types_vec.back();
      }
    }
  }
}

static VariableTypeRegistry registry;

bool VariableType::isVariableType(const at::Type& type) {
  // Since all VariableTypes are allocated contiguously in types_vec, we can
  // just check that the pointer is inside the correct range.
  ptrdiff_t offset = reinterpret_cast<const char*>(&type) - reinterpret_cast<const char*>(registry.types_vec.data());
  ptrdiff_t extent = VariableTypeRegistry::MaxTypes * sizeof(VariableType);
  return offset >= 0 && offset < extent;
}

at::Type* VariableType::getType(const at::Type& baseType) {
  return registry.types[static_cast<int>(baseType.ID())];
}

at::Type* VariableType::getType(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    throw std::runtime_error("tensor is undefined");
  }
  return getType(tensor.type());
}

std::vector<at::Type*> VariableType::allTypes() {
  std::vector<Type*> res;
  res.reserve(registry.types_vec.size());
  for (auto& type : registry.types_vec) {
    res.push_back(&type);
  }
  return res;
}

Variable & VariableType::checked_cast_variable(const Tensor & t, const char * name, int pos) {
  if (!t.defined()) {
    AT_ERROR("Expected a Tensor of type Variable but found an undefined Tensor for argument #", pos, " '", name, "'");
  }
  if (!isVariableType(t.type())) {
    AT_ERROR("Expected object of type Variable but found type ", t.type().toString(), " for argument #", pos, " '", name, "'");
  }
  return as_variable_ref(const_cast<Tensor&>(t));
}

Tensor & VariableType::unpack(const Tensor & t, const char * name, int pos) {
  return checked_cast_variable(t, name, pos).data();
}

SparseTensor VariableType::unpack(SparseTensor t, const char * name, int pos) {
  return SparseTensor(checked_cast_variable(t.tref, name, pos).data());
}

Tensor VariableType::unpack_opt(const Tensor & t, const char * name, int pos) {
  if (!t.defined()) {
    return Tensor();
  }
  return unpack(t, name, pos);
}

std::vector<at::Tensor> VariableType::unpack(at::TensorList tl, const char *name, int pos) {
  std::vector<at::Tensor> ret(tl.size());
  for (size_t i = 0; i < tl.size(); ++i) {
    const auto &t = tl[i];
    if (!t.defined()) {
      AT_ERROR("Expected a Tensor of type Variable but found an undefined Tensor at position #", i, " "
                    "for iterable argument #", pos, " '", name, "'");
    }
    if (!isVariableType(t.type())) {
      AT_ERROR("Expected object of type Variable but found type ", t.type().toString(), " at position #", i, " "
                    "for iterable argument #", pos, " '", name, "'");
    }
    ret[i] = static_cast<const Variable&>(t).data();
  }
  return ret;
}

// Assumed that saved tensor lists are never inplace outputs
static std::vector<SavedVariable> make_saved_variable_list(TensorList tensors) {
  return fmap(tensors, [](const Tensor& tensor) -> SavedVariable {
      return SavedVariable{tensor, false /* is output */}; });
}

template <typename... Tensors, size_t... Is>
std::tuple<Tensors...> as_variable_impl(
    std::tuple<Tensors...> tensors,
    Indices<Is...>) {
  // Expand the integer parameter pack into a sequence of Variable
  // constructions. This turns into (boolean omitted):
  // Variable(std::get<0>(tensors)), Variable(std::get<1>(tensors)), ...
  return std::tuple<Tensors...>(
      make_variable(std::get<Is>(tensors), /*requires_grad=*/false)...);
}

template <typename... Tensors>
std::tuple<Tensors...> as_variable(std::tuple<Tensors...> tensors) {
  // `sizeof...(Tensors)` gets us the size of the `Tensors` parameter pack at
  // compile time. We use it to parameterize a `MakeIndices` class, which will
  // expand into an Indices object containing the numbers 0 to
  // sizeof...(Tensors) - 1.
  return as_variable_impl(
      tensors, typename MakeIndices<sizeof...(Tensors)>::indices());
}

static Tensor as_variable(Tensor tensor) {
  return make_variable(std::move(tensor), /*requires_grad=*/false);
}

static std::vector<Tensor> as_variable(TensorList tl) {
  std::vector<Tensor> variables;
  for (auto& t : tl) {
    variables.emplace_back(make_variable(std::move(t), /*requires_grad=*/false));
  }
  return variables;
}

static Tensor as_view(const Tensor & base, Tensor tensor) {
  auto base_var = Variable(base);
  if (base_var.is_view()) {
    base_var = base_var.base();
  }
  return make_variable_view(std::move(base_var), std::move(tensor));
}

struct ComputeRequiresGrad : IterArgs<ComputeRequiresGrad> {
  bool out = false;
  using IterArgs<ComputeRequiresGrad>::operator();
  void operator()(const at::Tensor& tensor) {
    const auto& var = static_cast<const Variable&>(tensor);
    if (var.defined() && var.requires_grad()) {
      out = true;
    }
  }
  bool short_circuit() { return out; }
};

template<typename... Args>
static bool compute_requires_grad(Args&&... args) {
  if (!GradMode::is_enabled()) {
    return false;
  }
  return ComputeRequiresGrad().apply(std::forward<Args>(args)...).out;
}

static void check_no_requires_grad(const Tensor& tensor, const char* name) {
  auto& var = static_cast<const Variable&>(tensor);
  if (var.defined() && var.requires_grad()) {
    std::string msg = "the derivative for '";
    msg += name;
    msg += "' is not implemented";
    throw std::runtime_error(msg);
  }
}

static void check_inplace(const Tensor& tensor) {
  auto& var = static_cast<const Variable&>(tensor);
  if (var.requires_grad() && var.is_leaf() && GradMode::is_enabled()) {
    AT_ERROR(
      "a leaf Variable that requires grad has been used in an in-place operation.");
  }
}

static void throw_error_out_requires_grad(const char* name) {
  AT_ERROR(
      name, "(): functions with out=... arguments don't support automatic differentiation, "
      "but one of the arguments requires grad.");
}

static void rebase_history(Tensor& tensor, std::shared_ptr<Function> grad_fn) {
  if (grad_fn && tensor.defined()) {
    auto& var = as_variable_ref(tensor);
    grad_fn->set_num_inputs(1);
    var.rebase_history({std::move(grad_fn), 0});
  }
}

static void rebase_history(TensorList tensors, std::shared_ptr<Function> grad_fn) {
  if (grad_fn) {
    grad_fn->set_num_inputs(tensors.size());
    uint32_t output_nr = 0;
    for (auto& tensor : tensors) {
      if (tensor.defined()) {
        auto& var = as_variable_ref(const_cast<Tensor&>(tensor));
        var.rebase_history({grad_fn, output_nr});
      }
      output_nr++;
    }
  }
}

// var must be the only differentiable output of the function. Use the ArrayRef
// overload for functions with multiple differentiable outputs.
static void set_history(Tensor& tensor, std::shared_ptr<Function> grad_fn) {
  if (grad_fn && tensor.defined()) {
    auto& var = as_variable_ref(tensor);
    autograd::create_gradient_edge(var, std::move(grad_fn));
  }
}

static void set_history(TensorList tensors, std::shared_ptr<Function> grad_fn) {
  if (grad_fn) {
    grad_fn->set_num_inputs(tensors.size());
    uint32_t output_nr = 0;
    for (auto& tensor : tensors) {
      if (tensor.defined()) {
        auto& var = as_variable_ref(const_cast<Tensor&>(tensor));
        var.set_gradient_edge({grad_fn, output_nr});
      }
      output_nr++;
    }
  }
}

struct Flatten : IterArgs<Flatten> {
  Flatten(variable_list& out) : out(out) {}
  variable_list& out;
  void operator()(const at::Tensor& x) { out.emplace_back(x); }
  void operator()(at::ArrayRef<at::Tensor> xs) {
    out.insert(out.end(), xs.begin(), xs.end());
  }
};

template<typename... Args> inline variable_list flatten(Args&&... args) {
  variable_list out;
  out.reserve(count_tensors(std::forward<Args>(args)...));
  Flatten(out).apply(std::forward<Args>(args)...);
  return out; // RVO
}

static void increment_version(Tensor & t) {
  as_variable_ref(t).bump_version();
}

static bool isFloatingPoint(ScalarType s) {
  return s == kFloat || s == kDouble || s == kHalf;
}

Tensor & VariableType::s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const {
  // TODO: once copy is exposed in Declarations.yaml we may be able to bind
  // it automatically
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack(src, "src", 1);
  check_inplace(self);
  std::shared_ptr<CopyBackwards> grad_fn;
  auto requires_grad = compute_requires_grad(self, src);
  requires_grad &= isFloatingPoint(self.type().scalarType());
  if (requires_grad) {
    grad_fn = std::make_shared<CopyBackwards>();
    grad_fn->set_next_edges(collect_next_edges(self, src));
    grad_fn->set_num_inputs(1);
    grad_fn->src_type = &src.type();
    grad_fn->src_device = src.is_cuda() ? src.get_device() : -1;
  }
  baseType->s_copy_(self_, src_, non_blocking);
  increment_version(self);
  rebase_history(self, std::move(grad_fn));
  return self;
}

Tensor & VariableType::resize_(Tensor & self, IntList size) const {
  auto& self_ = unpack(self, "self", 0);
  if (as_variable_ref(self).requires_grad()) {
    AT_ERROR("cannot resize variables that require grad");
  }
  baseType->resize_(self_, size);
  return self;
}

Tensor & VariableType::resize_as_(Tensor & self, const Tensor & the_template) const {
  auto& self_ = unpack(self, "self", 0);
  auto& the_template_ = unpack(the_template, "the_template", 1);
  if (as_variable_ref(self).requires_grad()) {
    AT_ERROR("cannot resize variables that require grad");
  }
  baseType->resize_as_(self_, the_template_);
  return self;
}

Tensor VariableType::contiguous(const Tensor & self) const {
  unpack(self, "self", 0);
  if (self.is_contiguous()) {
    return self;
  }
  return self.clone();
}

static std::vector<std::vector<int64_t>> to_args_sizes(TensorList tensors) {
  std::vector<std::vector<int64_t>> args_sizes(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    args_sizes[i] = tensors[i].sizes();
  }
  return args_sizes;
}

${type_derived_method_definitions}

}} // namespace torch::autograd
