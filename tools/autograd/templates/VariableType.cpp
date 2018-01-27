#include "Python.h"
#include "VariableType.h"

// ${generated_comment}

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/grad_mode.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/autograd/generated/Functions.h"
#include "torch/csrc/autograd/functions/tensor.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/jit/tracer.h"
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
template<std::size_t N>
static void setattr(jit::Node* n, jit::Symbol name, std::array<bool, N> v) { n->is_(name, std::vector<int64_t>(v.begin(), v.end())); }

VariableType::VariableType(Context* context, Type* baseType)
  : Type(context)
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
  return make_variable(baseType->unsafeTensorFromTH(th_pointer, retain), false);
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
  return *VariableImpl::getType(baseType->toBackend(b));
}
Type & VariableType::toScalarType(ScalarType s) const {
  return *VariableImpl::getType(baseType->toScalarType(s));
}
TypeID VariableType::ID() const {
  throw std::runtime_error("VariableType::ID() not implemented");
}

const char * VariableType::typeString() {
  return "VariableType";
}

Variable & VariableType::checked_cast(const Type & type, const Tensor & t, const char * name, int pos) {
  if(!t.defined()) {
    runtime_error("Expected a Tensor of type %s but found an undefined Tensor for argument #%d '%s'",
        type.toString(), pos, name);
  }
  if (&t.type() != &type && &t.type() != &type.toBackend(toSparse(t.type().backend()))) {
    runtime_error("Expected object of type %s but found type %s for argument #%d '%s'",
        type.toString(), t.type().toString(), pos, name);
  }
  return static_cast<Variable&>(const_cast<Tensor&>(t));
}

Tensor & VariableType::unpack(const Tensor & t, const char * name, int pos) const {
  return checked_cast(*this, t, name, pos).data();
}

SparseTensor VariableType::unpack(SparseTensor t, const char * name, int pos) const {
  auto backend = is_cuda() ? kSparseCUDA : kSparseCPU;
  return SparseTensor(checked_cast(this->toBackend(backend), t.tref, name, pos).data());
}

Tensor & VariableType::unpack_long(const Tensor & t, const char * name, int pos) const {
  auto& type = *VariableImpl::getType(baseType->toScalarType(kLong));
  return checked_cast(type, t, name, pos).data();
}

Tensor & VariableType::unpack_int(const Tensor & t, const char * name, int pos) const {
  auto& type = *VariableImpl::getType(baseType->toScalarType(kInt));
  return checked_cast(type, t, name, pos).data();
}

Tensor & VariableType::unpack_byte(const Tensor & t, const char * name, int pos) const {
  auto& type = *VariableImpl::getType(baseType->toScalarType(kByte));
  return checked_cast(type, t, name, pos).data();
}

Tensor & VariableType::unpack_any(const Tensor & t, const char * name, int pos) const {
  if (!t.defined()) {
    runtime_error("Expected a Tensor of type Variable but found an undefined Tensor for argument #%d '%s'",
        pos, name);
  }
  auto scalarType = t.type().scalarType();
  auto backend = t.type().backend();
  auto& type = *VariableImpl::getType(baseType->toScalarType(scalarType).toBackend(backend));
  return checked_cast(type, t, name, pos).data();
}

Tensor VariableType::unpack_opt(const Tensor & t, const char * name, int pos) const {
  if (!t.defined()) {
    return Tensor();
  }
  return unpack(t, name, pos);
}

Tensor VariableType::unpack_any_opt(const Tensor & t, const char * name, int pos) const {
  if (!t.defined()) {
    return Tensor();
  }
  return unpack_any(t, name, pos);
}

std::vector<at::Tensor> VariableType::unpack(at::TensorList tl, const char *name, int pos) const {
  std::vector<at::Tensor> ret(tl.size());
  for (size_t i = 0; i < tl.size(); ++i) {
    const auto &t = tl[i];
    if (!t.defined()) {
      runtime_error("Expected a Tensor of type %s but found an undefined Tensor at position #%d "
                    "for iterable argument #%d '%s'",
                    toString(), i, pos, name);
    }
    if (&t.type() == this) {
      ret[i] = static_cast<const Variable&>(t).data();
    } else {
      runtime_error("Expected object of type %s but found type %s at position #%d "
                    "for iterable argument #%d '%s'",
                    toString(),t.type().toString(), i, pos, name);
    }
  }
  return ret;
}

std::vector<at::Tensor> VariableType::unpack_idxs(at::TensorList tl, const char *name, int pos) const {
  auto& longType = *VariableImpl::getType(baseType->toScalarType(kLong));
  auto& byteType = *VariableImpl::getType(baseType->toScalarType(kByte));
  std::vector<at::Tensor> ret(tl.size());
  for (size_t i = 0; i < tl.size(); ++i) {
    const auto &t = tl[i];
    if (!t.defined()) {
      continue;
    } else if (!(t.type() == longType || t.type() == byteType)) {
      runtime_error("Expected object of type %s or %s but found type %s at position #%d "
                    "for iterable argument #%d '%s'",
                    longType.toString(), byteType.toString(), t.type().toString(),
                    i, pos, name);
    } else  {
      ret[i] = static_cast<const Variable&>(t).data();
    }
  }
  return ret;
}

static Tensor as_variable(Tensor tensor) {
  return make_variable(std::move(tensor));
}

static std::tuple<Tensor, Tensor>
as_variable(std::tuple<Tensor, Tensor> tensors) {
  return std::make_tuple<>(
      make_variable(std::move(std::get<0>(tensors))),
      make_variable(std::move(std::get<1>(tensors))));
}

static std::tuple<Tensor, Tensor, Tensor>
as_variable(std::tuple<Tensor, Tensor, Tensor> tensors) {
  return std::make_tuple<>(
      make_variable(std::move(std::get<0>(tensors))),
      make_variable(std::move(std::get<1>(tensors))),
      make_variable(std::move(std::get<2>(tensors))));
}

static std::tuple<Tensor, Tensor, Tensor, Tensor>
as_variable(std::tuple<Tensor, Tensor, Tensor, Tensor> tensors) {
  return std::make_tuple<>(
      make_variable(std::move(std::get<0>(tensors))),
      make_variable(std::move(std::get<1>(tensors))),
      make_variable(std::move(std::get<2>(tensors))),
      make_variable(std::move(std::get<3>(tensors))));
}

static std::vector<Tensor> as_variable(TensorList tl) {
  std::vector<Tensor> variables;
  for (auto& t : tl) {
    variables.emplace_back(make_variable(std::move(t)));
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

#ifndef WITH_SCALARS
static void ensure_no_aten_scalars(Tensor & data) {
  if (data.defined() && data.dim() == 0) {
    data.as_strided_({1}, {1});
  }
}
#endif

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

// NB: This should be called with Tensor/TensorList arguments (not Variables)
template <typename... Args>
static function_list compute_next_functions(Args&&... args) {
  return Function::tensor_flags(std::forward<Args>(args)...).next_functions;
}

static void check_inplace(const Tensor& tensor) {
  auto& var = static_cast<const Variable&>(tensor);
  if (var.requires_grad() && var.is_leaf() && GradMode::is_enabled()) {
    at::runtime_error(
      "a leaf Variable that requires grad has been used in an in-place operation.");
  }
}

static void throw_error_out_requires_grad(const char* name) {
  at::runtime_error(
      "%s(): functions with out=... arguments don't support automatic differentiation, "
      "but one of the arguments requires grad.", name);
}

static void rebase_history(Tensor& tensor, std::shared_ptr<Function> grad_fn) {
  if (grad_fn && tensor.defined()) {
    auto& var = static_cast<Variable&>(tensor);
    grad_fn->num_inputs = 1;
    var.rebase_history(0, std::move(grad_fn));
  }
}

static void rebase_history(TensorList tensors, std::shared_ptr<Function> grad_fn) {
  if (grad_fn) {
    grad_fn->num_inputs = tensors.size();
    int output_nr = 0;
    for (auto& tensor : tensors) {
      if (tensor.defined()) {
        auto& var = static_cast<Variable&>(const_cast<Tensor&>(tensor));
        var.rebase_history(output_nr, grad_fn);
        output_nr++;
      }
    }
  }
}

// var must be the only differentiable output of the function. Use the ArrayRef
// overload for functions with multiple differentiable outputs.
static void set_history(Tensor& tensor, std::shared_ptr<Function> grad_fn) {
  if (grad_fn && tensor.defined()) {
    auto& var = static_cast<Variable&>(tensor);
    grad_fn->num_inputs = 1;
    var.get()->output_nr = 0;
    var.get()->_grad_fn = std::move(grad_fn);
  }
}

static void set_history(TensorList tensors, std::shared_ptr<Function> grad_fn) {
  if (grad_fn) {
    grad_fn->num_inputs = tensors.size();
    int64_t output_nr = 0;
    for (auto& tensor : tensors) {
      if (tensor.defined()) {
        auto& var = static_cast<Variable&>(const_cast<Tensor&>(tensor));
        var.get()->output_nr = output_nr;
        var.get()->_grad_fn = grad_fn;
        output_nr++;
      }
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

static void increment_version(const Tensor & t) {
  auto& var = static_cast<const Variable&>(t);
  var.version_counter().increment();
}

static bool isFloatingPoint(ScalarType s) {
  return s == kFloat || s == kDouble || s == kHalf;
}

Tensor & VariableType::s_copy_(Tensor & self, const Tensor & src, bool async) const {
  // TODO: once copy is exposed in Declarations.yaml we may be able to bind
  // it automatically
  auto& self_ = unpack(self, "self", 0);
  auto& src_ = unpack_any(src, "src", 1);
  check_inplace(self);
  std::shared_ptr<CopyBackwards> grad_fn;
  auto requires_grad = compute_requires_grad(self, src);
  requires_grad &= isFloatingPoint(self.type().scalarType());
  if (requires_grad) {
    grad_fn = std::make_shared<CopyBackwards>();
    grad_fn->next_functions = compute_next_functions( self, src );
    grad_fn->num_inputs = 1;
    grad_fn->src_type = &src.type();
    grad_fn->src_device = src.is_cuda() ? src.get_device() : -1;
  }
  baseType->s_copy_(self_, src_, async);
  increment_version(self);
  rebase_history(self, std::move(grad_fn));
  return self;
}

Tensor & VariableType::resize_(Tensor & self, IntList size) const {
  auto& self_ = unpack(self, "self", 0);
  if (static_cast<Variable&>(self).requires_grad()) {
    at::runtime_error("cannot resize variables that require grad");
  }
  baseType->resize_(self_, size);
  return self;
}

Tensor & VariableType::resize_as_(Tensor & self, const Tensor & the_template) const {
  auto& self_ = unpack(self, "self", 0);
  auto& the_template_ = unpack(the_template, "the_template", 1);
  if (static_cast<Variable&>(self).requires_grad()) {
    at::runtime_error("cannot resize variables that require grad");
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

static std::vector<int64_t> to_arg_sizes(TensorList tensors, int64_t dim) {
  std::vector<int64_t> arg_sizes(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    arg_sizes[i] = tensors[i].size(dim);
  }
  return arg_sizes;
}

${type_derived_method_definitions}

}} // namespace torch::autograd
