#include "Python.h"
#include "VariableType.h"

// ${generated_comment}

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/saved_variable.h"
#include "torch/csrc/autograd/generated/Functions.h"
#include "torch/csrc/autograd/functions/tensor.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/jit/tracer.h"

#include <initializer_list>
#include <iostream>
#include <functional>

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
static void setattr(jit::Node* n, jit::Symbol name, const at::IntList& v)  { n->is_(name, v); }
static void setattr(jit::Node* n, jit::Symbol name, bool v)                { n->i_(name, v); }
static void setattr(jit::Node* n, jit::Symbol name, double v)              { n->f_(name, v); }
template<unsigned long N>
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
Tensor VariableType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  return baseType->unsafeTensorFromTH(th_pointer, retain);
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
  if (&t.type() != &type) {
    runtime_error("Expected object of type %s but found type %s for argument #%d '%s'",
        type.toString(), t.type().toString(), pos, name);
  }
  return static_cast<Variable&>(const_cast<Tensor&>(t));
}

Tensor & VariableType::unpack(const Tensor & t, const char * name, int pos) const {
  return checked_cast(*this, t, name, pos).data();
}

Tensor & VariableType::unpack_long(const Tensor & t, const char * name, int pos) const {
  auto& type = *VariableImpl::getType(baseType->toScalarType(kLong));
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
  auto& type = *VariableImpl::getType(baseType->toScalarType(scalarType));
  return checked_cast(type, t, name, pos).data();
}

Tensor VariableType::unpack_opt(const Tensor & t, const char * name, int pos) const {
  if(!t.defined()) {
    return Tensor();
  }
  return unpack(t, name, pos);
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

Variable VariableType::as_variable(Tensor tensor) const {
  return make_variable(std::move(tensor));
}

std::tuple<Variable, Variable>
VariableType::as_variable(std::tuple<Tensor, Tensor> tensors) const {
  return std::make_tuple<>(
      make_variable(std::move(std::get<0>(tensors))),
      make_variable(std::move(std::get<1>(tensors))));
}

std::tuple<Variable, Variable, Variable>
VariableType::as_variable(std::tuple<Tensor, Tensor, Tensor> tensors) const {
  return std::make_tuple<>(
      make_variable(std::move(std::get<0>(tensors))),
      make_variable(std::move(std::get<1>(tensors))),
      make_variable(std::move(std::get<2>(tensors))));
}

std::vector<Variable> VariableType::as_variable(TensorList tl) const {
  std::vector<Variable> variables;
  for (auto& t : tl) {
    variables.emplace_back(make_variable(std::move(t)));
  }
  return variables;
}

static Variable as_view(Variable base, Tensor tensor) {
  if (base.is_view()) {
    base = base.base();
  }
  return make_variable_view(std::move(base), std::move(tensor));
}

static void ensure_no_aten_scalars(Tensor & data) {
  if (data.defined() && data.dim() == 0) {
    data.as_strided_({1}, {1});
  }
}

template<typename T>
static VarFlags compute_flags_tmpl(T tensors) {
  VarFlags flags = {false, false};
  for (const Tensor& tensor : tensors) {
    auto& var = static_cast<const Variable&>(tensor);
    if (var.defined()) {
      flags.requires_grad |= var.requires_grad();
      flags.is_volatile |= var.is_volatile();
    }
  }
  flags.requires_grad &= !flags.is_volatile;
  return flags;
}

using TensorRef = std::reference_wrapper<const Tensor>;
using TensorRefList = std::initializer_list<TensorRef>;

// ArrayRef is not covariant, which means there is no
// implicit conversion between TensorList (aka ArrayRef<Tensor>)
// and ArrayRef<Variable>.  What we do instead is manually
// construct a variable_list, which itself is implicitly convertible
// into an ArrayRef<Variable> (but don't return an ArrayRef<Variable>;
// ArrayRef is non-owning!)
static variable_list cast_tensor_list(const TensorList& tensors) {
  // TODO: Eliminate the intermediate vector allocation
  return variable_list(tensors.begin(), tensors.end());
}

static VarFlags compute_flags(const TensorRefList& tensors) {
  return compute_flags_tmpl(tensors);
}

static VarFlags compute_flags(TensorList tensors) {
  return compute_flags_tmpl(tensors);
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

static function_list compute_next_functions(const std::initializer_list<Tensor>& tensors) {
  return Function::flags(tensors).next_functions;
}

static function_list compute_next_functions(TensorList tensors) {
  return Function::flags(tensors).next_functions;
}

static void check_inplace(const Tensor& tensor) {
  auto& var = static_cast<const Variable&>(tensor);
  if (var.requires_grad() && !var.grad_fn()) {
    at::runtime_error(
      "a leaf Variable that requires grad has been used in an in-place operation.");
  }
}

static void set_flags(Variable& var, VarFlags flags, std::shared_ptr<Function> grad_fn, bool inplace=false, int output_nr = 0) {
  if (grad_fn) {
    grad_fn->num_inputs = 1;
  }
  if (inplace) {
    var.rebase_history(flags, output_nr, std::move(grad_fn));
  } else {
    // TODO: combine this code path with the Variable construction
    var.get()->requires_grad = flags.requires_grad;
    var.get()->is_volatile = flags.is_volatile;
    var.get()->output_nr = output_nr;
    var.get()->_grad_fn = std::move(grad_fn);
  }
}

static void set_flags(at::ArrayRef<Variable> vl, VarFlags flags, std::shared_ptr<Function> grad_fn) {
  if (grad_fn) {
    grad_fn->num_inputs = vl.size();
  }
  int64_t output_nr = 0;
  for (auto& var : vl) {
    // TODO: combine this with the Variable construction
    var.get()->requires_grad = flags.requires_grad;
    var.get()->is_volatile = flags.is_volatile;
    var.get()->output_nr = output_nr;
    var.get()->_grad_fn = grad_fn;
    output_nr++;
  }
}

std::vector<Tensor> as_tensor_list(std::vector<Variable> &vars) {
  std::vector<Tensor> tensors;
  for (auto& v : vars) {
    tensors.emplace_back(std::move(v));
  }
  return tensors;
}

static void increment_version(const Tensor & t) {
  auto& var = static_cast<const Variable&>(t);
  var.version_counter().increment();
}

static bool isFloatingPoint(ScalarType s) {
  return s == kFloat || s == kDouble || s == kHalf;
}

void VariableType::s_copy(const Tensor & src, Tensor & dst) const {
  // TODO: once copy is exposed in Declarations.yaml we may be able to bind
  // it automatically
  auto& src_ = unpack_any(src, "src", 0);
  auto& dst_ = unpack(dst, "dst", 1);
  check_inplace(dst);
  std::shared_ptr<CopyBackwards> grad_fn;
  auto flags = compute_flags({ dst, src });
  flags.requires_grad &= isFloatingPoint(dst.type().scalarType());
  if (flags.requires_grad) {
    // TODO: handle device movement
    grad_fn = std::make_shared<CopyBackwards>();
    grad_fn->is_executable = true;
    grad_fn->next_functions = compute_next_functions({ dst, src });
    grad_fn->num_inputs = 1;
    grad_fn->src_type = &src.type();
  }
  baseType->s_copy(src_, dst_);
  increment_version(dst);
  set_flags(static_cast<Variable&>(dst), flags, std::move(grad_fn), true);
}

Tensor & VariableType::resize_(Tensor & self, IntList size) const {
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  auto& self_var = static_cast<Variable&>(self);
  if (self_var.grad_fn()) {
    at::runtime_error("cannot resize non-leaf variables");
  }
  if (self_var.requires_grad()) {
    at::runtime_error("cannot resize variables which require grad");
  }
  baseType->resize_(self_, size);
  return self;
}

Tensor & VariableType::resize_as_(Tensor & self, const Tensor & the_template) const {
  return resize_(self, the_template.sizes());
}

Tensor VariableType::contiguous(const Tensor & self) const {
  unpack(self, "self", 0);
  if (self.is_contiguous()) {
    return self;
  }
  return self.clone();
}

std::vector<int64_t> to_arg_sizes(TensorList tensors, int64_t dim) {
  std::vector<int64_t> arg_sizes(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    arg_sizes[i] = tensors[i].size(dim);
  }
  return arg_sizes;
}

${type_derived_method_definitions}

}} // namespace torch::autograd
