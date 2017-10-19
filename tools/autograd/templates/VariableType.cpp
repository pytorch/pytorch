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
bool VariableType::isCuda() const { return baseType->isCuda(); }
bool VariableType::isSparse() const { return baseType->isSparse(); }
bool VariableType::isDistributed() const { return baseType->isDistributed(); }

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

std::vector<at::Tensor> VariableType::unpack(const at::TensorList &tl, const char *name, int pos) const {
  std::vector<at::Tensor> ret(tl.size());
  for (size_t i = 0; i < tl.size(); ++i) {
    const auto &t = tl[i];
    if (!t.defined()) {
      runtime_error("Expected a Tensor of type %s but found an undefined Tensor at position #%d "
                    "for iterable argument #%d '%s'",
                    toString(), i, pos, name);
    }
    if (&t.type() == this) {
      ret[i] = static_cast<VariableImpl*>(t.pImpl)->data;
    } else {
      runtime_error("Expected object of type %s but found type %s at position #%d "
                    "for iterable argument #%d '%s'",
                    toString(),t.type().toString(), i, pos, name);
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

Variable VariableType::as_variable(const Scalar & scalar) const {
  auto tensor = scalar.toTensor();
  if (&tensor.type() != baseType) {
    tensor = tensor.toType(*baseType);
  }
  return make_variable(std::move(tensor));
}

struct VariableFlags {
  bool requires_grad;
  bool is_volatile;
};

template<typename T>
static VariableFlags compute_flags_tmpl(T tensors) {
  VariableFlags flags = {false, false};
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

static VariableFlags compute_flags(const TensorRefList& tensors) {
  return compute_flags_tmpl(tensors);
}

static VariableFlags compute_flags(TensorList tensors) {
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
  auto live_refs = var.version_counter().live_refs();
  if (live_refs > 1) {
    at::runtime_error(
      "in-place operations can be only used on variables that don't share "
      "storage with any other variables, but detected that there are %d objects "
      "sharing it", live_refs);
  }
}

static void set_flags(Variable& var, VariableFlags flags, std::shared_ptr<Function> grad_fn) {
  var.requires_grad() = flags.requires_grad;
  var.is_volatile() = flags.is_volatile;
  if (grad_fn) {
    var.output_nr() = grad_fn->num_inputs++;
    var.grad_fn() = std::move(grad_fn);
  }
}

static void increment_version(const Tensor & t) {
  auto& var = static_cast<const Variable&>(t);
  var.version_counter().increment();
}

static void take_version_counter(Tensor & dst, const Tensor & src) {
  // replaces the version counter in dst with the one in src
  // call when dst is a view of src
  auto& src_var = static_cast<const Variable&>(src);
  auto& dst_var = static_cast<Variable&>(dst);
  dst_var.version_counter() = src_var.version_counter();
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
  std::shared_ptr<Identity> grad_fn;
  auto flags = compute_flags({ src });
  flags.requires_grad &= isFloatingPoint(dst.type().scalarType());
  if (flags.requires_grad) {
    // TODO: handle type conversions
    grad_fn = std::make_shared<Identity>();
    grad_fn->is_executable = true;
    grad_fn->next_functions = compute_next_functions({ src });
  }
  baseType->s_copy(src_, dst_);
  increment_version(dst);
  set_flags(static_cast<Variable&>(dst), flags, std::move(grad_fn));
}

Tensor & VariableType::m_resize_(Tensor & self, IntList size) const {
  auto& self_ = unpack(self, "self", 0);
  check_inplace(self);
  auto& self_var = static_cast<Variable&>(self);
  if (self_var.grad_fn()) {
    at::runtime_error("cannot resize non-leaf variables");
  }
  if (self_var.requires_grad()) {
    at::runtime_error("cannot resize variables which require grad");
  }
  baseType->m_resize_(self_, size);
  return self;
}

Tensor & VariableType::m_resize_as_(Tensor & self, const Tensor & the_template) const {
  return m_resize_(self, the_template.sizes());
}

Tensor VariableType::m_contiguous(const Tensor & self) const {
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
