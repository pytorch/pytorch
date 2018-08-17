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
#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/symbolic_variable.h"
#include "torch/csrc/jit/ir.h"

#include "torch/csrc/utils/variadic.h"
#include "torch/csrc/autograd/functions/utils.h"

#include <ATen/detail/VariableHooksInterface.h>

#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#ifdef Type
#undef Type
#endif
#endif

using namespace at;
using namespace torch::autograd::generated;

namespace torch { namespace autograd {

VariableType::VariableType(Context* context, Type* baseType)
  : Type(context, baseType->type_id(), /*is_variable=*/true, /*is_undefined=*/false)
  , baseType(baseType)
  , id_(context->freshTypeID()) {
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

std::unique_ptr<Storage> VariableType::storage(bool resizable) const {
  return baseType->storage();
}
std::unique_ptr<Storage> VariableType::storage(size_t size, bool resizable) const {
  return baseType->storage(size);
}
std::unique_ptr<Storage> VariableType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
  return baseType->storageFromBlob(data, size, deleter);
}
std::unique_ptr<Storage> VariableType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  return baseType->unsafeStorageFromTH(th_pointer, retain);
}
std::unique_ptr<Storage> VariableType::storageWithAllocator(int64_t size, Allocator* allocator) const {
  return baseType->storageWithAllocator(size, allocator);
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
  return static_cast<TypeID>(id_);
}

const char * VariableType::typeString() {
  return "VariableType";
}

std::vector<std::unique_ptr<Type>> type_to_variable_type;

// XXX - this is not threadsafe with uses of Variables
void register_variable_type_for(Type* baseType) {
  AT_ASSERT(baseType);
  size_t base_id = static_cast<size_t>(baseType->ID());
  if(type_to_variable_type.size() <= base_id) {
    type_to_variable_type.resize(base_id + 1);
  }
  type_to_variable_type[base_id].reset(new VariableType(&at::globalContext(), baseType));
}

struct VariableTypeRegistry {
  VariableTypeRegistry() {
    auto& context = at::globalContext();
    for (int p = 0; p < static_cast<int>(Backend::NumOptions); ++p) {
      for (int s = 0; s < static_cast<int>(ScalarType::NumOptions); ++s) {
        auto baseType = context.getTypeRaw(static_cast<Backend>(p), static_cast<ScalarType>(s));
        if (baseType && baseType->backend() != Backend::Undefined) {
          register_variable_type_for(baseType);
        }
      }
    }
  }
};

struct VariableHooks : public at::VariableHooksInterface {
  VariableHooks(at::VariableHooksArgs) {}
  void registerVariableTypeFor(at::Context*, at::Backend, at::ScalarType) const override;
  at::Type& getVariableType(const at::Type&) const override;
};

// Sigh, the registry doesn't support namespaces :(
using at::RegistererVariableHooksRegistry;
using at::VariableHooksRegistry;

// WARNING: YOU MUST DO THE NEXT TWO STATIC INITIALIZERS IN THIS ORDER.
//
// If you do it in the other order, this is what can happen if
// these static initializers are called before Context is
// initialized:
//
//    - VariableHooks::registerVariableTypeFor will be activated
//      to register a variable type
//
//    - We run the constructor of VariableTypeRegistry, which
//      calls at::globalContext()
//
//    - Context is not initialized yet, so we call the constructor
//      of Context
//
//    - We register CPU types, calling VariableHooks::registerVariableTypeFor
//
//    - We register the CPU type as a variable type
//
//    - In VariableTypeRegistry, we try to register the Variable type AGAIN!!
//      Disaster.
//
static VariableTypeRegistry registry;
REGISTER_VARIABLE_HOOKS(VariableHooks)

// Pre-condition: backend/scalar_type is a valid type in the type_registry
void VariableHooks::registerVariableTypeFor(at::Context* context, at::Backend backend, at::ScalarType scalar_type) const {
  auto* baseType = context->getTypeRaw(backend, scalar_type);
  register_variable_type_for(baseType);
}

at::Type& VariableHooks::getVariableType(const at::Type& baseType) const {
  return *VariableType::getType(baseType);
}

bool VariableType::isVariableType(const at::Type& type) {
  return type.is_variable();
}

at::Type* VariableType::getType(const at::Type& baseType) {
  auto id = static_cast<size_t>(baseType.ID());
  if(id >= type_to_variable_type.size())
    return nullptr;
  return type_to_variable_type[id].get();
}

at::Type* VariableType::getType(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    throw std::runtime_error("tensor is undefined");
  }
  return getType(tensor.type());
}

namespace {
std::vector<at::Type*> allTypesForBackends(at::ArrayRef<at::Backend> backends) {
  auto& context = at::globalContext();
  std::vector<Type*> res;
  res.reserve(backends.size() * static_cast<int>(ScalarType::NumOptions));
  for (auto p : backends) {
    for (int s = 0; s < static_cast<int>(ScalarType::NumOptions); s++) {
      auto baseType = context.getTypeRaw(static_cast<Backend>(p), static_cast<ScalarType>(s));
      if (baseType) {
        res.emplace_back(VariableType::getType(*baseType));
      }
    }
  }
  return res;
}
}

std::vector<at::Type*> VariableType::allCPUTypes() {
  return allTypesForBackends({ Backend::CPU, Backend::SparseCPU });
}

std::vector<at::Type*> VariableType::allCUDATypes() {
  at::globalContext().lazyInitCUDA();
  return allTypesForBackends({ Backend::CUDA, Backend::SparseCUDA });
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

SparseTensorRef VariableType::unpack(SparseTensorRef t, const char * name, int pos) {
  return SparseTensorRef(checked_cast_variable(t.tref, name, pos).data());
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

static Tensor as_variable(Tensor tensor) {
  return make_variable(std::move(tensor), /*requires_grad=*/false);
}

static std::vector<Tensor> as_variable(TensorList tl) {
  return fmap(tl, [](const Tensor& t) -> Tensor {
      return make_variable(std::move(t), /*requires_grad=*/false);
  });
}

template <typename... Tensors, size_t... Is>
std::tuple<Tensors...> as_variable_impl(
    std::tuple<Tensors...> tensors,
    Indices<Is...>) {
  // Expand the integer parameter pack into a sequence of Variable
  // constructions. This turns into (boolean omitted):
  // Variable(std::get<0>(tensors)), Variable(std::get<1>(tensors)), ...
  return std::tuple<Tensors...>(
      as_variable(std::get<Is>(tensors))...);
}

// NB: Because this was not forward declared, recursive std::tuple won't work.
// You can probably rejigger this to make it supported if you really need it.
template <typename... Tensors>
std::tuple<Tensors...> as_variable(std::tuple<Tensors...> tensors) {
  // `sizeof...(Tensors)` gets us the size of the `Tensors` parameter pack at
  // compile time. We use it to parameterize a `MakeIndices` class, which will
  // expand into an Indices object containing the numbers 0 to
  // sizeof...(Tensors) - 1.
  return as_variable_impl(
      tensors, typename MakeIndices<sizeof...(Tensors)>::indices());
}

static Tensor as_view(const Tensor & base, Tensor tensor) {
  auto base_var = Variable(base);
  if (base_var.is_view()) {
    base_var = base_var.base();
  }
  return make_variable_view(std::move(base_var), std::move(tensor));
}

static std::vector<Tensor> as_view(const Tensor & base, std::vector<Tensor> tensors) {
  auto base_var = Variable(base);
  if (base_var.is_view()) {
    base_var = base_var.base();
  }
  for(Tensor &tensor : tensors) {
    tensor = make_variable_view(base_var, std::move(tensor));
  }
  return tensors;
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

// TODO: Blegh, bare references

static void rebase_history(Variable& var, std::shared_ptr<Function> grad_fn) {
  if (grad_fn && var.defined()) {
    grad_fn->add_input_metadata(var);
    var.rebase_history({std::move(grad_fn), 0});
  }
}

static void rebase_history(ArrayRef<Variable> vars, std::shared_ptr<Function> grad_fn) {
  if (grad_fn) {
    for (auto& var : vars) {
      if (var.defined()) {
        // TODO: eliminate const_cast
        auto output_nr = grad_fn->add_input_metadata(var);
        const_cast<Variable&>(var).rebase_history({grad_fn, output_nr});
      } else {
        grad_fn->add_input_metadata(Function::undefined_input());
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

template<typename... Args> inline variable_list flatten_tensor_args(Args&&... args) {
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
    grad_fn->src_type = &src.type();
    if (src.is_cuda()) {
      grad_fn->src_device = src.get_device();
    }
  }
  baseType->s_copy_(self_, src_, non_blocking);
  increment_version(self);
  rebase_history(as_variable_ref( self ), std::move(grad_fn));
  return self;
}

Tensor & VariableType::_s_copy_from(const Tensor & self, Tensor & dst, bool non_blocking) const {
  AT_ERROR("copy_from does not support automatic differentiation; use copy_ instead");
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
    args_sizes[i] = tensors[i].sizes().vec();
  }
  return args_sizes;
}

${type_derived_method_definitions}

}} // namespace torch::autograd
