#include "Python.h"
#include "torch/csrc/autograd/variable.h"

#include "torch/csrc/assertions.h"
#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/generated/Functions.h"
#include "torch/csrc/autograd/functions/accumulate_grad.h"
#include "torch/csrc/autograd/functions/tensor.h"

using namespace at;

namespace torch { namespace autograd {

Variable make_variable(at::Tensor data, std::shared_ptr<Function> grad_fn) {
  TORCH_ASSERT(grad_fn);
  auto flags = VarFlags(true, false);
  int output_nr = grad_fn->num_inputs++;
  return make_variable(std::move(data), flags, output_nr, std::move(grad_fn));
}

VariableImpl::VariableImpl(Tensor data_, VarFlags flags, int output_nr, std::shared_ptr<Function> grad_fn)
  : TensorImpl(getType(data_))
  , data(std::move(data_))
  , grad()
  , _grad_fn(std::move(grad_fn))
  , version_counter()
  , requires_grad(flags.requires_grad)
  , is_volatile(flags.is_volatile)
  , is_view(false)
  , output_nr(output_nr)
  , pyobj(nullptr) {
  if (!data.defined()) {
    throw std::runtime_error("data is undefined");
  }
}

VariableImpl::~VariableImpl() {
}

const char * VariableImpl::toString() const {
  return "Variable";
}

IntList VariableImpl::sizes() const {
  return data.sizes();
}

IntList VariableImpl::strides() const {
  return data.strides();
}

int64_t VariableImpl::dim() const {
  return data.dim();
}

const char * VariableImpl::typeString() {
  return "VariableType";
}

void * VariableImpl::unsafeGetTH(bool retain) {
  return data.unsafeGetTH(retain);
}

std::unique_ptr<at::Storage> VariableImpl::storage() {
  return data.storage();
}

Scalar VariableImpl::localScalar() {
  return data.pImpl->localScalar();
}

void VariableImpl::assign_(Scalar s) {
  data.assign_(s);
}

std::shared_ptr<Function> VariableImpl::get_grad_accumulator() {
  if (_grad_fn) {
    throw std::logic_error("get_grad_accumulator() should be only called on leaf Variables");
  }
  if (!requires_grad) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex);

  auto result = grad_accumulator.lock();
  if (result) return result;

  result = std::make_shared<AccumulateGrad>(Variable(this, true));
  grad_accumulator = result;
  return result;
}

VariableViewImpl::VariableViewImpl(Variable base_, at::Tensor data_, VarFlags flags,
                                   int output_nr, std::shared_ptr<Function> grad_fn)
  : VariableImpl(std::move(data_), flags, output_nr, std::move(grad_fn))
  , base(std::move(base_))
  , attr_version(0) {
  TORCH_ASSERTM(base.defined(), "base is undefined");
  if (base.is_view()) {
    base = base.base();
  }
  is_view = true;
  version_counter = base.version_counter();
  attr_version = version_counter.current_version();
}

std::shared_ptr<Function>& VariableViewImpl::get_grad_fn() {
  std::lock_guard<std::mutex> lock(mutex);
  if (base.requires_grad() && !requires_grad) {
    // TODO: See test_inplace_view_flags. It would be good to support this operation
    // but that might require sharing requires_grad between the base and the view
    throw std::runtime_error(
        "requires_grad is False and base.requires_grad is True. Cannot use "
        "this view in a differentiable operation. Re-create the view from its "
        "base Variable after the last in-place modification.");
  }
  if (base.is_volatile() && !is_volatile) {
    throw std::runtime_error(
        "is_volatile is False and base.is_volatile is True. Cannot use "
        "this view in a differentiable operation. Re-create the view from its "
        "base Variable after the last in-place modification.");
  }
  auto current_version = version_counter.current_version();
  if (attr_version != current_version) {
    TORCH_ASSERT(output_nr == 0);
    auto fn = std::make_shared<generated::AsStridedBackward>();
    fn->self_geometry = TensorGeometry(base);
    fn->size = sizes();
    fn->stride = strides();
    fn->storage_offset = data.storage_offset();
    fn->set_flags(Function::flags({ base }));
    fn->num_inputs = 1;
    _grad_fn = std::move(fn);
    attr_version = current_version;
  }
  return _grad_fn;
}

void VariableViewImpl::rebase_history(VarFlags flags, int output_nr, std::shared_ptr<Function> grad_fn) {
  TORCH_ASSERT(output_nr == 0);
  TORCH_ASSERT(flags.requires_grad == bool(grad_fn));
  if (grad_fn) {
    TORCH_ASSERTM(grad_fn->num_inputs == 1, "Functions which modify views in-place must return a single Variable");
  } else {
    // TODO: perhaps we should enable this case by setting base.requires_grad=False
    // and base.grad_fn = nullptr.
    TORCH_ASSERTM(!base.requires_grad(), "base.requires_grad does not match view.requires_grad");
  }
  this->requires_grad = flags.requires_grad;
  this->is_volatile = flags.is_volatile;
  this->output_nr = output_nr;
  base.requires_grad() |= flags.requires_grad;
  base.is_volatile() |= flags.is_volatile;
  if (grad_fn) {
    base.output_nr() = 0;
    base.get()->_grad_fn = std::make_shared<CopySlices>(
        base, TensorGeometry(data), std::move(grad_fn));
  }
}

namespace {

struct VariableTypes {
  VariableTypes() {
    auto& context = at::globalContext();
    for (int p = 0; p < static_cast<int>(Backend::NumOptions); ++p) {
      for (int s = 0; s < static_cast<int>(ScalarType::NumOptions); s++) {
        auto baseType = context.type_registry[p][s].get();
        if (baseType) {
          auto id = static_cast<int>(baseType->ID());
          types[id].reset(new VariableType(&context, baseType));
        }
      }
    }
  }

  std::unique_ptr<Type> types[static_cast<int>(TypeID::NumOptions)];
};

} // anonymous namespace

Type* VariableImpl::getType(const Tensor& tensor)
{
  if (!tensor.defined()) {
    throw std::runtime_error("tensor is undefined");
  }
  return getType(tensor.type());
}

Type* VariableImpl::getType(const Type& baseType)
{
  static VariableTypes vt;
  return vt.types[static_cast<int>(baseType.ID())].get();
}

Variable Variable::detach() const {
  Variable detached = make_variable(data());
  detached.is_volatile() = is_volatile();
  detached.version_counter() = version_counter();
  return detached;
}

void Variable::detach_() {
  if (is_view()) {
    throw std::runtime_error("Can't detach views in-place. Use detach() instead");
  }
  get()->requires_grad = false;
  output_nr() = 0;
  get()->_grad_fn = nullptr;
}


}} // namespace torch::autograd
