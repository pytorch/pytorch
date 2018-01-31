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
  // TODO: If you ever want to support returning an undefined tensor from
  // a function, you'll have to uncomment the line below.  Not sure if
  // we actually want to support this.
  // if (!data.defined()) return Variable();
  TORCH_ASSERT(grad_fn);
  int output_nr = grad_fn->num_inputs++;
  return make_variable(std::move(data), output_nr, std::move(grad_fn));
}

VariableImpl::VariableImpl(Tensor data_, bool requires_grad, int output_nr, std::shared_ptr<Function> grad_fn)
  : TensorImpl(VariableType::getType(data_))
  , data(std::move(data_))
  , grad()
  , _grad_fn(std::move(grad_fn))
  , version_counter()
  , _requires_grad(requires_grad)
  , is_view(false)
  , output_nr(output_nr)
  , pyobj(nullptr) {
  TORCH_ASSERTM(!_grad_fn || !_requires_grad, "_requires_grad should be false if grad_fn is set");
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

std::shared_ptr<Function> VariableImpl::get_grad_accumulator() {
  if (_grad_fn) {
    throw std::logic_error("get_grad_accumulator() should be only called on leaf Variables");
  }
  if (!_requires_grad) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(mutex);

  auto result = grad_accumulator.lock();
  if (result) return result;

  result = std::make_shared<AccumulateGrad>(Variable(this, true));
  grad_accumulator = result;
  return result;
}

VariableViewImpl::VariableViewImpl(Variable base_, at::Tensor data_, int output_nr,
                                   std::shared_ptr<Function> grad_fn)
  : VariableImpl(std::move(data_), false, output_nr, std::move(grad_fn))
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
  if (!_grad_fn && !base.requires_grad()) {
    return _grad_fn;
  }
  auto current_version = version_counter.current_version();
  if (attr_version != current_version) {
    TORCH_ASSERT(output_nr == 0);
    auto fn = std::make_shared<generated::AsStridedBackward>();
    fn->self_geometry = TensorGeometry(base);
    fn->size = sizes();
    fn->stride = strides();
    fn->storage_offset = data.storage_offset();
    fn->set_flags(Function::flags(base));
    fn->num_inputs = 1;
    _grad_fn = std::move(fn);
    attr_version = current_version;
  }
  return _grad_fn;
}

void VariableViewImpl::rebase_history(int output_nr, std::shared_ptr<Function> grad_fn) {
  TORCH_ASSERT(output_nr == 0);
  TORCH_ASSERT(grad_fn);
  TORCH_ASSERTM(grad_fn->num_inputs == 1, "Functions which modify views in-place must return a single Variable");
  this->output_nr = output_nr;
  base.output_nr() = 0;
  base.get()->_grad_fn = std::make_shared<CopySlices>(
      base, TensorGeometry(data), std::move(grad_fn));
  get_grad_fn();  // trigger an update to the view's grad_fn
}

Variable Variable::detach() const {
  Variable detached = make_variable(data());
  detached.version_counter() = version_counter();
  return detached;
}

void Variable::detach_() {
  if (is_view()) {
    throw std::runtime_error("Can't detach views in-place. Use detach() instead");
  }
  get()->_requires_grad = false;
  output_nr() = 0;
  get()->_grad_fn = nullptr;
}


}} // namespace torch::autograd
