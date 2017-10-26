#include "torch/csrc/autograd/variable.h"

#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/generated/Functions.h"
#include "torch/csrc/autograd/functions/accumulate_grad.h"

using namespace at;

namespace torch { namespace autograd {

VariableImpl::VariableImpl(Tensor data_, bool requires_grad, bool is_volatile)
  : TensorImpl(getType(data_))
  , data(std::move(data_))
  , grad()
  , version_counter()
  , requires_grad(requires_grad)
  , is_volatile(is_volatile)
  , is_view(false)
  , output_nr(0)
  , pyobj(nullptr) {
  if (!data.defined()) {
    throw std::runtime_error("data is undefined");
  }
}

VariableImpl::VariableImpl(Tensor data, std::shared_ptr<Function> grad_fn)
  : VariableImpl(std::move(data))
{
  requires_grad = grad_fn->is_executable;
  output_nr = grad_fn->num_inputs++;
  _grad_fn = std::move(grad_fn);
}

VariableImpl::VariableImpl(Tensor data)
  : VariableImpl(std::move(data), false, false)
{
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

  std::lock_guard<std::mutex> lock(grad_accumulator_lock);

  auto result = grad_accumulator.lock();
  if (result) return result;

  result = std::make_shared<AccumulateGrad>(Variable(this, true));
  grad_accumulator = result;
  return result;
}

VariableViewImpl::VariableViewImpl(Variable base_, at::Tensor data_)
  : VariableImpl(std::move(data_))
  , base(std::move(base_))
  , expected_version(0) {
  if (!base.defined()) {
    throw std::runtime_error("base is undefined");
  }
  if (base.is_view()) {
    base = base.base();
  }
  is_view = true;
  version_counter = base.version_counter();
  expected_version = version_counter.current_version();
}

std::shared_ptr<Function>& VariableViewImpl::get_grad_fn() {
  std::lock_guard<std::mutex> lock(grad_accumulator_lock);
  if (expected_version != version_counter.current_version()) {
    auto fn = std::make_shared<generated::AsStridedBackward>();
    fn->self_geometry = TensorGeometry(base);
    fn->size = sizes();
    fn->stride = strides();
    fn->storage_offset = Variable(this, true).storage_offset();
    fn->set_flags(Function::flags({ base }));
    fn->num_inputs = 1;
    _grad_fn = std::move(fn);
    expected_version = version_counter.current_version();
  }
  return _grad_fn;
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

}} // namespace torch::autograd
