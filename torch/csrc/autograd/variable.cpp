#include "torch/csrc/autograd/variable.h"

#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/autograd/functions/accumulate_grad.h"

using namespace at;

namespace torch { namespace autograd {

VariableTensor::VariableTensor(Tensor data_, bool requires_grad, bool is_volatile)
  : TensorImpl(getType(data_))
  , data(std::move(data_))
  , grad()
  , version_counter(new VariableVersion())
  , requires_grad(requires_grad)
  , is_volatile(is_volatile)
  , output_nr(0)
  , pyobj(nullptr) {
  if (!data.defined()) {
    throw std::runtime_error("data is undefined");
  }
}

VariableTensor::VariableTensor(Tensor data, std::shared_ptr<Function> grad_fn)
  : VariableTensor(std::move(data))
{
  this->grad_fn = grad_fn;
  requires_grad = grad_fn->is_executable;
  output_nr = grad_fn->num_inputs++;
}

VariableTensor::VariableTensor(Tensor data)
  : VariableTensor(std::move(data), false, false)
{
}

VariableTensor::~VariableTensor() {
}

const char * VariableTensor::toString() const {
  return "Variable";
}

IntList VariableTensor::sizes() {
  return data.sizes();
}

int64_t VariableTensor::dim() {
  return data.dim();
}

const char * VariableTensor::typeString() {
  return "VariableType";
}

void * VariableTensor::unsafeGetTH(bool retain) {
  return data.unsafeGetTH(retain);
}

IntList VariableTensor::strides() {
  return data.strides();
}

Scalar VariableTensor::localScalar() {
  return data.pImpl->localScalar();
}

void VariableTensor::assign_(Scalar s) {
  data.assign_(s);
}

std::shared_ptr<Function> VariableTensor::get_grad_accumulator() {
  if (grad_fn) {
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

Type* VariableTensor::getType(const Tensor& tensor)
{
  if (!tensor.defined()) {
    throw std::runtime_error("tensor is undefined");
  }
  return getType(tensor.type());
}

Type* VariableTensor::getType(const Type& baseType)
{
  static VariableTypes vt;
  return vt.types[static_cast<int>(baseType.ID())].get();
}

}} // namespace torch::autograd
