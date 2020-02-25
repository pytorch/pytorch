#include <ATen/ATen.h>
#include <aten/src/ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>

#include <torch/csrc/jit/ir.h>

// New fuser includes
#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/fuser/common/utils.h>


#include <iterator>

namespace torch {
namespace jit {

using namespace torch::jit::fuser;

// Defines pure virtual destructor
FusionBackend::~FusionBackend() { }

namespace {

// TODO: ensure tensors are not sparse (for now)
// TODO: ensure tensors are float tensors (for now)
// Returns true iff:
//  - There is at least one input and one output
//  - All inputs are complete tensors or scalars
//  - All outputs are complete tensors
//  - All tensors are on the same device
bool validateNode(const Node* const node) {
  const auto inputs = node->inputs();
  const auto outputs = node->outputs();

  if (inputs.size() == 0 || outputs.size() == 0) {
    return false;
  }

  const auto device = getFusionDevice(node);

  auto lambda = [device](
    const at::ArrayRef<const Value*> values
  , const bool allow_scalars) {
    for (const auto* const val : values) {
      if (val->isCompleteTensor()) {
        const auto cur_device = *(val->type()->expect<TensorType>()->device());
        if (device != cur_device) {
          return false;
        }
      } else if (!allow_scalars || !isScalar(val)) {
        return false;
      }
    }

    return true;
  };

  return (lambda(inputs, true) && lambda(outputs, false));
}

std::mutex& fusionBackendLock() {
  static std::mutex fusion_backends_lock_{};
  return fusion_backends_lock_;
}

static std::unordered_map<at::Device::Type, FusionBackend*>&
getFusionBackendsEx() {
  static std::unordered_map<at::Device::Type, FusionBackend*> fusion_backends;
  return fusion_backends;
}

} // namespace

void registerFusionBackendEx(
    at::Device::Type backend_type,
    FusionBackend* backend) {
  std::lock_guard<std::mutex> guard(fusionBackendLock());
  getFusionBackendsEx()[backend_type] = backend;
}

bool hasFusionBackendEx(at::Device::Type backend_type) {
  std::lock_guard<std::mutex> guard(fusionBackendLock());
  return (getFusionBackendsEx().count(backend_type) > 0);
}

RegisterFusionBackendEx::RegisterFusionBackendEx(
  at::Device::Type backend_type
, FusionBackend* backend) {
  registerFusionBackendEx(backend_type, backend);
}

// Returns true iff the node is fusible
bool isFusible(const Node* const node) {
  //if (!validateNode(node)) {
    //return false;
  //}

  //const auto device_type = getFusionDeviceType(node);
  const auto device_type = c10::kCUDA;

  switch (device_type) {
    case c10::kCPU:
    case c10::kCUDA:
      return (getFusionBackendsEx().count(device_type) > 0) &&
        getFusionBackendsEx()[device_type]->isFusible(node);
    default:
      return false;
  }

  TORCH_CHECK(false, "End of non-void function");
}

// Returns the key corresponding to the fusion
int fuse(const Node* const node) {
  TORCH_CHECK(isFusible(node), "Asked to create an impossible fusion!");

  const auto device_type = getFusionDeviceType(node);

  switch (device_type) {
    case c10::kCPU:
    case c10::kCUDA:
      TORCH_CHECK((getFusionBackendsEx().count(device_type) > 0),
          "Trying to fuse on device type without register FusionBackend!");
      return getFusionBackendsEx()[device_type]->fuse(node);
    default:
      TORCH_CHECK(false, "Trying to fuse on device type that doesn't support fusion!");
  }

  TORCH_CHECK(false, "End of non-void function");
}

void compileFusion(Node* fusion) {
  const auto device_type = getFusionDeviceType(fusion);

  switch (device_type) {
    case c10::kCPU:
    case c10::kCUDA:
      TORCH_CHECK((getFusionBackendsEx().count(device_type) > 0),
          "Trying to compile fusion on device type without register FusionBackend!");
      return getFusionBackendsEx()[device_type]->compileFusion(fusion);
    default:
      TORCH_CHECK(false, "Trying to fuse on device type that doesn't support fusion!");
  }

  TORCH_CHECK(false, "End of function should not be reached!");
}

// Acquires inputs, allocates outputs, and calls fusion
// TODO: outputs should be preallocated in the graph (see fusion pass)
void callFusion(const Node* const fusion, Stack& stack) {
  // Acquires inputs
  const Graph& graph = *fusion->g(attr::Subgraph);
  const auto nInputs = graph.inputs().size();
  at::ArrayRef<IValue> inputs = last(stack, nInputs);
  //Life time issue, can't drop inputs from stack yet, as inputs are `ArrayRef`
  //drop(stack, nInputs);

  // Constructs output
  std::vector<at::Tensor> outputs;
  for (const auto* const output : graph.outputs()) {
    auto type = output->type()->expect<TensorType>();

    const auto device = *(type->device());
    const auto scalar_type = *(type->scalarType());

    auto options = at::TensorOptions()
      .dtype(scalar_type)
      .layout(at::kStrided)
      .device(device)
      .requires_grad(type->requires_grad());

    const auto sizes = extractSizes(type);

    //auto tensor = at::empty({10, 7, 3, 5}, options);
    auto tensor = at::empty(sizes, options);
    outputs.push_back(tensor);
  }

  // Adds outputs to stack
  stack.insert(
    stack.end()
  , outputs.begin()
  , outputs.end());
  // Life time issue, can't remove outputs from stack yet, as outputs is used
  // later in callFusionOnXXX;
  //, std::make_move_iterator(outputs.begin())
  //, std::make_move_iterator(outputs.end()));

  // Calls fusion
  const auto device = *(graph.outputs()[0]->type()->expect<TensorType>()->device());
  switch(device.type()) {
    case c10::kCPU:
    case c10::kCUDA:
      TORCH_CHECK((getFusionBackendsEx().count(device.type()) > 0),
          "Trying to run fusion on device type without register FusionBackend!");
      return getFusionBackendsEx()[device.type()]->callFusion(fusion, outputs, inputs);
    default:
      TORCH_CHECK(false, "Acquired an unknown fusion device type!");
  }
  drop(stack, nInputs);
}

} // namespace jit
} // namespace torch
