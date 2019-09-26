#include <torch/extension.h>

#include <c10/core/Allocator.h>
#include <ATen/CPUGenerator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <c10/util/Half.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>
#include <ATen/core/ATenDispatch.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>

namespace at {

static Tensor empty_complex(IntArrayRef size, const TensorOptions & options, c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(!optional_memory_format.has_value(), "memory format is not supported")
  AT_ASSERT(options.device().is_cpu());

  for (auto x: size) {
    TORCH_CHECK(x >= 0, "Trying to create tensor using size with negative dimension: ", size);
  }
  auto* allocator = at::getCPUAllocator();
  int64_t nelements = at::prod_intlist(size);
  auto dtype = options.dtype();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      dtype,
      nelements,
      allocator->allocate(nelements * dtype.itemsize()),
      allocator,
      /*resizable=*/true);

  auto tensor = detail::make_tensor<TensorImpl>(storage_impl, at::TensorTypeId::ComplexCPUTensorId);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }
  return tensor;
}

static auto& complex_empty_registration = globalATenDispatch().registerOp(
    TensorTypeId::ComplexCPUTensorId,
    "aten::empty.memory_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor",
    &empty_complex);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { }
