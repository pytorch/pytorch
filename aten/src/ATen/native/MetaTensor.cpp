#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at {
namespace native {

// Will be promoted to a public API later, but not now
Tensor empty_meta(
  IntArrayRef size,
  c10::optional<ScalarType> dtype,
  c10::optional<Layout> layout,
  c10::optional<Device> device,
  c10::optional<bool> pin_memory,
  c10::optional<c10::MemoryFormat> memory_format
) {
  // TODO: deduplicate this logic with empty_cpu

  auto tensor = detail::make_tensor<TensorImpl>(
    // NB: We include the computed dispatch key, not because it will actually
    // participate in dispatch, but so that tests like is_sparse/is_cuda
    // give the correct result (a CUDA meta tensor "is cuda").  If we don't
    // like this, remove the computeDispatchKey line
    DispatchKeySet{DispatchKey::Meta, computeDispatchKey(dtype, layout, device)},
    scalarTypeToTypeMeta(dtype_or_default(dtype)),
    device
  );
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  auto memory_format_ = memory_format.value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format_);

  return tensor;
}

} // namespace native
} // namespace at
