#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/ExpandUtils.h>
#include "c10/core/MemoryFormat.h"

namespace at {
namespace native {

static ptrdiff_t dataOffset(const Tensor& tensor, ptrdiff_t linearIndex) {
  auto size = tensor.sizes();
  auto stride = tensor.strides();
  int nDim = tensor.dim();
  ptrdiff_t dataOffset = 0;
  for (int i = nDim - 1; i >= 0; i--) {
    dataOffset += (linearIndex % size[i]) * stride[i];
    linearIndex /= size[i];
  }
  return dataOffset;
}

static inline int64_t wrapLinearIndex(int64_t linearIndex, int64_t numel) {
  return linearIndex < 0 ? linearIndex + numel : linearIndex;
}

static inline void checkLinearIndex(int64_t linearIndex, int64_t numel) {
  TORCH_CHECK(linearIndex < numel && linearIndex >= -numel, "out of range: %d out of %d", (int)linearIndex, (int)numel);
}

void take_cpu_out_template(
    Tensor& output,
    Tensor const& input,
    Tensor const& index)
{
    TORCH_CHECK(output.device().type() == at::kCPU, "device type of output (", output.device().type(), ") is not on the CPU");
    TORCH_CHECK(input.device().type() == at::kCPU, "device type of input (", input.device().type(), ") is not on the CPU");
    TORCH_CHECK(index.device().type() == at::kCPU, "device type of index (", index.device().type(), ") is not on the CPU");

    TORCH_CHECK(output.layout() == Layout::Strided, "take() only supports strided layout, got layout: ", output.layout(), " on output tensor");
    TORCH_CHECK(input.layout() == Layout::Strided, "take() only supports strided layout, got layout: ", input.layout(), " on input tensor");
    TORCH_CHECK(index.layout() == Layout::Strided, "take() only supports strided layout, got layout: ", index.layout(), " on index tensor");

    TORCH_CHECK(output.scalar_type() == input.scalar_type(), "output and input scalar type must match. but got different types: ", output.scalar_type(), " and ", input.scalar_type());
    TORCH_CHECK(index.scalar_type() == kLong, "index must be an int64 tensor");

    output.resize_(index.sizes());
    auto output_contiguous = output.contiguous();
    auto index_continuous = index.contiguous();
    bool is_contiguous = input.is_contiguous();
    auto input_size = input.numel();

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Bool, input.scalar_type(), "take_cpu", [&] {
        auto output_data = output_contiguous.data_ptr<scalar_t>();
        auto input_data = input.data_ptr<scalar_t>();
        auto index_data = index.data_ptr<int64_t>();

        // Exceptions must not be thrown across parallel sections, so we
        // record the position of the invalid index and throw the exception after the
        // loop.
        std::atomic<int64_t> invalidIdxPos(-1);

        at::parallel_for(0, index.numel(), at::internal::GRAIN_SIZE,
            [&](int64_t start, int64_t end) {
            for (auto i = start; i < end; i++) {
                int64_t idx = index_data[i];
                if (idx < input_size && idx >= -input_size) {
                    idx = wrapLinearIndex(idx, input_size);
                    if (is_contiguous) {
                        output_data[i] = input_data[idx];
                    } else {
                        output_data[i] = input_data[dataOffset(input, idx)];
                    }
                } else {
                    int64_t tmp = -1;
                    invalidIdxPos.compare_exchange_strong(tmp, i);
                }
            }
        });

        if (invalidIdxPos >= 0) {
            checkLinearIndex(index_data[invalidIdxPos], input_size);
        }

        //TODO: what to do with this
        //THLongTensor_free(index);
        //THTensor_(freeCopyTo)(dst, r_);
    });
}

Tensor take_cpu(const Tensor& self, const Tensor& index) {
    auto output = at::empty(index.sizes(), self.options());
    take_cpu_out_template(output, self, index);
    return output;
}

Tensor& take_out_cpu(Tensor& out, const Tensor& self, const Tensor& index) {
    take_cpu_out_template(out, self, index);
    return out;
}

} // at::native
} // at
