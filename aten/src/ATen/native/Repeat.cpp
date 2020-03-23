#include <ATen/ATen.h>
#include <ATen/native/Repeat.h>
#include <ATen/Parallel.h>
#include <c10/util/SmallVector.h>

static void compute_cpu(int64_t *repeat_ptr, int64_t *cumsum_ptr, int64_t *result_ptr, int64_t size) {
    at::parallel_for(0, size, 1, [&](int64_t i_begin, int64_t i_end) {
        for(int64_t i = i_begin; i < i_end; i++) {
            int64_t end = cumsum_ptr[i];
            int64_t size = repeat_ptr[i];
            int64_t start = end - size;
            for(int64_t j = start; j < end; j++) {
                result_ptr[j] = i;
            }
        }
    });
}

namespace at { namespace native {

Tensor repeat_interleave_cpu(const Tensor &repeat) {
    return repeat_interleave_common<compute_cpu>(repeat);
}

Tensor repeat_interleave(const Tensor &self, const Tensor &repeats, c10::optional<int64_t> dim) {
    Tensor input = self;
    if(!dim) {
        input = self.flatten();
        dim = 0;
    }

    Tensor repeats_ = repeats;
    if (repeats.dim() == 0 || (repeats.dim() == 1 && repeats.size(0) == 1)) {
        repeats_ = repeats.reshape({1}).expand({input.size(dim.value())});
    } else if (repeats.dim() == 1) {
        TORCH_CHECK(repeats.size(0) == input.size(dim.value()), "repeats must have the same size as input along dim")
    } else {
        AT_ERROR("repeats must be 0-dim or 1-dim tensor");
    }

    /* This has the same semantics as:
     *   `input.index_select(dim.value(), at::repeat_interleave(repeats_))`,
     * but implementing with gather is faster than `input.index_select`
     */
    Tensor indices = at::repeat_interleave(repeats_);

    SmallVector<int64_t, 4> indices_sizes(/*n=*/self.dim(), /*value=*/1);
    indices_sizes[dim.value()] = indices.size(0);
    auto self_sizes = self.sizes();
    SmallVector<int64_t, 4> expand_sizes(self_sizes.begin(), self_sizes.end());
    expand_sizes[dim.value()] = indices.size(0);
    indices = indices.view(indices_sizes).expand(expand_sizes);

    return input.gather(dim.value(), indices);
}

Tensor repeat_interleave(const Tensor &self, int64_t repeats, c10::optional<int64_t> dim) {
    return at::native::repeat_interleave(self, at::tensor({repeats}, self.options().dtype(kLong)), dim);
}

}}
