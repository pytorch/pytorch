#include <ATen/ATen.h>

#include <vector>

namespace at {
namespace native {

std::vector<Tensor> lab_chunk(const Tensor& self, int64_t chunks, int64_t _dim) {
    TORCH_CHECK(chunks > 0, "torch.lab_chunk expects chunks greater than 0,"
        " but got chunks ", chunks);

    const int64_t dim = maybe_wrap_dim(_dim, self.dim());

    /* Integer ceiling division; computes the smallest chunk_size such that
     * chunk_size * chunks >= self.size(dim).
     */
    const int64_t chunk_size = (self.size(dim) + chunks - 1) / chunks;

    std::vector<Tensor> out;

    std::vector<int64_t> sizes = self.sizes().vec();

    for (int64_t offset = 0; offset < self.size(dim); offset += chunk_size) {
        /* The last chunk will be smaller if the tensor size along dimension dim
         * is not divisible by chunks (and therefore not divisible by chunk_size).
         */
        sizes[dim] = std::min<int64_t>(chunk_size, self.size(dim) - offset);

        out.push_back(self.as_strided(sizes, self.strides(),
              self.storage_offset() + offset * self.stride(dim)));
    }

    return out;
}

Tensor lab_diagonal(const Tensor& self, int64_t offset, int64_t _dim1, int64_t _dim2) {
    int64_t dim1 = maybe_wrap_dim(_dim1, self.dim());
    int64_t dim2 = maybe_wrap_dim(_dim2, self.dim());

    TORCH_CHECK(dim1 != dim2, "torch.lab_diagonal: dim1 and dim2 must be distinct,"
        " but got dim1 ", _dim1, " dim2 ", _dim2);

    std::vector<int64_t> sizes = self.sizes().vec();

    /* Negative offsets are below the main diagonal, adjusting the coordinates in dim1.
     * Positive offsets are above the main diagonal, adjusting the coordinates in dim2.
     */
    const int64_t storage_offset = self.storage_offset()
      + std::abs(offset) * self.stride(offset < 0 ? dim1 : dim2);

    const int64_t diag_sz = std::min<int64_t>(
        sizes[dim1] - std::max<int64_t>(static_cast<int64_t>(0), -offset),
        sizes[dim2] - std::max<int64_t>(static_cast<int64_t>(0), offset)
    );

    /* Construct adjusted dim sizes and strides, erasing dim1 and dim2
     * and appending the new dimension to the end of the shape.
     */
    if (dim1 > dim2) {
        std::swap(dim1, dim2);
    }

    sizes.erase(sizes.begin() + dim2);
    sizes.erase(sizes.begin() + dim1);
    sizes.push_back(diag_sz);

    std::vector<int64_t> strides = self.strides().vec();
    const int64_t combined_stride = strides[dim1] + strides[dim2];

    strides.erase(strides.begin() + dim2);
    strides.erase(strides.begin() + dim1);
    strides.push_back(combined_stride);

    return self.as_strided(sizes, strides, storage_offset);
}

}} // at::native
