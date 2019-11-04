#include <ATen/ATen.h>
#include <ATen/native/ScatterGather.h>

namespace at { namespace native {
namespace {

void gather_kernel_cpu(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) {
  int64_t num_dims = std::max<int64_t>(self.dim(), 1);
  std::vector<int64_t> result_sizes = result.sizes().vec();
  std::vector<int64_t> result_strides = result.strides().vec();
  std::vector<int64_t> self_strides = self.strides().vec();
  std::vector<int64_t> index_strides = index.strides().vec();
  ensure_nonempty(result_strides);
  ensure_nonempty(self_strides);
  ensure_nonempty(index_strides);
  ensure_nonempty(result_sizes);

  int64_t elems_per_row = (index.dim() == 0 ? 1 : index.size(dim));
  int64_t self_dim_size = (self.dim() == 0 ? 1 : self.size(dim));

  AT_DISPATCH_ALL_TYPES_AND2(ScalarType::Bool, ScalarType::Half, self.scalar_type(), "gather_out_cpu", [&](){
    scalar_t *result_data = result.data_ptr<scalar_t>();
    scalar_t *self_data = self.data_ptr<scalar_t>();
    int64_t *index_data = index.data_ptr<int64_t>();
    if (result.numel() == 0) {
      return;
    }
    bool finished = false;
    std::vector<int64_t> counter(num_dims, 0);

    int64_t result_dim_stride = result_strides[dim];
    int64_t index_dim_stride = index_strides[dim];
    int64_t self_dim_stride = self_strides[dim];

    while(!finished) {
      for(int64_t j = 0; j < elems_per_row; j++) {
        int64_t index_value = *(index_data + j * index_dim_stride);
        TORCH_CHECK(index_value >= 0 && index_value < self_dim_size, "Invalid index in gather: out of range");
        *(result_data + j * result_dim_stride) = *(self_data + index_value * self_dim_stride);
      }
      if(num_dims == 1) {
        break;
      }
      for(int64_t i = 0; i < num_dims; i++) {
        if(i == dim) {
          if(i == num_dims - 1) {
            finished = true;
            break;
          }
          continue;
        }
        counter[i]++;
        result_data += result_strides[i];
        self_data += self_strides[i];
        index_data += index_strides[i];
        if(counter[i] == result_sizes[i]) {
          if(i == num_dims - 1) {
            finished = true;
            break;
          }
          int64_t size = result_sizes[i];
          result_data -= size * result_strides[i];
          self_data -= size * self_strides[i];
          index_data -= size * index_strides[i];
          counter[i] = 0;
        } else {
          break;
        }
      }
    }
  });
}

} // anonymous namespace

REGISTER_DISPATCH(gather_stub, &gather_kernel_cpu);

}}  // namespace at::native