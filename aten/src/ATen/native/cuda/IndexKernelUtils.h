
#include <cstdint>

namespace at::native {

template <int64_t Alignment>
void vectorized_gather_kernel_launch(char * out, char * inp, int64_t * idx, int num_ind,
                                     int64_t slice_size_in_bytes, int64_t ind_dim_size, int64_t inp_stride_bytes, int64_t out_stride_bytes,
                                     bool allow_neg_indices=false);


}
