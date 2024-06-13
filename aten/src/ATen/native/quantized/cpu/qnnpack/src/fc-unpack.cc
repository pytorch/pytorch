#include <pytorch_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/pack.h>
#include <qnnpack_func.h>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace qnnpack {
// For runtime quantization unpacking.
void PackBMatrix::unpackWeights(
  const uint8_t* kernel_zero_points,
  int8_t* kernel
) const {
  union {
    void* const as_void_ptr;
    uint8_t* as_uint8_ptr;
    int32_t* as_int32_ptr;
  } packed = {packed_weights_};

  // C = A * B
  // A = M*K
  // B = K*N
  const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
  const uint32_t kr = pytorch_qnnp_params.q8conv.kr;

  // Convert prepacked weight to original weight / bias.
  for (size_t nr_block_start = 0; nr_block_start < output_channels_; nr_block_start += nr) {
    const size_t nr_block_size = min(output_channels_ - nr_block_start, nr);
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      packed.as_int32_ptr++;
    }
    packed.as_int32_ptr += (nr - nr_block_size);
    for (size_t kr_block_start = 0; kr_block_start < input_channels_; kr_block_start += kr) {
      const size_t kr_block_size = min(input_channels_ - kr_block_start, kr);
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
             kr_block_offset++) {
          kernel[(nr_block_start + nr_block_offset) * input_channels_ +
          (kr_block_start + kr_block_offset)] = *(packed.as_uint8_ptr++);
        }
        if (kernel_zero_points != 0) {
          for (size_t kr_block_offset = 0; kr_block_offset < (kr - kr_block_size);
               kr_block_offset++) {
            packed.as_uint8_ptr++;
          }
        } else {
          packed.as_uint8_ptr += (kr - kr_block_size);
        }
      }
      if (kernel_zero_points != 0) {
        size_t remaining_nr_blocks = ((nr - nr_block_size) & (nr - 1));
        for (size_t nr_block_offset = 0; nr_block_offset < remaining_nr_blocks;
             nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            packed.as_uint8_ptr++;
          }
        }
      } else {
        packed.as_uint8_ptr += ((nr - nr_block_size) & (nr - 1)) * kr;
      }
    }
  }

}

} // namespace qnnpack
