#pragma once

namespace at {
namespace mps {

static const char* computing_metal_shaders = R"COMPUTE_METAL(
#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

template <typename index_t>
kernel void compute_mps_kernel(device index_t* _repeat_ptr,
    	device int64_t* _cumsum_ptr,
    	device index_t* _result_ptr,
      device int64_t& _size,
      device NSUInteger& _threadGroupSize,
      uint idx [[thread_position_in_grid]])
    {

      using namespace at::mps;

      int64_t block = 512;
    	int64_t stride = (threadGroupSize * block) / C10_WARP_SIZE;
    	int warp_id = idx / C10_WARP_SIZE;
    	int tid_in_warp = idx % C10_WARP_SIZE;
    	for (int64_t i = warp_id; i < _size; i += stride)
    	{
    		int64_t end = _cumsum_ptr[i];
    		index_t repeat = _repeat_ptr[i];
    		for (int64_t j = start + tid_in_warp; j < end; j += C10_WARP_SIZE)
    		{
    			_repeat_ptr[j] = i;
    		}
    	}
    }

)COMPUTE_METAL";
}
} // namespace at
