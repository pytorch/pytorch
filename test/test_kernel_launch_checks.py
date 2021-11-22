# Owner(s): ["module: unknown"]

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.check_kernel_launches import (
    check_cuda_kernel_launches, check_code_for_cuda_kernel_launches
)


class AlwaysCheckCudaLaunchTest(TestCase):
    def test_check_code(self):
        """Verifies that the regex works for a few different situations"""

        # Try some different spacings
        self.assertEqual(2, check_code_for_cuda_kernel_launches("""
some_function_call<TemplateArg><<<1,2,0,stream>>>(arg1,arg2,arg3);
C10_CUDA_KERNEL_LAUNCH_CHECK();
some_function_call<TemplateArg><<<1,2,0,stream>>>(arg1,arg2,arg3);

some_function_call<TemplateArg><<<1,2,0,stream>>>(arg1,arg2,arg3);
C10_CUDA_KERNEL_LAUNCH_CHECK();
some_function_call<TemplateArg><<<1,2,0,stream>>>(arg1,arg2,arg3);
some_other_stuff;
some_function_call<TemplateArg><<<1,2,0,stream>>>(arg1,arg2,arg3);
C10_CUDA_KERNEL_LAUNCH_CHECK();
some_function_call<TemplateArg><<<1,2,0,stream>>> (arg1,arg2,arg3);
C10_CUDA_KERNEL_LAUNCH_CHECK();
some_function_call<TemplateArg><<<1,2,0,stream>>> ( arg1 , arg2 , arg3 ) ;

    C10_CUDA_KERNEL_LAUNCH_CHECK();
        """))

        # Does it work for macros?
        self.assertEqual(0, check_code_for_cuda_kernel_launches(r"""
#define SOME_MACRO(x) some_function_call<<<1,2>>> ( x ) ;  \
    C10_CUDA_KERNEL_LAUNCH_CHECK();

#define SMALL_INDEX(TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM)  \
  indexAddSmallIndex<TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM> \
    <<<smallIndexGrid, smallIndexBlock, 0, stream>>>(                                \
      selfInfo, sourceInfo, indexInfo,                                               \
      selfAddDim, sourceAddDim, sliceSize, selfAddDimSize);                          \
  C10_CUDA_KERNEL_LAUNCH_CHECK();
        """))

        # Does it work for lambdas?
        self.assertEqual(1, check_code_for_cuda_kernel_launches(r"""
            rrelu_with_noise_cuda_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
                    numel,
                    rng_engine_inputs,
                    output_data,
                    input_data,
                    noise_data,
                    lower,
                    upper,
                    [] __device__ (curandStatePhilox4_32_10_t* state) {
                    return curand_uniform2_double(state);
                    });
                    C10_CUDA_KERNEL_LAUNCH_CHECK();

            rrelu_with_noise_cuda_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
                    numel,
                    rng_engine_inputs,
                    output_data,
                    input_data,
                    noise_data,
                    lower,
                    upper,
                    [] __device__ (curandStatePhilox4_32_10_t* state) {
                    return curand_uniform2_double(state);
                    });
                    uh oh;
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
        """))

    def test_check_cuda_launches(self):
        unsafeLaunchesCount = check_cuda_kernel_launches()
        self.assertTrue(unsafeLaunchesCount == 0)


if __name__ == '__main__':
    run_tests()
