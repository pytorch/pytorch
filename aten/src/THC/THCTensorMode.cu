#include <THC/THC.h>
#include <THC/THCThrustAllocator.cuh>
#include <THC/THCTensorTypeUtils.cuh>
#include <THC/THCReduceApplyUtils.cuh>
#include <THC/THCTensor.hpp>
#include <THC/THCStorage.hpp>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <THC/THCTensorMode.cuh>

#include <THC/generic/THCTensorMode.cu>
#include <THC/THCGenerateAllTypes.h>
