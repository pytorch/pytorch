#include "THC.h"
#include "THCThrustAllocator.cuh"
#include "THCTensorTypeUtils.cuh"
#include "THCReduceApplyUtils.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "THCTensorMode.cuh"

#include "generic/THCTensorMode.cu"
#include "THCGenerateAllTypes.h"
