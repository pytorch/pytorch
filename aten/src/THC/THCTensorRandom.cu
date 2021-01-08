#include <THC/THCTensorRandom.h>
#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include <THC/THCTensorCopy.h>
#include <THC/THCTensorMath.h>
#include <THC/THCReduceApplyUtils.cuh>
#include <THC/THCTensorRandom.cuh>
#include <ATen/Config.h>

#include <thrust/functional.h>

#define MAX_NUM_BLOCKS 200
#define BLOCK_SIZE 256

#include <THC/generic/THCTensorRandom.cu>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorRandom.cu>
#include <THC/THCGenerateBoolType.h>
