#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>
#include "common.h"

template <typename Dtype, typename Acctype>
struct gatedLinearCSigMul_functor
{
  __device__ void operator()(Dtype *target, const Dtype *sigTensor, const Dtype *mulTensor) const
  {
    const Acctype sigNum = Acctype(1)/(Acctype(1)+ exp(ScalarConvert<Dtype, Acctype>::to(-*sigTensor)));
    const Dtype mulNum = *mulTensor;
    *target = ScalarConvert<Acctype, Dtype>::to(sigNum * mulNum);
  }
};

template <typename Dtype, typename Acctype>
struct gatedLinearDerivativeSecondHalf_functor
{
  __device__ void operator()(Dtype *target, const Dtype *sigTensor, const Dtype *mulTensor) const
  {
    const Acctype sigNum = Acctype(1)/(Acctype(1)+ exp(ScalarConvert<Dtype, Acctype>::to(-*sigTensor)));
    const Dtype mulNum = *mulTensor;
    *target *= ScalarConvert<Acctype, Dtype>::to((Acctype(1) - sigNum) * sigNum * mulNum);
  }
};

#include "generic/GatedLinearUnit.cu"
#include "THCGenerateFloatTypes.h"