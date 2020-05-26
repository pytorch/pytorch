#include <THCUNN/THCUNN.h>
#include <TH/THHalf.h>
#include <THC/THCNumerics.cuh>
#include <THC/THCApply.cuh>
#include <THCUNN/common.h>
#include <ATen/WrapDimUtils.h>

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


template<typename Dtype, typename Acctype>
struct gatedLinearDerivative
{
   const int64_t stride_i_;
   const int64_t stride_gI_;
   gatedLinearDerivative(int64_t stride_i, int64_t stride_gI)
      :stride_i_(stride_i), stride_gI_(stride_gI){}
   __device__ void operator()(Dtype * gI, const Dtype * gO, const Dtype * input) const
   {
      const Dtype * sigTensor = input + stride_i_;
      const Acctype sigNum = Acctype(1)/(Acctype(1)+ exp(ScalarConvert<Dtype, Acctype>::to(-*sigTensor)));
      *gI = ScalarConvert<Acctype, Dtype>::to(sigNum * *gO);
      Dtype * gIsecond = gI + stride_gI_;
      *gIsecond = ScalarConvert<Acctype, Dtype>::to((Acctype(1) - sigNum) * sigNum * *gO * *input);
   }
};

#include <THCUNN/generic/GatedLinearUnit.cu>
#include <THC/THCGenerateFloatTypes.h>
