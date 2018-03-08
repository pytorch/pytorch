#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <tuple>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace at {
namespace native {

template<typename scalar>
inline scalar IoU(scalar* rawInput, int idx_x, int idx_y)
{
    scalar lr = std::fmin(rawInput[idx_x*4] + rawInput[idx_x*4+2],
                         rawInput[idx_y*4] + rawInput[idx_y*4+2]);
    scalar rl = std::fmax(rawInput[idx_x*4], rawInput[idx_y*4]);
    scalar tb = std::fmin(rawInput[idx_x*4+1] + rawInput[idx_x*4+3],
                         rawInput[idx_y*4+1] + rawInput[idx_y*4+3]);
    scalar bt = std::fmax(rawInput[idx_x*4+1], rawInput[idx_y*4+1]);
    scalar inter = std::fmax(0, lr-rl)*std::fmax(0, tb-bt);
    scalar uni = (rawInput[idx_x*4+2]*rawInput[idx_x*4+3] 
                 + rawInput[idx_y*4+2]*rawInput[idx_y*4+3] - inter);
    return inter/uni;
}


std::tuple<Tensor, Tensor> non_max_suppression_cpu(const Tensor& input,
                              const Tensor& scores,
                              double thresh)
{

  AT_ASSERT(input.ndimension() == 3,
      "First argument should be a 3D Tensor, (batch_sz x n_boxes x 4)");
  AT_ASSERT(scores.ndimension() == 2,
      "Second argument should be a 2D Tensor, (batch_sz x n_boxes)");
  AT_ASSERT(input.size(0) == scores.size(0),
      "First and second arguments must have equal-sized first dimensions");
  AT_ASSERT(input.size(1) == scores.size(1),
      "First and second arguments must have equal-sized second dimensions");
  AT_ASSERT(input.size(2) == 4,
      "First argument dimension 2 must have size 4, and should be of the form [x, y, w, h]");
  AT_ASSERT(input.is_contiguous(), "First argument must be a contiguous Tensor");
  AT_ASSERT(scores.is_contiguous(), "Second argument must be a contiguous Tensor");
  AT_ASSERT(input.type().scalarType() == kFloat || input.type().scalarType() == kDouble,
      "First argument must be Float or Double Tensor")
  AT_ASSERT(scores.type().scalarType() == kFloat || scores.type().scalarType() == kDouble,
      "Second argument must be Float or Double Tensor")
  AT_ASSERT(input.is_contiguous(), "First argument must be a contiguous Tensor");
  AT_ASSERT(scores.is_contiguous(), "Second argument must be a contiguous Tensor");

 
  Tensor sorted_inds = std::get<1>(scores.sort(-1, true));
  //Tensor rawIdx = std::get<1>(scores.sort(-1, true));
  
  auto num_boxes = input.size(1);
  auto batch_size = input.size(0);
  auto mask = input.type().toScalarType(kByte).tensor({batch_size, num_boxes});
  mask.fill_(1);
  auto *rawMask = mask.data<unsigned char>();
  auto *rawIdx = sorted_inds.data<int64_t>();

  if (input.type().scalarType() == kFloat)
  {
    auto *rawInput = input.data<float>();

    for(int batch=0; batch<batch_size; ++batch)
    {
      int pos=batch*num_boxes;
      while(pos < (1+batch)*num_boxes-1)
      {
#pragma omp parallel for
        for(int i=pos+1; i<num_boxes*(1+batch); ++i)
        {
          int idx_x = rawIdx[pos]+num_boxes*batch;
          int idx_y = rawIdx[i]+num_boxes*batch;
          if (IoU(rawInput, idx_x, idx_y) > thresh)
            rawMask[i] = 0;
        }
        ++pos;
        while(pos < (1+batch)*num_boxes-1 and (rawMask[pos] == 0))
          ++pos;
      }
    }
  }
  else
  {
    auto *rawInput = input.data<double>();
    for(int batch=0; batch<batch_size; ++batch)
    {
      int pos=batch*num_boxes;
      while(pos < (1+batch)*num_boxes-1)
      {
#pragma omp parallel for
        for(int i=pos+1; i<num_boxes*(1+batch); ++i)
        {
          int idx_x = rawIdx[pos]+num_boxes*batch;
          int idx_y = rawIdx[i]+num_boxes*batch;
          if (IoU(rawInput, idx_x, idx_y) > thresh)
            rawMask[i] = 0;
        }
        ++pos;
        while(pos < (1+batch)*num_boxes-1 and (rawMask[pos] == 0))
          ++pos;
      }
    }
  }
  //see ./cuda/NonMaxSuppression.cu for comment about return value.
  return std::make_tuple(mask, sorted_inds);
}

}//native
}//at
