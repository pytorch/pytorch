#include "ATen/NativeFunctions.h"
#include <tuple>

namespace at {
namespace native {

__device__ __forceinline__ float fmin(float a, float b)
{
  return a > b ? b : a;
}

__device__ __forceinline__ float fmax(float a, float b)
{
  return a > b ? a : b;
}

__device__ __forceinline__ float IoU(const float* box_x, const float* box_y)
{
  // Calculate IoU between the boxes.
  float rightmost_l = fmax(box_x[0], box_y[0]);
  float leftmost_r = fmin(box_x[0] + box_x[2], box_y[0] + box_y[2]);
  float delta_x = fmax(0., leftmost_r - rightmost_l);

  float bottommost_tp = fmax(box_x[1], box_y[1]);
  float topmost_b = fmin(box_x[1] + box_x[3], box_y[1] + box_y[3]);
  float delta_y = fmax(0., topmost_b - bottommost_tp);

  float uni = box_x[2] * box_x[3] + box_y[2] * box_y[3];

  return delta_x * delta_y / (uni - delta_x * delta_y);

}

__global__ void nms_kernel(unsigned char* mask, 
                          const float* boxes,
                          const int64_t* inds,
                          const int64_t num_boxes,
                          const float thresh)
{
  
  int col = 0;
  while(col < num_boxes-1)
  {
#pragma unroll
    for(int i = threadIdx.x; i < num_boxes-1; i+=blockDim.x)
      if(i >= col)
      {
        float iou = IoU(&boxes[4*inds[i+1+num_boxes*blockIdx.x] + 4*num_boxes*blockIdx.x],
                        &boxes[4*inds[col+num_boxes*blockIdx.x] + 4*num_boxes*blockIdx.x]);
        mask[i+1+blockIdx.x*num_boxes] *= (iou>thresh) ? 0 : 1;
      }
    __syncthreads();
    ++col;
    while((col < num_boxes - 1) && (mask[col+blockIdx.x*num_boxes]==0))
      ++col;
  }
}

std::tuple<Tensor, Tensor> non_max_suppression_cuda(
                                const Tensor& input,
                                const Tensor& scores,
                                const double thresh)
{
  AT_ASSERT(input.ndimension() == 3,
            "First argument should be a 3D Tensor, (batch_sz x n_boxes x 4)");
  AT_ASSERT(scores.ndimension() == 2,
            "Second argument should be a 2D Tensor, (batch_sz x n_boxes)");
  AT_ASSERT(input.size(0) == scores.size(0),
            "First and second arguments must have equal-sized first dimension");
  AT_ASSERT(input.size(1) == scores.size(1),
            "First and second arguments must have equal-sized second dimension");
  AT_ASSERT(input.size(2) == 4,
           "First argument dimension 2 must have size 4, and should be of the form [x, y, w, h]");
  AT_ASSERT(input.is_contiguous(), "First argument must be a contiguous Tensor");
  AT_ASSERT(scores.is_contiguous(), "Second argument must be a contiguous Tensor");


  auto num_boxes = input.size(1);
  auto batch_size = input.size(0);
  auto mask = input.type().toScalarType(kByte).tensor({batch_size, num_boxes});
  mask.fill_(1);
  
  //need the indices of the boxes sorted by score.
  Tensor sorted_inds = std::get<1>(scores.sort(-1, true));


  dim3 mask_block(512); //would be nice to have 1024 here for gpus that support it,
                        //but not sure how to do this cleanly without calling
                        //cudaGetDeviceProperties in the funcion body...

  dim3 mask_grid(batch_size);
  nms_kernel<<<mask_grid, mask_block, 0, globalContext().getCurrentCUDAStream()>>>(
                                    mask.data<unsigned char>(),
                                    input.data<float>(),
                                    sorted_inds.data<int64_t>(),
                                    num_boxes,
                                    thresh);
  AT_ASSERT(cudaGetLastError() == cudaSuccess, "nms_mask_kernel failed");

  //It's not entirely clear what the best thing to return is here. The algorithm will
  //produce a different number of boxes for each batch, so there is no obvious way of
  //way of returning the surving boxes/indices as a tensor. Returning a mask on the
  //sorted boxes together with the sorted indices seems reasonable; that way, the user
  //can easily take the N highest-scoring surviving boxes to form a tensor if they wish. 
  return std::make_tuple(mask, sorted_inds);
}

}}
