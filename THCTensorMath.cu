#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCTensorRandom.h"

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>

#define NB_THREADS_PER_BLOCK 256

void THCudaTensor_fill(THCudaTensor *self_, float value)
{
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  thrust::fill(self_data, self_data+THCudaTensor_nElement(self), value);

  THCudaTensor_freeCopyTo(self, self_);
}

void THCudaTensor_zero(THCudaTensor *self_)
{
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  cudaMemset(THCudaTensor_data(self), 0, sizeof(float)*THCudaTensor_nElement(self));
  THCudaTensor_freeCopyTo(self, self_);
}

struct addvalue_functor
{
  const float value;

  addvalue_functor(float value_) : value(value_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    return (x+value);
  }
};

void THCudaTensor_add(THCudaTensor *self_, float value)
{
  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    long size = THCudaTensor_nElement(self);
    thrust::device_ptr<float> self_data(THCudaTensor_data(self));

    thrust::transform(self_data, self_data+size, self_data, addvalue_functor(value));

    THCudaTensor_freeCopyTo(self, self_);
  }
}

void THCudaTensor_mul(THCudaTensor *self_, float value)
{
  THCudaTensor *self = THCudaTensor_newContiguous(self_);

  cublasSscal(THCudaTensor_nElement(self), value, THCudaTensor_data(self), 1);
  THCublasCheck();

  THCudaTensor_freeCopyTo(self, self_);
}

void THCudaTensor_div(THCudaTensor *self_, float value)
{
  THCudaTensor *self = THCudaTensor_newContiguous(self_);

  cublasSscal(THCudaTensor_nElement(self), 1/value, THCudaTensor_data(self), 1);
  THCublasCheck();

  THCudaTensor_freeCopyTo(self, self_);
}

void THCudaTensor_cadd(THCudaTensor *self_, float value, THCudaTensor *src)
{
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src), 3, "size do not match");

  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    src = THCudaTensor_newContiguous(src);

    cublasSaxpy(THCudaTensor_nElement(self), value, THCudaTensor_data(src), 1, THCudaTensor_data(self), 1);
    THCublasCheck();

    THCudaTensor_free(src);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

void THCudaTensor_cadd_tst(THCudaTensor *self_, THCudaTensor* src1, float value, THCudaTensor *src2)
{
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src1), 3, "size do not match");
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src2), 3, "size do not match");

  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);

    src1 = THCudaTensor_newContiguous(src1);
    src2 = THCudaTensor_newContiguous(src2);

    THCudaTensor_copy(self, src1);
    cublasSaxpy(THCudaTensor_nElement(self), value, THCudaTensor_data(src2), 1, THCudaTensor_data(self), 1);
    THCublasCheck();

    THCudaTensor_free(src1);
    THCudaTensor_free(src2);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

void THCudaTensor_cmul(THCudaTensor *self_, THCudaTensor *src)
{
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src), 2, "size do not match");

  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    long size = THCudaTensor_nElement(self);
    src = THCudaTensor_newContiguous(src);
    thrust::device_ptr<float> self_data(THCudaTensor_data(self));
    thrust::device_ptr<float> src_data(THCudaTensor_data(src));

    thrust::transform(src_data, src_data+size, self_data, self_data, thrust::multiplies<float>());

    THCudaTensor_free(src);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

void THCudaTensor_cdiv(THCudaTensor *self_, THCudaTensor *src)
{
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src), 2, "size do not match");

  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    long size = THCudaTensor_nElement(self);
    src = THCudaTensor_newContiguous(src);
    thrust::device_ptr<float> self_data(THCudaTensor_data(self));
    thrust::device_ptr<float> src_data(THCudaTensor_data(src));

    thrust::transform(self_data, self_data+size, src_data, self_data, thrust::divides<float>());

    THCudaTensor_free(src);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

__global__ void THCudaTensor_kernel_addcmul(float *data, float value, float *src1, float *src2, long size)
{
  long k = (((blockIdx.y * gridDim.x) + blockIdx.x) * blockDim.x) + threadIdx.x;
  
  if(k < size)
    data[k] += value*src1[k]*src2[k];
}


void THCudaTensor_addcmul(THCudaTensor *self_, float value, THCudaTensor *src1, THCudaTensor *src2)
{
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src1), 3, "size do not match");
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src2), 4, "size do not match");

  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    long size = THCudaTensor_nElement(self);
    src1 = THCudaTensor_newContiguous(src1);
    src2 = THCudaTensor_newContiguous(src2);

    int nBlockPerRow, nBlockPerColumn, nThreadPerBlock;
    THCudaGetGridSize(&nBlockPerRow, &nBlockPerColumn, &nThreadPerBlock, size);
    dim3 threads(nThreadPerBlock);
    dim3 grid(nBlockPerRow, nBlockPerColumn);

    THCudaTensor_kernel_addcmul<<<grid, threads>>>(THCudaTensor_data(self), value, THCudaTensor_data(src1), THCudaTensor_data(src2), size);

    cudaError errcode = cudaGetLastError();
    if(errcode != cudaSuccess)
      THError(cudaGetErrorString(errcode));

    THCudaTensor_free(src1);
    THCudaTensor_free(src2);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

__global__ void THCudaTensor_kernel_addcdiv(float *data, float value, float *src1, float *src2, long size)
{
  long k = (((blockIdx.y * gridDim.x) + blockIdx.x) * blockDim.x) + threadIdx.x;

  if(k < size)
    data[k] += value*src1[k]/src2[k];
}


void THCudaTensor_addcdiv(THCudaTensor *self_, float value, THCudaTensor *src1, THCudaTensor *src2)
{
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src1), 3, "size do not match");
  THArgCheck(THCudaTensor_nElement(self_) == THCudaTensor_nElement(src2), 4, "size do not match");

  {
    THCudaTensor *self = THCudaTensor_newContiguous(self_);
    long size = THCudaTensor_nElement(self);
    src1 = THCudaTensor_newContiguous(src1);
    src2 = THCudaTensor_newContiguous(src2);

    int nBlockPerRow, nBlockPerColumn, nThreadPerBlock;
    THCudaGetGridSize(&nBlockPerRow, &nBlockPerColumn, &nThreadPerBlock, size);
    dim3 threads(nThreadPerBlock);
    dim3 grid(nBlockPerRow, nBlockPerColumn);

    THCudaTensor_kernel_addcdiv<<<grid, threads>>>(THCudaTensor_data(self), value, THCudaTensor_data(src1), THCudaTensor_data(src2), size);

    cudaError errcode = cudaGetLastError();
    if(errcode != cudaSuccess)
      THError(cudaGetErrorString(errcode));

    THCudaTensor_free(src1);
    THCudaTensor_free(src2);
    THCudaTensor_freeCopyTo(self, self_);
  }
}

float THCudaTensor_dot(THCudaTensor *self, THCudaTensor *src)
{
  THArgCheck(THCudaTensor_nElement(self) == THCudaTensor_nElement(src), 2, "size do not match");

  {
    self = THCudaTensor_newContiguous(self);
    src = THCudaTensor_newContiguous(src);

    float result = cublasSdot(THCudaTensor_nElement(self),
                              THCudaTensor_data(self), 1,
                              THCudaTensor_data(src), 1);

    THCublasCheck();

    THCudaTensor_free(src);
    THCudaTensor_free(self);

    return result;
  }
}

float THCudaTensor_minall(THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  float result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(self), (float)(THInf), thrust::minimum<float>());

  THCudaTensor_free(self);
  return result;
}

float THCudaTensor_maxall(THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  float result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(self), (float)(-THInf), thrust::maximum<float>());

  THCudaTensor_free(self);
  return result;
}

float THCudaTensor_sumall(THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  float result = thrust::reduce(self_data, self_data+THCudaTensor_nElement(self), (float)(0), thrust::plus<float>());

  THCudaTensor_free(self);
  return result;
}

void THCudaTensor_addmv(THCudaTensor *self, float beta, float alpha, THCudaTensor *mat, THCudaTensor *vec)
{
  if( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected");

  if( mat->size[1] != vec->size[0] )
    THError("size mismatch");

  if(self->nDimension != 1)
    THError("size mismatch");

  if( self->size[0] != mat->size[0] )
    THError("size mismatch");

  if(mat->stride[0] == 1)
  {
    cublasSgemv('n', mat->size[0], mat->size[1],
                alpha, THCudaTensor_data(mat), mat->stride[1],
                THCudaTensor_data(vec), vec->stride[0],
                beta, THCudaTensor_data(self), self->stride[0]);
  }
  else if(mat->stride[1] == 1)
  {
    cublasSgemv('t',  mat->size[1], mat->size[0],
                alpha, THCudaTensor_data(mat), mat->stride[0],
                THCudaTensor_data(vec), vec->stride[0],
                beta, THCudaTensor_data(self), self->stride[0]);
  }
  else
  {
    mat = THCudaTensor_newContiguous(mat);
    
    cublasSgemv('t',  mat->size[1], mat->size[0],
                alpha, THCudaTensor_data(mat), mat->stride[0],
                THCudaTensor_data(vec), vec->stride[0],
                beta, THCudaTensor_data(self), self->stride[0]);
    
    THCudaTensor_free(mat);
  }

  THCublasCheck();  
}

void THCudaTensor_addmm(THCudaTensor *self, float beta, float alpha, THCudaTensor *m1, THCudaTensor *m2)
{
  char transpose, transpose_m1, transpose_m2;
  THCudaTensor *self_, *m1_, *m2_;

  if( (m1->nDimension != 2) || (m2->nDimension != 2) ) 
    THError("matrix and matrix expected"); 

  if(self->nDimension != 2)
    THError("size mismatch"); 

  if( (self->size[0] != m1->size[0]) || (self->size[1] != m2->size[1]) || (m1->size[1] != m2->size[0]) ) 
    THError("size mismatch"); 

  /* self */
  if(self->stride[0] == 1)
  {
    transpose = 'n';
    self_ = self;
  }
  else if(self->stride[1] == 1)
  {
    THCudaTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    THCudaTensor_transpose(self, NULL, 0, 1);
    THCudaTensor_transpose(m1, NULL, 0, 1);
    THCudaTensor_transpose(m2, NULL, 0, 1);
    transpose = 't';
    self_ = self;
  }
  else
  {
    transpose = 'n';
    THCudaTensor_transpose(self, NULL, 0, 1);
    self_ = THCudaTensor_newClone(self);
    THCudaTensor_transpose(self, NULL, 0, 1);
    THCudaTensor_transpose(self_, NULL, 0, 1);
  }

  /* m1 */
  if(m1->stride[0] == 1)
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride[1] == 1)
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = 't';
    m1_ = THCudaTensor_newContiguous(m1);
  }

  /* m2 */
  if(m2->stride[0] == 1)
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride[1] == 1)
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = 't';
    m2_ = THCudaTensor_newContiguous(m2);
  }

  /* do the operation */
  cublasSgemm(transpose_m1,
              transpose_m2,
              self_->size[0],
              self_->size[1],
              m1_->size[1],
              alpha,
              THCudaTensor_data(m1_),
              (transpose_m1 == 'n' ? m1_->stride[1] : m1_->stride[0]),
              THCudaTensor_data(m2_),
              (transpose_m2 == 'n' ? m2_->stride[1] : m2_->stride[0]),
              beta,
              THCudaTensor_data(self_),
              self_->stride[1]);

  THCublasCheck();

  /* free intermediate variables */
  if(m1_ != m1)
    THCudaTensor_free(m1_);

  if(m2_ != m2)
    THCudaTensor_free(m2_);

  if(self_ != self)
    THCudaTensor_freeCopyTo(self_, self);

  if(transpose == 't')
  {
    THCudaTensor_transpose(self, NULL, 0, 1);
    THCudaTensor_transpose(m1, NULL, 0, 1);
    THCudaTensor_transpose(m2, NULL, 0, 1);
  }
}

void THCudaTensor_addr(THCudaTensor *self, float alpha, THCudaTensor *vec1, THCudaTensor *vec2)
{
  if( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
    THError("vector and vector expected");

  if(self->nDimension != 2)
    THError("size mismatch");

  if( (self->size[0] != vec1->size[0]) || (self->size[1] != vec2->size[0]) )
    THError("size mismatch");

  if(self->stride[0] == 1)
  {
    cublasSger(vec1->size[0], vec2->size[0],
               alpha, THCudaTensor_data(vec1), vec1->stride[0],
               THCudaTensor_data(vec2), vec2->stride[0],
               THCudaTensor_data(self), self->stride[1]);
  }
  else if(self->stride[1] == 1)
  {
    cublasSger(vec2->size[0], vec1->size[0],
               alpha, THCudaTensor_data(vec2), vec2->stride[0],
               THCudaTensor_data(vec1), vec1->stride[0],
               THCudaTensor_data(self), self->stride[0]);
  }
  else
  {
    THCudaTensor *cself = THCudaTensor_newClone(self);

    cublasSger(vec2->size[0], vec1->size[0],
               alpha, THCudaTensor_data(vec2), vec2->stride[0],
               THCudaTensor_data(vec1), vec1->stride[0],
               THCudaTensor_data(cself), cself->stride[0]);

    THCudaTensor_freeCopyTo(cself, self);
  }

  THCublasCheck();
}

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(NAME, CFUNC)                   \
  struct NAME##_functor                                                \
  {                                                                     \
    __host__ __device__ float operator()(const float& x) const          \
    {                                                                   \
      return CFUNC(x);                                                  \
    }                                                                   \
  };                                                                    \
                                                                        \
  void THCudaTensor_##NAME(THCudaTensor *self_)                         \
  {                                                                     \
    THCudaTensor *self = THCudaTensor_newContiguous(self_);             \
    long size = THCudaTensor_nElement(self);                            \
    thrust::device_ptr<float> self_data(THCudaTensor_data(self));       \
                                                                        \
    thrust::transform(self_data, self_data+size, self_data, NAME##_functor()); \
                                                                        \
    THCudaTensor_freeCopyTo(self, self_);                               \
  }

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log, log)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log1p, log1p)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(exp, exp)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cos, cos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(acos, acos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cosh, cosh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sin, sin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(asin, asin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sinh, sinh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tan, tan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(atan, atan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tanh, tanh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sqrt, sqrt)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(ceil, ceil)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(floor, floor)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(abs, fabs)

struct pow_functor
{
  const float value;

  pow_functor(float value_) : value(value_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    return pow(x, value);
  }
};

void THCudaTensor_pow(THCudaTensor *self_, float value)
{
  THCudaTensor *self = THCudaTensor_newContiguous(self_);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  
  thrust::transform(self_data, self_data+size, self_data, pow_functor(value));

  THCudaTensor_freeCopyTo(self, self_);
}

float THCudaTensor_meanall(THCudaTensor *self)
{
  THArgCheck(self->nDimension > 0, 1, "empty Tensor");
  return THCudaTensor_sumall(self)/THCudaTensor_nElement(self);
}

struct square_functor
{
  const float mean;

  square_functor(float mean_) : mean(mean_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    return (x-mean)*(x-mean);
  }
};

float THCudaTensor_varall(THCudaTensor *self)
{
  self = THCudaTensor_newContiguous(self);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  float mean = THCudaTensor_meanall(self);
  float result = thrust::transform_reduce(self_data, self_data+size, square_functor(mean), (float)0, thrust::plus<float>());

  result = result/(THCudaTensor_nElement(self)-1);

  THCudaTensor_free(self);
  return result;
}

float THCudaTensor_stdall(THCudaTensor *self)
{
  return sqrt(THCudaTensor_varall(self));
}

struct norm_functor
{
  const float exponent;

  norm_functor(float exponent_) : exponent(exponent_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    return pow(fabs(x), exponent);
  }
};

float THCudaTensor_norm(THCudaTensor *self, float value)
{
  self = THCudaTensor_newContiguous(self);
  long size = THCudaTensor_nElement(self);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));

  float result = thrust::transform_reduce(self_data, self_data+size, norm_functor(value), (float)0, thrust::plus<float>());

  THCudaTensor_free(self);
  return pow(result, (float)1.0/value);
}

struct dist_functor
{
  const float exponent;

  dist_functor(float exponent_) : exponent(exponent_) {}

  __host__ __device__ float operator()(const float& x, const float& y) const
  {
    return pow(fabs(x-y), exponent);
  }
};

float THCudaTensor_dist(THCudaTensor *self, THCudaTensor *src, float value)
{
  self = THCudaTensor_newContiguous(self);
  long size = THCudaTensor_nElement(self);
  src = THCudaTensor_newContiguous(src);
  thrust::device_ptr<float> self_data(THCudaTensor_data(self));
  thrust::device_ptr<float> src_data(THCudaTensor_data(src));

  float result = thrust::inner_product(self_data, self_data+size, src_data, (float) 0,thrust::plus<float>(), dist_functor(value));

  THCudaTensor_free(src);
  THCudaTensor_free(self);
  
  return pow(result, (float)1.0/value);
}

void THCudaTensor_rand(THCudaTensor *r_, THLongStorage *size)
{
  THCudaTensor_resize(r_, size, NULL);
  THCudaTensor_uniform(r_, 0, 1);
}

void THCudaTensor_randn(THCudaTensor *r_, THLongStorage *size)
{
  THCudaTensor_resize(r_, size, NULL);
  THCudaTensor_normal(r_, 0, 1);
}
