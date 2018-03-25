#pragma once

#include <sstream>
#include "ATen/TensorUtils.h"

namespace at {

/*
 * The basic strategy for apply is as follows:
 *
 * 1. Starting with the outermost index, loop until we reach a dimension where the
 * data is no longer contiguous, i.e. the stride at that dimension is not equal to
 * the size of the tensor defined by the outer dimensions. Let's call this outer
 * (contiguous) tensor A. Note that if the Tensor is contiguous, then A is equal
 * to the entire Tensor. Let's call the inner tensor B.
 *
 * 2. We loop through the indices in B, starting at its outermost dimension. For
 * example, if B is a 2x2 matrix, then we do:
 *
 * B[0][0]
 * B[0][1]
 * B[1][0]
 * B[1][1]
 *
 * We set the offset into the underlying storage as (storageOffset + stride_B * index_B),
 * i.e. basically we compute the offset into the storage as we would normally for a
 * Tensor. But because we are guaranteed the subsequent data is contiguous in memory, we
 * can simply loop for sizeof(A) iterations and perform the operation, without having to
 * follow the order described by the strides of A.
 *
 * 3. As an optimization, we merge dimensions of A that are contiguous in memory. For
 * example, if A is a 3x3x3x3 tensor narrowed from a 3x3x4x3 tensor, then the first two
 * dimensions can be merged for the purposes of APPLY, reducing the number of nested
 * loops.
 */

// TODO: turn this macro into a proper template
#define __ATH_TENSOR_APPLYX_PREAMBLE(TYPE, ATENSOR, DIM, ALLOW_CONTIGUOUS) \
  TYPE *ATENSOR##_data = NULL; \
  int64_t *ATENSOR##_counter = NULL, *ATENSOR##_sizes = NULL, *ATENSOR##_strides = NULL, *ATENSOR##_dimOffset = NULL; \
  int64_t ATENSOR##_stride = 0, ATENSOR##_size = 0, ATENSOR##_dim = 0, ATENSOR##_i; \
  int ATENSOR##_contiguous = ALLOW_CONTIGUOUS && DIM < 0; \
\
  if(ATENSOR.sizes().equals({0})) \
    TH_TENSOR_APPLY_hasFinished = true; \
  else \
  { \
    ATENSOR##_data = ATENSOR.data<TYPE>(); \
    ATENSOR##_size = 1; \
    ATENSOR##_stride = 1; \
    for(ATENSOR##_i = ATENSOR.dim() - 1; ATENSOR##_i >= 0; ATENSOR##_i--) { \
      if(ATENSOR.sizes()[ATENSOR##_i] != 1) { \
        if(ATENSOR.strides()[ATENSOR##_i] == ATENSOR##_size && ATENSOR##_i != DIM) \
          ATENSOR##_size *= ATENSOR.sizes()[ATENSOR##_i]; \
        else{ \
          ATENSOR##_contiguous = 0; \
          break; \
        } \
      } \
    } \
    if (!ATENSOR##_contiguous) { \
      /* Find the dimension of contiguous sections */ \
      ATENSOR##_dim = 1; \
      for(ATENSOR##_i = ATENSOR.dim() - 2; ATENSOR##_i >= 0; ATENSOR##_i--) \
      { \
        if(ATENSOR.strides()[ATENSOR##_i] != ATENSOR.strides()[ATENSOR##_i+1] * ATENSOR.sizes()[ATENSOR##_i+1] || ATENSOR##_i == DIM || ATENSOR##_i+1 == DIM) \
          ATENSOR##_dim++; \
      } \
      /* Allocate an array of 3*dim elements, where dim is the number of contiguous sections */ \
      ATENSOR##_counter = new int64_t[3*ATENSOR##_dim]; \
      ATENSOR##_sizes = ATENSOR##_counter + ATENSOR##_dim; \
      ATENSOR##_strides = ATENSOR##_counter + 2*ATENSOR##_dim; \
      TH_TENSOR_dim_index = ATENSOR##_dim-1; \
      ATENSOR##_dimOffset = (DIM == ATENSOR.dim()-1) ? &ATENSOR##_i : &ATENSOR##_counter[DIM]; \
      ATENSOR##_sizes[TH_TENSOR_dim_index] = ATENSOR.sizes()[ATENSOR.dim()-1]; \
      ATENSOR##_strides[TH_TENSOR_dim_index] = ATENSOR.strides()[ATENSOR.dim()-1]; \
      /* ATENSOR##_counter tracks where we are in the storage. The offset into the */ \
      /* storage is given by storage_offset + (i * j), where i is the stride */ \
      /* vector and j is tensor_counter vector. This sets the starting position for the loop. */ \
      for(ATENSOR##_i = ATENSOR##_dim-1; ATENSOR##_i >= 0; --ATENSOR##_i) { \
        ATENSOR##_counter[ATENSOR##_i] = 0; \
      } \
      for(ATENSOR##_i = ATENSOR.dim()-2; ATENSOR##_i >= 0; --ATENSOR##_i) { \
        if (ATENSOR.strides()[ATENSOR##_i] == ATENSOR.strides()[ATENSOR##_i+1] * ATENSOR.sizes()[ATENSOR##_i+1] && ATENSOR##_i != DIM && ATENSOR##_i+1 != DIM) { \
          ATENSOR##_sizes[TH_TENSOR_dim_index] = ATENSOR.sizes()[ATENSOR##_i] * ATENSOR##_sizes[TH_TENSOR_dim_index]; \
          if (DIM != ATENSOR.dim()-1 && ATENSOR##_i < DIM) \
            ATENSOR##_dimOffset--; \
        } else { \
          --TH_TENSOR_dim_index; \
          ATENSOR##_sizes[TH_TENSOR_dim_index] = ATENSOR.sizes()[ATENSOR##_i]; \
          ATENSOR##_strides[TH_TENSOR_dim_index] = ATENSOR.strides()[ATENSOR##_i]; \
        } \
      } \
      /* Size of the inner most section */ \
      ATENSOR##_size = ATENSOR##_sizes[ATENSOR##_dim-1]; \
      /* Stride of the inner most section */ \
      ATENSOR##_stride = ATENSOR##_strides[ATENSOR##_dim-1]; \
    } \
  } \
  ATENSOR##_i = 0;

// TODO: turn this macro into a proper template
#define  __ATH_TENSOR_APPLYX_UPDATE_COUNTERS(ATENSOR, ALWAYS_UPDATE) \
  if(ATENSOR##_i == ATENSOR##_size || ALWAYS_UPDATE) \
  { \
    if(ATENSOR##_contiguous) \
      break; \
\
    if(ATENSOR##_dim == 1) \
       break; \
\
    /* Reset pointer to beginning of loop */ \
    ATENSOR##_data -= ATENSOR##_size*ATENSOR##_stride; \
    for(ATENSOR##_i = ATENSOR##_dim-2; ATENSOR##_i >= 0; ATENSOR##_i--) \
    { \
      ATENSOR##_counter[ATENSOR##_i]++; \
      /* Jump ahread by the stride of this dimension */ \
      ATENSOR##_data += ATENSOR##_strides[ATENSOR##_i]; \
\
      if(ATENSOR##_counter[ATENSOR##_i]  == ATENSOR##_sizes[ATENSOR##_i]) \
      { \
        if(ATENSOR##_i == 0) \
        { \
          TH_TENSOR_APPLY_hasFinished = true; \
          break; \
        } \
          else \
        { \
          /* Reset the pointer to the beginning of the chunk defined by this dimension */ \
          ATENSOR##_data -= ATENSOR##_counter[ATENSOR##_i]*ATENSOR##_strides[ATENSOR##_i]; \
          ATENSOR##_counter[ATENSOR##_i] = 0; \
        } \
      } \
      else \
        break; \
    } \
    ATENSOR##_i = 0; \
  }

template <typename scalar1, typename scalar2, typename Op>
void CPU_tensor_apply2_dim(Tensor& tensor1, Tensor& tensor2, int64_t dim, Op op) {
  checkBackend("CPU_tensor_apply2", {tensor1, tensor2}, Backend::CPU);
  bool TH_TENSOR_APPLY_hasFinished = false;
  int64_t TH_TENSOR_dim_index = 0;
  __ATH_TENSOR_APPLYX_PREAMBLE(scalar1, tensor1, dim, 1)
  __ATH_TENSOR_APPLYX_PREAMBLE(scalar2, tensor2, dim, 1)
  auto t1_numel = tensor1.numel();
  auto t2_numel = tensor2.numel();
  if(t1_numel != t2_numel) {
    std::ostringstream oss;
    oss << "inconsistent tensor size, expected " << tensor1.sizes() << " and " << tensor2.sizes()
        << " to have the same number of elements, but got " << t1_numel << " and " << t2_numel << " elements respectively";
    throw std::runtime_error(oss.str());
  }
  while(!TH_TENSOR_APPLY_hasFinished)
  {
    /* Loop through the inner most region of the Tensor */
    for(; tensor1_i < tensor1_size && tensor2_i < tensor2_size; tensor1_i++, tensor2_i++, tensor1_data += tensor1_stride, tensor2_data += tensor2_stride)
    {
      op(*tensor1_data, *tensor2_data);
    }
    __ATH_TENSOR_APPLYX_UPDATE_COUNTERS(tensor1, 0)
    __ATH_TENSOR_APPLYX_UPDATE_COUNTERS(tensor2, 0)
  }
  if(tensor1_counter != NULL)
    delete [] tensor1_counter;
  if(tensor2_counter != NULL)
    delete [] tensor2_counter;
}

/*
  Apply a pointwise operator to two tensors.

  The calling convention for op is a function/functor that takes takes two references to
  type scalar; at least one of these references should be non-const in order to write the output.
  For example, to compute a = b^2, op would be of the form:
  [](scalar &a_val, const scalar &b_val) { a_val = b_val * b_val; };
*/
template<typename scalar1, typename scalar2, typename Op>
void CPU_tensor_apply2(Tensor tensor1, Tensor tensor2, Op op) {
  CPU_tensor_apply2_dim<scalar1, scalar2, Op>(tensor1, tensor2, -1, op);
}

template<typename scalar1, typename scalar2, typename scalar3, typename Op>
void CPU_tensor_apply3_dim(Tensor &tensor1, Tensor& tensor2, Tensor& tensor3, int64_t dim, Op op) {
  checkBackend("CPU_tensor_apply3", {tensor1, tensor2, tensor3}, Backend::CPU);
  bool TH_TENSOR_APPLY_hasFinished = false;
  int64_t TH_TENSOR_dim_index = 0;
  __ATH_TENSOR_APPLYX_PREAMBLE(scalar1, tensor1, dim, 1)
  __ATH_TENSOR_APPLYX_PREAMBLE(scalar2, tensor2, dim, 1)
  __ATH_TENSOR_APPLYX_PREAMBLE(scalar3, tensor3, dim, 1)

  int elements_equal = 1;
  auto t1_numel = tensor1.numel();
  auto t2_numel = tensor2.numel();
  auto t3_numel = tensor3.numel();
  if(t1_numel!= t2_numel) {
    elements_equal = 0;
  } else if(t1_numel != t3_numel) {
    elements_equal = 0;
  }
  if (elements_equal == 0) {
    std::ostringstream oss;
    oss << "inconsistent tensor size, expected " << tensor1.sizes() << ", " << tensor2.sizes() << ", and " << tensor3.sizes()
        << " to have the same number of elements, but got " << t1_numel << ", " << t2_numel << ", and " << t3_numel << " elements respectively";
    throw std::runtime_error(oss.str());
  }

  while(!TH_TENSOR_APPLY_hasFinished)
  {
    /* Loop through the inner most region of the Tensor */
    for(; tensor1_i <  tensor1_size && tensor2_i < tensor2_size && tensor3_i < tensor3_size; tensor1_i++, tensor2_i++, tensor3_i++, tensor1_data += tensor1_stride, tensor2_data += tensor2_stride, tensor3_data += tensor3_stride)
    {
      op(*tensor1_data, *tensor2_data, *tensor3_data);
    }
    __ATH_TENSOR_APPLYX_UPDATE_COUNTERS(tensor1, 0)
    __ATH_TENSOR_APPLYX_UPDATE_COUNTERS(tensor2, 0)
    __ATH_TENSOR_APPLYX_UPDATE_COUNTERS(tensor3, 0)
  }
  if(tensor1_counter != NULL)
    delete [] tensor1_counter;
  if(tensor2_counter != NULL)
    delete [] tensor2_counter;
  if(tensor3_counter != NULL)
    delete [] tensor3_counter;
}

/*
  Apply a pointwise operator to three tensors.

  The calling convention for op is a function/functor that takes takes three references to
  type scalar; at least one of these references should be non-const in order to write the output.
  For example, to compute a = b + c, op would be of the form:
  [](scalar &a_val, const scalar &b_val, const scalar &c_val) { a_val = b_val + c_val; };
*/
template<typename scalar1, typename scalar2, typename scalar3, typename Op>
void CPU_tensor_apply3(Tensor tensor1, Tensor tensor2, Tensor tensor3, Op op) {
  CPU_tensor_apply3_dim<scalar1, scalar2, scalar3, Op>(tensor1, tensor2, tensor3, -1, op);
}

template <typename scalar1, typename scalar2, typename scalar3, typename scalar4, typename Op>
void CPU_tensor_apply4_dim(Tensor &tensor1, Tensor& tensor2, Tensor& tensor3, Tensor& tensor4, int64_t dim, Op op) {
  checkBackend("CPU_tensor_apply4", {tensor1, tensor2, tensor3, tensor4}, Backend::CPU);
  bool TH_TENSOR_APPLY_hasFinished = false;
  int64_t TH_TENSOR_dim_index = 0;
  __ATH_TENSOR_APPLYX_PREAMBLE(scalar1, tensor1, dim, 1)
  __ATH_TENSOR_APPLYX_PREAMBLE(scalar2, tensor2, dim, 1)
  __ATH_TENSOR_APPLYX_PREAMBLE(scalar3, tensor3, dim, 1)
  __ATH_TENSOR_APPLYX_PREAMBLE(scalar4, tensor4, dim, 1)

  int elements_equal = 1;
  auto t1_numel = tensor1.numel();
  auto t2_numel = tensor2.numel();
  auto t3_numel = tensor3.numel();
  auto t4_numel = tensor4.numel();
  if(t1_numel!= t2_numel) {
    elements_equal = 0;
  } else if(t1_numel != t3_numel) {
    elements_equal = 0;
  } else if(t1_numel != t4_numel) {
      elements_equal = 0;
  }
  if (elements_equal == 0) {
    std::ostringstream oss;
    oss << "inconsistent tensor size, expected " << tensor1.sizes() << ", " << tensor2.sizes() << ", "
        << tensor3.sizes() << ", and " << tensor4.sizes() << " to have the same number of elements, but got "
        << t1_numel << ", " << t2_numel << ", " << t3_numel << ", and " << t4_numel << " elements respectively";
    throw std::runtime_error(oss.str());
  }

  while(!TH_TENSOR_APPLY_hasFinished)
  {
    /* Loop through the inner most region of the Tensor */
    for(; tensor1_i <  tensor1_size && tensor2_i < tensor2_size && tensor3_i < tensor3_size && tensor4_i < tensor4_size
        ; tensor1_i++, tensor2_i++, tensor3_i++, tensor4_i++,
          tensor1_data += tensor1_stride, tensor2_data += tensor2_stride, tensor3_data += tensor3_stride, tensor4_data += tensor4_stride)
    {
      op(*tensor1_data, *tensor2_data, *tensor3_data, *tensor4_data);
    }
    __ATH_TENSOR_APPLYX_UPDATE_COUNTERS(tensor1, 0)
    __ATH_TENSOR_APPLYX_UPDATE_COUNTERS(tensor2, 0)
    __ATH_TENSOR_APPLYX_UPDATE_COUNTERS(tensor3, 0)
    __ATH_TENSOR_APPLYX_UPDATE_COUNTERS(tensor4, 0)
  }
  if(tensor1_counter != NULL)
    delete [] tensor1_counter;
  if(tensor2_counter != NULL)
    delete [] tensor2_counter;
  if(tensor3_counter != NULL)
    delete [] tensor3_counter;
  if(tensor4_counter != NULL)
    delete [] tensor4_counter;
}

/*
  Apply a pointwise operator to four tensors.

  The calling convention for op is a function/functor that takes takes four references to
  type scalar; at least one of these references should be non-const in order to write the output.
  For example, to compute a = b + c * d, op would be of the form:
  [](scalar &a_val, const scalar &b_val, const scalar &c_val, const scalar &d_val) {
    a_val = b_val + c_val * d_val;
  };
*/
template<typename scalar1, typename scalar2, typename scalar3, typename scalar4, typename Op>
void CPU_tensor_apply4(Tensor tensor1, Tensor tensor2, Tensor tensor3, Tensor tensor4, Op op) {
  CPU_tensor_apply4_dim<scalar1, scalar2, scalar3, scalar4, Op>(tensor1, tensor2, tensor3, tensor4, -1, op);
}

}
