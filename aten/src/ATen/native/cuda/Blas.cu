#include <ATen/Dispatch.h>
#include <ATen/native/Blas.h>

namespace at { namespace native {
namespace {

void addmv_impl_cuda(Tensor& result, const Tensor &self, const Tensor &mat, const Tensor &vec, Scalar beta, Scalar alpha) {
//   auto r_stride = THTensor_strideLegacyNoScalars(r_, 0);

//   if(mat->stride(0) == 1)
//   {
// #ifdef THC_REAL_IS_FLOAT
//     THCudaBlas_Sgemv(state, 'n', mat->size(0), mat->size(1),
//                     alpha, THCTensor_(data)(state, mat), mat->stride(1),
//                     THCTensor_(data)(state, vec), vec_stride,
//                     beta, THCTensor_(data)(state, r_), r_stride);
// #elif defined(THC_REAL_IS_DOUBLE)
//     THCudaBlas_Dgemv(state, 'n', mat->size(0), mat->size(1),
//                     alpha, THCTensor_(data)(state, mat), mat->stride(1),
//                     THCTensor_(data)(state, vec), vec_stride,
//                     beta, THCTensor_(data)(state, r_), r_stride);
// #endif
//   }
//   else if(mat->stride(1) == 1)
//   {
// #ifdef THC_REAL_IS_FLOAT
//     THCudaBlas_Sgemv(state, 't',  mat->size(1), mat->size(0),
//                     alpha, THCTensor_(data)(state, mat), mat->stride(0),
//                     THCTensor_(data)(state, vec), vec_stride,
//                     beta, THCTensor_(data)(state, r_), r_stride);
// #elif defined(THC_REAL_IS_DOUBLE)
//     THCudaBlas_Dgemv(state, 't',  mat->size(1), mat->size(0),
//                      alpha, THCTensor_(data)(state, mat), mat->stride(0),
//                      THCTensor_(data)(state, vec), vec_stride,
//                      beta, THCTensor_(data)(state, r_), r_stride);
// #endif
//   }
//   else
//   {
//     THCTensor *cmat = THCTensor_(newContiguous)(state, mat);

// #ifdef THC_REAL_IS_FLOAT
//     THCudaBlas_Sgemv(state, 't',  mat->size(1), mat->size(0),
//                     alpha, THCTensor_(data)(state, cmat), cmat->stride(0),
//                     THCTensor_(data)(state, vec), vec_stride,
//                     beta, THCTensor_(data)(state, r_), r_stride);
// #elif defined(THC_REAL_IS_DOUBLE)
//     THCudaBlas_Dgemv(state, 't',  mat->size(1), mat->size(0),
//                     alpha, THCTensor_(data)(state, cmat), cmat->stride(0),
//                     THCTensor_(data)(state, vec), vec_stride,
//                     beta, THCTensor_(data)(state, r_), r_stride);
// #endif

//     THCTensor_(free)(state, cmat);
//   }

//   // In cublasSgemv, cublasDgemv (x,0).mv(0) does not
//   // handle beta, whereas cublasSgemm, cublasDgemm do for case where (x,0).mm(0,y).
//   if (THTensor_sizeLegacyNoScalars(vec, 0) == 0 && mat->size(0) != 0) {
//     if(THCNumerics<scalar_t>::eq(beta, ScalarConvert<int, scalar_t>::to(0))) {
//       THCTensor_(zero)(state, r_);
//     } else if(THCNumerics<scalar_t>::ne(beta, ScalarConvert<int, scalar_t>::to(1))) {
//       THCTensor_(mul)(state, r_, r_, beta);
//     }
//   }

// #elif defined(THC_REAL_IS_HALF) || defined(THC_REAL_IS_BFLOAT16)
//     // Currently no Hgemv/SgemvEx in Cublas
//     THCTensor *vecAsMatrix = THCTensor_(newWithTensor)(state, vec);
//     THCTensor_(resize2d)(state, vecAsMatrix, vec_size, 1);

//     THCTensor *tAsMatrix = THCTensor_(newWithTensor)(state, t);
//     THCTensor_(resize2d)(state, tAsMatrix, THTensor_sizeLegacyNoScalars(tAsMatrix, 0), 1);

//     THCTensor_(addmm)(state, r_, tAsMatrix, mat, vecAsMatrix, beta, alpha);

//     // r_ will have answer as matrix, need to return a vector
//     THCTensor_(resize1d)(state, r_, THTensor_sizeLegacyNoScalars(r_, 0));
//     THCTensor_(free)(state, vecAsMatrix);
//     THCTensor_(free)(state, tAsMatrix);
// #endif
// #else
//   ERROR_ONLY_FP_TYPES("addmv");
// #endif
  return;
}

} // anonymous namespace

REGISTER_DISPATCH(addmv_stub, &addmv_impl_cuda);

}} // namespace at::native