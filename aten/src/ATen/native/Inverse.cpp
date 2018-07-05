#include "TH.h"  // for USE_LAPACK
#include "ATen/native/LinearAlgebraUtils.h"

#ifdef USE_LAPACK
extern "C" void dgetrf_(
    int* n, int* m, double* a, int* lda,
    int *ipiv, int* info);
extern "C" void sgetrf_(
    int* n, int* m, float* a, int* lda,
    int *ipiv, int* info);
extern "C" void dgetri_(
    int* n, double* a, int* lda,
    int *ipiv, double* b, int* ldb, int* info);
extern "C" void sgetri_(
    int* n, float* a, int* lda,
    int* ipiv, float* b, int* ldb, int* info);
#endif

namespace at { namespace native {

template<class scalar_t>
void lapackGetri(
    int n, scalar_t* a, int lda,
    int* ipiv, scalar_t* b, int ldb, int* info) {
  AT_ERROR("getri only takes float or double Tensors");
}

template<class scalar_t>
void lapackGetrf(
    int n, int m, scalar_t* a, int lda,
    int* ipiv, int* info) {
  AT_ERROR("getrf only takes float or double Tensors");
}

#ifdef USE_LAPACK
template<> void lapackGetri<float>(
    int n, float* a, int lda,
    int* ipiv, float* b, int ldb, int* info) {
  sgetri_(&n, a, &lda, ipiv, b, &ldb, info);
}

template<> void lapackGetri<double>(
    int n, double* a, int lda,
    int* ipiv, double* b, int ldb, int* info) {
  dgetri_(&n, a, &lda, ipiv, b, &ldb, info);
}

template<> void lapackGetrf<float>(
    int n, int m, float* a, int lda,
    int* ipiv, int* info) {
  sgetrf_(&n, &m, a, &lda, ipiv, info);
}

template<> void lapackGetrf<double>(
    int n, int m, double* a, int lda,
    int* ipiv, int* info) {
  dgetrf_(&n, &m, a, &lda, ipiv, info);
}
#endif

// Tensor inverse
Tensor inverse_cpu(const Tensor& self) {
    if(self.dim() < 2)
    {
        AT_ERROR("Inverse: expected a tensor with 2 or more dimensions, got ", self.dim());
    }
    int n = self.size(-1);
    int m = self.size(-2);
    if(n != m)
    {
        AT_ERROR("Inverse: expected square matrices.");
    }
    auto input_3d = self.view({-1, m, n}).contiguous();
    auto input_colmajor = cloneBatchedColumnMajor(input_3d);

    int num_batches = input_3d.size(0);
    int *pivots = new int[n];
    int info;
    auto work = at::empty({m, n}, self.type());

    AT_DISPATCH_FLOATING_TYPES(self.type(), "getri", [&] {
        scalar_t *matStart = input_colmajor.data<scalar_t>();
        scalar_t *work_ptr = work.data<scalar_t>();
        for(int i=0; i<num_batches; ++i)
        {
            // LU
            lapackGetrf(n, m, matStart, n, pivots, &info);
            if(info != 0) {
                break;
            }

            // getri
            lapackGetri(n, matStart, n, pivots, work_ptr, n*n, &info);
            if(info != 0) {
                break;
            }

            // Update data pointer
            matStart += m*n;
        }
    });

    delete pivots;

    if(info != 0)
    {
        AT_ERROR("Unable to invert some of the matrices.");
    }

    return input_colmajor.view(self.sizes());
}

Tensor &inverse_out_cpu(Tensor& result, const Tensor& self) {
    result = inverse_cpu(self);
    return result;
}

}} // namespace at::native
