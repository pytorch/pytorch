#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


using namespace at;

Tensor gather_with_broadcast_cuda(IntArrayRef outsizes, const Tensor &src, int64_t dim, const Tensor &index) {
  Tensor result = at::empty(outsizes, src.options());
  auto iter = TensorIterator::dim_apply_op(result, index, src, dim);
  int64_t size = outsizes[dim];
  at::native::gpu_apply_dim_kernel(iter,
    [] GPU_LAMBDA (float *result_data, int64_t result_stride, int64_t *index_data, int64_t index_stride, float *src_data, int64_t src_stride) {
      for (int64_t i = 0; i < 100; i++) {
        int64_t index = *(index_data + i * index_stride);
        *(result_data + i * result_stride) = *(src_data + index * src_stride);
      }
    });
  return result;
}

// Test TensorIterator's dim_apply GPU implementation by manually implementing gather
TEST(TensorIteratorTest, DimApply) {
  Tensor src = at::randn({20, 1, 20, 10}, kCUDA);
  Tensor index = at::randint(20, {100, 10, 20, 1}, src.options().dtype(ScalarType::Long));
  Tensor result1 = src.expand({20, 10, 20, 10}).gather(0, index.expand({100, 10, 20, 10}));
  Tensor result2 = gather_with_broadcast_cuda(result1.sizes(), src, 0, index);
  EXPECT_TRUE(at::allclose(result1, result2));
}
