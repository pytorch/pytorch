/**
 * Performance benchmark for RestrictPtrTraits migration
 *
 * This benchmark compares:
 * - OLD: PackedTensorAccessor with RestrictPtrTraits (broken - __restrict__ ignored)
 * - NEW: Separate __restrict__ pointers with PackedTensorAccessorMetadata (working)
 *
 * Using a kernel pattern similar to embedding_bag_nbits_rowwise_offsets_kernel
 */

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>

#include <iostream>

using namespace at;

template <typename index_t>
__global__ void embedding_bag_kernel_old(
    const PackedTensorAccessor64<uint8_t, 2, RestrictPtrTraits> weight,
    const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
    const PackedTensorAccessor32<float, 1, RestrictPtrTraits> per_sample_weights,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> output,
    const bool include_last_offset) {

  const int32_t B = output.size(0);
  const int32_t D = output.size(1);
  const int32_t D_bytes = weight.size(1);

  const int32_t b = blockIdx.x;
  if (b >= B) return;

  const bool use_per_sample = per_sample_weights.size(0) > 0;

  int64_t indices_start = offsets[b];
  int64_t indices_end;
  if (include_last_offset) {
    indices_end = offsets[b + 1];
  } else {
    indices_end = (b + 1) < offsets.size(0) ? offsets[b + 1] : indices.size(0);
  }

  int32_t L = indices_end - indices_start;

  if (L == 0) {
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      output[b][d] = 0.0f;
    }
    return;
  }

  for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
    float accumulator = 0.0f;

    for (int32_t l = indices_start; l < indices_end; ++l) {
      int64_t idx = indices[static_cast<int32_t>(l)];
      float sample_weight = use_per_sample ? per_sample_weights[static_cast<int32_t>(l)] : 1.0f;

      // Simulate dequantization: read scale/bias from end of row, data from start
      const uint8_t* row_ptr = &weight[idx][0];
      float scale = *reinterpret_cast<const float*>(&row_ptr[D_bytes - 8]);
      float bias = *reinterpret_cast<const float*>(&row_ptr[D_bytes - 4]);

      float val = static_cast<float>(weight[idx][d]) * scale + bias;
      accumulator += val * sample_weight;
    }

    output[b][d] = accumulator;
  }
}

template <typename index_t>
__global__ void embedding_bag_kernel_new(
    const uint8_t* __restrict__ weight_data,
    const PackedTensorAccessorMetadata<2, int64_t> weight_meta,
    const index_t* __restrict__ indices_data,
    const PackedTensorAccessorMetadata<1, int32_t> indices_meta,
    const index_t* __restrict__ offsets_data,
    const PackedTensorAccessorMetadata<1, int32_t> offsets_meta,
    const float* __restrict__ per_sample_weights_data,
    const PackedTensorAccessorMetadata<1, int32_t> per_sample_weights_meta,
    float* __restrict__ output_data,
    const PackedTensorAccessorMetadata<2, int32_t> output_meta,
    const bool include_last_offset) {

  const int32_t B = output_meta.size(0);
  const int32_t D = output_meta.size(1);
  const int64_t D_bytes = weight_meta.size(1);

  const int32_t b = blockIdx.x;
  if (b >= B) return;

  const bool use_per_sample = per_sample_weights_meta.size(0) > 0;

  int64_t indices_start = offsets_data[packed_accessor_offset(offsets_meta, b)];
  int64_t indices_end;
  if (include_last_offset) {
    indices_end = offsets_data[packed_accessor_offset(offsets_meta, b + 1)];
  } else {
    indices_end = (b + 1) < offsets_meta.size(0)
        ? offsets_data[packed_accessor_offset(offsets_meta, b + 1)]
        : indices_meta.size(0);
  }

  int32_t L = indices_end - indices_start;

  if (L == 0) {
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      output_data[packed_accessor_offset(output_meta, b, d)] = 0.0f;
    }
    return;
  }

  for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
    float accumulator = 0.0f;

    for (int32_t l = indices_start; l < indices_end; ++l) {
      int64_t idx = indices_data[packed_accessor_offset(indices_meta, static_cast<int32_t>(l))];
      float sample_weight = use_per_sample
          ? per_sample_weights_data[packed_accessor_offset(per_sample_weights_meta, static_cast<int32_t>(l))]
          : 1.0f;

      // Simulate dequantization: read scale/bias from end of row, data from start
      int64_t row_offset = packed_accessor_offset(weight_meta, idx, static_cast<int64_t>(0));
      const uint8_t* row_ptr = &weight_data[row_offset];
      float scale = *reinterpret_cast<const float*>(&row_ptr[D_bytes - 8]);
      float bias = *reinterpret_cast<const float*>(&row_ptr[D_bytes - 4]);

      float val = static_cast<float>(weight_data[packed_accessor_offset(weight_meta, idx, static_cast<int64_t>(d))]) * scale + bias;
      accumulator += val * sample_weight;
    }

    output_data[packed_accessor_offset(output_meta, b, d)] = accumulator;
  }
}

float benchmark_kernel(
    std::function<void(cudaStream_t)> kernel_launch,
    cudaStream_t stream,
    int warmup_iters = 20,
    int bench_iters = 100) {

  for (int i = 0; i < warmup_iters; ++i) {
    kernel_launch(stream);
  }
  cudaStreamSynchronize(stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, stream);
  for (int i = 0; i < bench_iters; ++i) {
    kernel_launch(stream);
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return milliseconds / bench_iters;
}

TEST(RestrictMigrationBenchmark, EmbeddingBagKernel) {
  if (!at::cuda::is_available()) return;

  // Realistic embedding bag dimensions
  const int64_t num_embeddings = 100000;
  const int64_t embedding_dim = 128;
  const int64_t D_bytes = embedding_dim + 8;
  const int64_t batch_size = 512;
  const int64_t avg_bag_size = 20;

  std::cout << "\n=== Embedding Bag Kernel Benchmark ===" << std::endl;
  std::cout << "num_embeddings=" << num_embeddings
            << ", embedding_dim=" << embedding_dim
            << ", batch_size=" << batch_size
            << ", avg_bag_size=" << avg_bag_size << std::endl;

  auto weight = at::randint(0, 256, {num_embeddings, D_bytes}, at::CUDA(at::kByte));

  auto weight_cpu = weight.cpu();
  auto weight_accessor = weight_cpu.accessor<uint8_t, 2>();
  for (int64_t i = 0; i < num_embeddings; ++i) {
    float scale = 0.1f;
    float bias = 0.0f;
    std::memcpy(&weight_accessor[i][D_bytes - 8], &scale, sizeof(float));
    std::memcpy(&weight_accessor[i][D_bytes - 4], &bias, sizeof(float));
  }
  weight = weight_cpu.cuda();

  const int64_t total_indices = batch_size * avg_bag_size;
  auto indices = at::randint(0, num_embeddings, {total_indices}, at::CUDA(at::kInt));

  std::vector<int32_t> offsets_vec(batch_size + 1);
  for (int64_t i = 0; i <= batch_size; ++i) {
    offsets_vec[i] = static_cast<int32_t>(i * avg_bag_size);
  }
  auto offsets = at::from_blob(offsets_vec.data(), {batch_size + 1}, at::kInt).clone().cuda();

  auto per_sample_weights = at::rand({total_indices}, at::CUDA(at::kFloat));

  auto output_old = at::empty({batch_size, embedding_dim}, at::CUDA(at::kFloat));
  auto output_new = at::empty({batch_size, embedding_dim}, at::CUDA(at::kFloat));

  auto stream = at::cuda::getCurrentCUDAStream();

  const int block_size = 128;

  // Create accessors for OLD kernel
  auto weight_acc = weight.packed_accessor64<uint8_t, 2, RestrictPtrTraits>();
  auto indices_acc = indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>();
  auto offsets_acc = offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>();
  auto per_sample_weights_acc = per_sample_weights.packed_accessor32<float, 1, RestrictPtrTraits>();
  auto output_old_acc = output_old.packed_accessor32<float, 2, RestrictPtrTraits>();

  // Create metadata for NEW kernel
  auto weight_meta = make_packed_accessor_metadata(weight_acc);
  auto indices_meta = make_packed_accessor_metadata(indices_acc);
  auto offsets_meta = make_packed_accessor_metadata(offsets_acc);
  auto per_sample_weights_meta = make_packed_accessor_metadata(per_sample_weights_acc);
  auto output_new_meta = make_packed_accessor_metadata(
      output_new.packed_accessor32<float, 2, RestrictPtrTraits>());

  float time_old = benchmark_kernel([&](cudaStream_t s) {
    embedding_bag_kernel_old<int32_t><<<batch_size, block_size, 0, s>>>(
        weight_acc, indices_acc, offsets_acc, per_sample_weights_acc,
        output_old_acc, true);
  }, stream);

  float time_new = benchmark_kernel([&](cudaStream_t s) {
    embedding_bag_kernel_new<int32_t><<<batch_size, block_size, 0, s>>>(
        weight.data_ptr<uint8_t>(), weight_meta,
        indices.data_ptr<int32_t>(), indices_meta,
        offsets.data_ptr<int32_t>(), offsets_meta,
        per_sample_weights.data_ptr<float>(), per_sample_weights_meta,
        output_new.data_ptr<float>(), output_new_meta,
        true);
  }, stream);

  ASSERT_TRUE(output_old.allclose(output_new, 1e-3, 1e-3))
      << "Output mismatch between old and new kernels";

  std::cout << "\nResults:" << std::endl;
  std::cout << "  OLD (RestrictPtrTraits in accessor): " << time_old << " ms" << std::endl;
  std::cout << "  NEW (metadata + __restrict__ ptr):   " << time_new << " ms" << std::endl;

  float speedup = time_old / time_new;
  std::cout << "\nSpeedup: " << speedup << "x" << std::endl;

  if (speedup > 1.05) {
    std::cout << "*** NEW approach is faster - __restrict__ optimization is working! ***" << std::endl;
  } else if (speedup < 0.95) {
    std::cout << "(NEW approach is slower - unexpected)" << std::endl;
  } else {
    std::cout << "(Similar performance - __restrict__ may not help this memory access pattern)" << std::endl;
  }
}

// Additional test with varying bag sizes to stress the memory access pattern
TEST(RestrictMigrationBenchmark, EmbeddingBagVaryingBagSize) {
  if (!at::cuda::is_available()) return;

  const int64_t num_embeddings = 50000;
  const int64_t embedding_dim = 256;
  const int64_t D_bytes = embedding_dim + 8;
  const int64_t batch_size = 256;

  std::cout << "\n=== Embedding Bag with Varying Bag Sizes ===" << std::endl;

  auto weight = at::randint(0, 256, {num_embeddings, D_bytes}, at::CUDA(at::kByte));

  auto weight_cpu = weight.cpu();
  auto weight_accessor = weight_cpu.accessor<uint8_t, 2>();
  for (int64_t i = 0; i < num_embeddings; ++i) {
    float scale = 0.1f;
    float bias = 0.0f;
    std::memcpy(&weight_accessor[i][D_bytes - 8], &scale, sizeof(float));
    std::memcpy(&weight_accessor[i][D_bytes - 4], &bias, sizeof(float));
  }
  weight = weight_cpu.cuda();

  std::vector<int32_t> offsets_vec;
  offsets_vec.push_back(0);
  int32_t current_offset = 0;
  for (int64_t i = 0; i < batch_size; ++i) {
    int32_t bag_size = 5 + (i % 46);
    current_offset += bag_size;
    offsets_vec.push_back(current_offset);
  }
  const int64_t total_indices = current_offset;

  auto indices = at::randint(0, num_embeddings, {total_indices}, at::CUDA(at::kInt));
  auto offsets = at::from_blob(offsets_vec.data(), {static_cast<int64_t>(offsets_vec.size())}, at::kInt).clone().cuda();
  auto per_sample_weights = at::rand({total_indices}, at::CUDA(at::kFloat));

  auto output_old = at::empty({batch_size, embedding_dim}, at::CUDA(at::kFloat));
  auto output_new = at::empty({batch_size, embedding_dim}, at::CUDA(at::kFloat));

  auto stream = at::cuda::getCurrentCUDAStream();
  const int block_size = 256;

  // Create accessors and metadata
  auto weight_acc = weight.packed_accessor64<uint8_t, 2, RestrictPtrTraits>();
  auto indices_acc = indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>();
  auto offsets_acc = offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>();
  auto per_sample_weights_acc = per_sample_weights.packed_accessor32<float, 1, RestrictPtrTraits>();
  auto output_old_acc = output_old.packed_accessor32<float, 2, RestrictPtrTraits>();

  auto weight_meta = make_packed_accessor_metadata(weight_acc);
  auto indices_meta = make_packed_accessor_metadata(indices_acc);
  auto offsets_meta = make_packed_accessor_metadata(offsets_acc);
  auto per_sample_weights_meta = make_packed_accessor_metadata(per_sample_weights_acc);
  auto output_new_meta = make_packed_accessor_metadata(
      output_new.packed_accessor32<float, 2, RestrictPtrTraits>());

  float time_old = benchmark_kernel([&](cudaStream_t s) {
    embedding_bag_kernel_old<int32_t><<<batch_size, block_size, 0, s>>>(
        weight_acc, indices_acc, offsets_acc, per_sample_weights_acc,
        output_old_acc, true);
  }, stream);

  float time_new = benchmark_kernel([&](cudaStream_t s) {
    embedding_bag_kernel_new<int32_t><<<batch_size, block_size, 0, s>>>(
        weight.data_ptr<uint8_t>(), weight_meta,
        indices.data_ptr<int32_t>(), indices_meta,
        offsets.data_ptr<int32_t>(), offsets_meta,
        per_sample_weights.data_ptr<float>(), per_sample_weights_meta,
        output_new.data_ptr<float>(), output_new_meta,
        true);
  }, stream);

  ASSERT_TRUE(output_old.allclose(output_new, 1e-3, 1e-3))
      << "Output mismatch between old and new kernels";

  std::cout << "  OLD: " << time_old << " ms" << std::endl;
  std::cout << "  NEW: " << time_new << " ms" << std::endl;
  std::cout << "  Speedup: " << (time_old / time_new) << "x" << std::endl;
}
