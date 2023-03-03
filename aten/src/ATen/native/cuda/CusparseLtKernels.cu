/*
The following source file implements a sparse linear operator using cusparseLt
*/

#include <ATen/Functions.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>
#include <torch/custom_class.h>
#include <iostream>

#include <iostream>

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_sparse.h>
#include <cutlass/util/host_reorder.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/host_uncompress.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_copy.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/tensor_view_io.h>

#include <functional>
#include <limits>
#include <typeinfo>

#define CUTLASS_STATUS_CHECK(status)                                      \
  {                                                                       \
    TORCH_CHECK(status == cutlass::Status::kSuccess,                      \
                "Got CUTLASS error: ", cutlassGetStatusString(status));   \
  }

namespace at {
namespace native {

uint16_t _mask_to_meta(bool pos0, bool pos1, bool pos2, bool pos3) {
  auto pos_tuple = std::make_tuple(pos0, pos1, pos2, pos3);
  // NOTE:
  // See
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-sparse-matrix-storage
  // There are only 6 valid configurations (4 choose 2) and for each there is a
  // special number.
  if (pos_tuple == std::make_tuple(1, 1, 0, 0)) {
    return 4; // 0100
  }
  if (pos_tuple == std::make_tuple(1, 0, 1, 0)) {
    return 8; // 1000
  }
  if (pos_tuple == std::make_tuple(0, 1, 1, 0)) {
    return 9; // 1001
  }
  if (pos_tuple == std::make_tuple(1, 0, 0, 1)) {
    return 12; // 1100
  }
  if (pos_tuple == std::make_tuple(0, 1, 0, 1)) {
    return 13; // 1101
  }
  if (pos_tuple == std::make_tuple(0, 0, 1, 1)) {
    return 14; // 1110
  }
  TORCH_CHECK(
      false,
      "Unsupported mask configuration: ",
      pos0,
      pos1,
      pos2,
      pos3,
      ". Please follow 2 by 4 pattern.");
  return 0;
}

class TwoFourSparseGemm {
  public:
    using MetaReorderedPtr = std::unique_ptr<void, void(*)(void*)>;

    TwoFourSparseGemm(const Tensor& sparse, const Tensor& dense, const Tensor& mask) :
      length_m_(sparse.size(0)),
      length_n_(dense.size(1)),
      length_k_(dense.size(0)),
      meta_reordered_(nullptr, [](void*) {})
    {
      // The code section below describes datatype for input, output
      // matrices and computation between elements in input matrices,
      // which will all be used as
     // template parameters for cutlass::gemm::device::SparseGemm
      using ElementInputA =
          cutlass::half_t; // <- data type of elements in input matrix A
      using ElementInputB =
          cutlass::half_t; // <- data type of elements in input matrix B
      using ElementOutput =
          cutlass::half_t; // <- data type of elements in output matrix D

      // The code section below describes matrix layout of input and
      // output matrices.  Row Major for Matrix A, Column Major for
      // Matrix B and Row Major for Matrix C
      using LayoutInputA = cutlass::layout::RowMajor;
      using LayoutInputB = cutlass::layout::RowMajor;
      using LayoutOutput = cutlass::layout::RowMajor;

      using Gemm = cutlass::gemm::device::SparseGemm<
          ElementInputA,
          LayoutInputA,
          ElementInputB,
          LayoutInputB,
          ElementOutput,
          LayoutOutput,
          float,
          cutlass::arch::OpClassTensorOp,
          cutlass::arch::Sm80,
          cutlass::gemm::GemmShape<64, 128, 64>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          cutlass::gemm::GemmShape<16, 8, 32>,
          cutlass::epilogue::thread::LinearCombination<
                    cutlass::half_t,
                    128 / cutlass::sizeof_bits<ElementOutput>::value,
                    float,
                    float>,
          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
          6>;

      // Data type and layout of meta data matrix E can be inferred
      // from template Gemm.
      using ElementInputE = typename Gemm::ElementE;
      using LayoutInputE = cutlass::layout::RowMajor;
      using ReorderedLayoutInputE = typename Gemm::LayoutE;

      using HostTensorInputE = typename cutlass::HostTensor<ElementInputE, ReorderedLayoutInputE>;

      TORCH_CHECK(
        length_k_ % 16 == 0, "Expected size(1) of mask to be divisible by 16.");
      TORCH_CHECK(mask.is_contiguous(), "Expected mask to be contiguous.");
      TORCH_CHECK(mask.dtype() == at::kBool, "Expected mask to be of dtype bool.");

      HostTensorInputE meta(cutlass::make_Coord(length_m_, length_k_ / 16));
      auto mask_cpu = mask.cpu();
      const bool* mask_ptr = mask_cpu.data_ptr<bool>();
      for (int64_t i = 0; i < length_m_; i++) {
        for (int64_t j = 0; j < length_k_; j += 16) {
          uint16_t meta_val = 0;
          for (int64_t k = 0; k < 4; k++) {
            bool pos0 = mask_ptr[i * length_k_ + j + k * 4];
            bool pos1 = mask_ptr[i * length_k_ + j + k * 4 + 1];
            bool pos2 = mask_ptr[i * length_k_ + j + k * 4 + 2];
            bool pos3 = mask_ptr[i * length_k_ + j + k * 4 + 3];
            uint16_t val = _mask_to_meta(pos0, pos1, pos2, pos3);
            meta_val = (meta_val | (val << (4 * k)));
          }
          // PyTorch doesn't have a uint16_t dtype, so we're using the
          // signed equivalent.  However, we don't want to actually
          // convert or overflow. We just want to store the bits as is
          // and then retrieve them again later on.
          int16_t meta_storage;
          std::memcpy(&meta_storage, &meta_val, sizeof(meta_storage));
          meta.at({i, j / 16}) = meta_storage;
        }
      }

      auto meta_reordered = new HostTensorInputE(cutlass::make_Coord(length_m_, length_k_ / 16));
      cutlass::reorder_meta(meta_reordered->host_ref(), meta.host_ref(),
                            {length_m_,
                             0, // currently unused by cutlass::reorder_meta()
                             length_k_ / 16});
      meta_reordered->sync_device();

      meta_reordered_ =
          MetaReorderedPtr(meta_reordered,
                           [](void* p) -> void {
                               delete reinterpret_cast<HostTensorInputE*>(p);
                           });

      constexpr int kSparse = Gemm::kSparse;
      TORCH_CHECK(
        dense.size(0) % kSparse == 0,
        "Expected dense.size(0) of value ",
        dense.size(0),
        " to be evenly divisible by ",
        kSparse,
        " but got.");
      TORCH_CHECK(
        sparse.size(1) * kSparse == dense.size(0),
        "Expected sparse.size(1) of value ",
        sparse.size(1),
        " to match dense.size(0) of value ",
        dense.size(0),
        " to match after being multiplied by ",
        kSparse);

      op_ = [](
          const Tensor& sparse,
          const Tensor& dense,
          const int length_m,
          const int length_n,
          const int length_k,
          void* meta_reordered) -> Tensor {
        auto result = sparse.new_empty({sparse.size(0), dense.size(1)});

        auto tensor_a = sparse;
        auto tensor_b = dense;
        auto tensor_c = result;
        auto tensor_d = result;

        TORCH_CHECK(tensor_a.size(0) == length_m);
        TORCH_CHECK(tensor_a.size(1) == length_k / kSparse);
        TORCH_CHECK(tensor_b.size(0) == length_k);
        TORCH_CHECK(tensor_b.size(1) == length_n);

        TORCH_CHECK(
                    tensor_a.device() == tensor_b.device(),
                    "Check 0: Expected all Tensors to live on the GPU.");
        TORCH_CHECK(
                    tensor_b.device() == tensor_c.device(),
                    "Check 1: Expected all Tensors to live on the GPU.");
        TORCH_CHECK(
                    tensor_c.device() == tensor_d.device(),
                    "Check 2: Expected all Tensors to live on the GPU.");

        // Create a tuple of problem size for matrix multiplication
        cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

        LayoutInputA layout_a(tensor_a.stride(0));
        LayoutInputB layout_b(tensor_b.stride(0));
        LayoutOutput layout_c(tensor_c.stride(0));
        LayoutOutput layout_d(tensor_d.stride(0));
        auto tensor_a_device_ref = cutlass::TensorRef<cutlass::half_t, LayoutInputA>((cutlass::half_t*)tensor_a.data_ptr<at::Half>(), layout_a);
        auto tensor_b_device_ref = cutlass::TensorRef<cutlass::half_t, LayoutInputB>((cutlass::half_t*)tensor_b.data_ptr<at::Half>(), layout_b);
        auto tensor_c_device_ref = cutlass::TensorRef<cutlass::half_t, LayoutOutput>((cutlass::half_t*)tensor_c.data_ptr<at::Half>(), layout_c);
        auto tensor_d_device_ref = cutlass::TensorRef<cutlass::half_t, LayoutOutput>((cutlass::half_t*)tensor_d.data_ptr<at::Half>(), layout_d);
        auto tensor_e_device_ref = reinterpret_cast<HostTensorInputE*>(meta_reordered)->device_ref();

        // Initialize alpha and beta for dot product computation
        float alpha = 1;
        float beta = 0;

        // Split K dimension into 1 partitions
        int split_k_slices = 1;

        // Create a tuple of gemm kernel arguments. This is later
        // passed as arguments to launch instantiated CUTLASS kernel
        typename Gemm::Arguments arguments{
          problem_size, // <- problem size of matrix multiplication
          tensor_a_device_ref, // <- reference to matrix A on device
          tensor_b_device_ref, // <- reference to matrix B on device
          tensor_c_device_ref, // <- reference to matrix C on device
          tensor_d_device_ref, // <- reference to matrix D on device
          tensor_e_device_ref, // <- reference to matrix E on device
          {alpha, beta}, // <- tuple of alpha and beta
          split_k_slices}; // <- k-dimension split factor

        Gemm gemm_op;

        cutlass::Status status = gemm_op.can_implement(arguments);
        CUTLASS_STATUS_CHECK(status);

        // FIXME: check is CUTLASS workspace allocation needed here!

        status = gemm_op.initialize(arguments, nullptr, at::cuda::getCurrentCUDAStream());
        CUTLASS_STATUS_CHECK(status);

        status = gemm_op.run(at::cuda::getCurrentCUDAStream());
        CUTLASS_STATUS_CHECK(status);

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return result;
      };
    }

    TwoFourSparseGemm(const TwoFourSparseGemm&) = delete;
    TwoFourSparseGemm(TwoFourSparseGemm&&) = delete;

    TwoFourSparseGemm& operator=(const TwoFourSparseGemm&) = delete;
    TwoFourSparseGemm& operator=(TwoFourSparseGemm&&) = delete;

    Tensor operator()(const Tensor& sparse, const Tensor& dense) const {
      return op_(sparse, dense, length_m_, length_n_, length_k_, meta_reordered_.get());
    }

  private:
    int length_m_;
    int length_n_;
    int length_k_;
    MetaReorderedPtr meta_reordered_;
    std::function<Tensor(const Tensor&, const Tensor&, const int, const int, const int, void*)> op_;
};

int64_t _cusparselt_create_sparse_gemm(const Tensor& sparse, const Tensor& dense, const Tensor& mask) {
  auto handle = new TwoFourSparseGemm(sparse, dense, mask);
  static_assert(sizeof(void*) <= sizeof(int64_t));
  return reinterpret_cast<int64_t>(handle);
}

void _cusparselt_destroy_sparse_gemm(const int64_t handle) {
  static_assert(sizeof(void*) <= sizeof(int64_t));
  delete reinterpret_cast<TwoFourSparseGemm*>(handle);
}

// TODO: Pull back in device and cuda version constraints.
Tensor _cusparselt_linear(
    const Tensor& sparse,
    const Tensor& dense,
    const int64_t handle) {
  static_assert(sizeof(void*) <= sizeof(int64_t));
  return reinterpret_cast<TwoFourSparseGemm*>(handle)->operator()(sparse, dense);
}

} // namespace native
} // namespace at
