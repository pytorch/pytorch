#include "ops.h"
#include "dispatch.h"
#include "dtype_utils.h"
#include "../generated/shaders.h"
#include "../backend/Allocator.h"

#include <torch/library.h>

namespace torch_vulkan { namespace ops {

// Max elements safe for a 1D cast shader dispatch (65535 wg * 256 threads)
static constexpr int64_t kMaxCastElements = 65535LL * 256LL;  // ~16.8M

// Detect if a 2D tensor is a zero-copy transposed view (column-major layout).
// Returns true when stride[0]=1 (inner stride), stride[1]=size[0] (outer stride).
// This pattern arises from vulkan_t() zero-copy transpose of a row-major [M, N] tensor.
// After transposition: shape=[N,M], strides=[1, N] — the underlying data is [N, M] row-major,
// but addressed as [M, N] column-major. We can feed this to mm_ex with transpose_a=true.
static bool is_t_transposed(const at::Tensor& t) {
    if (t.dim() != 2 || t.is_contiguous()) return false;
    return t.stride(0) == 1 && t.stride(1) == t.size(0);
}

// Get the physical contiguous storage behind a (possibly transposed) 2D tensor.
// If the tensor is a zero-copy transposed view from vulkan_t(), recover the original
// contiguous storage by swapping sizes+strides back. This avoids a strided_copy dispatch.
static at::Tensor get_physical_storage(const at::Tensor& t, bool is_transposed) {
    if (!is_transposed) return t.contiguous();
    // t is a transposed view: shape=[N,M], strides=[1,N]
    // Physical storage is [M,N] row-major: shape=[M,N], strides=[N,1]
    // Reconstruct it as a metadata view (no data movement).
    int64_t N = t.size(0), M = t.size(1);
    auto impl = c10::make_intrusive<at::TensorImpl>(
        c10::Storage(t.storage()),
        t.key_set(),
        t.dtype());
    std::vector<int64_t> phys_size = {M, N}, phys_strides = {N, 1};
    impl->set_sizes_and_strides(phys_size, phys_strides);
    impl->set_storage_offset(t.storage_offset());
    return at::Tensor(std::move(impl));
}

// Detect if an N-D tensor's last two dims are transposed (zero-copy view from vulkan_transpose).
// Returns true when stride[-2]==1 and stride[-1]==size[-2], and outer dims are normally strided.
// This pattern means: physical data is [... , size[-1], size[-2]] but addressed as [... , size[-2], size[-1]].
static bool is_last2_transposed(const at::Tensor& t) {
    int64_t ndim = t.dim();
    if (ndim < 2 || t.is_contiguous()) return false;
    if (t.stride(ndim - 2) != 1) return false;
    if (t.stride(ndim - 1) != t.size(ndim - 2)) return false;
    // Check outer dims: stride[i] must equal product of inner physical sizes
    // Physical inner dims are [size[-1], size[-2]], so inner product = size[-1]*size[-2]
    int64_t expected = t.size(ndim - 1) * t.size(ndim - 2);
    for (int64_t i = ndim - 3; i >= 0; i--) {
        if (t.stride(i) != expected) return false;
        expected *= t.size(i);
    }
    return true;
}

// Recover the physical (un-transposed) storage for a last-2-dims-swapped zero-copy view.
// Returns a contiguous-layout view of the physical storage (no data copy).
static at::Tensor get_physical_storage_nd(const at::Tensor& t) {
    int64_t ndim = t.dim();
    auto sizes = t.sizes().vec();
    auto strides = t.strides().vec();
    // Swap last two sizes and strides back to physical (contiguous) layout
    std::swap(sizes[ndim - 2], sizes[ndim - 1]);
    std::swap(strides[ndim - 2], strides[ndim - 1]);
    auto impl = c10::make_intrusive<at::TensorImpl>(
        c10::Storage(t.storage()),
        t.key_set(),
        t.dtype());
    impl->set_sizes_and_strides(sizes, strides);
    impl->set_storage_offset(t.storage_offset());
    return at::Tensor(std::move(impl));
}

// Extract rows [start, end) from a 2D Vulkan tensor via CPU staging.
// Required because vulkan_contiguous returns self even for slice views.
static at::Tensor extract_rows(const at::Tensor& t, int64_t start, int64_t end) {
    int64_t rows = end - start;
    int64_t cols = t.size(1);
    int64_t bytes_per_row = cols * t.element_size();
    flush_stream();  // Ensure pending GPU writes are visible
    auto& alloc = VulkanAllocator::instance();
    auto* src_buf = alloc.get_buffer(t.data_ptr());
    TORCH_CHECK(src_buf, "extract_rows: cannot get Vulkan buffer");
    std::vector<uint8_t> staging(rows * bytes_per_row);
    src_buf->read(staging.data(),
                  static_cast<VkDeviceSize>(rows * bytes_per_row),
                  static_cast<VkDeviceSize>(start * bytes_per_row));
    auto chunk = at::empty({rows, cols}, t.options());
    auto* dst_buf = alloc.get_buffer(chunk.data_ptr());
    TORCH_CHECK(dst_buf, "extract_rows: cannot get Vulkan buffer for chunk");
    dst_buf->write(staging.data(), rows * bytes_per_row);
    return chunk;
}

// Dispatch a single 2D mm tile: out += A_chunk @ B_chunk, accumulating into out.
// A_chunk: [M, chunk_k], B_chunk: [chunk_k, N] — both contiguous f32 Vulkan tensors.
static void mm_tiled_dispatch(const at::Tensor& A, const at::Tensor& B,
                               at::Tensor& output, bool transpose_a, bool transpose_b) {
    int64_t M = transpose_a ? A.size(1) : A.size(0);
    int64_t K = transpose_a ? A.size(0) : A.size(1);
    int64_t N = transpose_b ? B.size(0) : B.size(1);
    struct { uint32_t M, N, K, transpose_a, transpose_b; } params{
        static_cast<uint32_t>(M), static_cast<uint32_t>(N), static_cast<uint32_t>(K),
        transpose_a ? 1u : 0u, transpose_b ? 1u : 0u
    };
    uint32_t wg_x = (static_cast<uint32_t>(M) + 15) / 16;
    uint32_t wg_y = (static_cast<uint32_t>(N) + 15) / 16;
    dispatch_shader("matmul_mm_tiled_fwd",
                    shaders::matmul_mm_tiled_fwd, shaders::matmul_mm_tiled_fwd_size,
                    {A, B, output}, wg_x, wg_y, 1, &params, sizeof(params));
}


at::Tensor vulkan_mm(const at::Tensor& self, const at::Tensor& mat2) {
    TORCH_CHECK(self.dim() == 2 && mat2.dim() == 2,
                "mm: expected 2D tensors, got ", self.dim(), "D and ", mat2.dim(), "D");
    TORCH_CHECK(self.size(1) == mat2.size(0),
                "mm: mat1 and mat2 shapes cannot be multiplied (",
                self.size(0), "x", self.size(1), " and ",
                mat2.size(0), "x", mat2.size(1), ")");

    // Fast path: detect zero-copy transposed views from vulkan_t() and use mm_ex
    // transpose flags to avoid a strided-copy dispatch.
    // mm_ex expects PHYSICAL (un-transposed) matrices with transpose flags set.
    // If self is a transposed view: self_view=[K,M] but physical=[M,K] → ta=true
    // If mat2 is a transposed view: mat2_view=[N,K] but physical=[K,N] → tb=true
    bool ta = is_t_transposed(self);
    bool tb = is_t_transposed(mat2);
    if (ta || tb) {
        // Recover physical (un-transposed) storage for the flagged inputs
        auto self_use = ta ? get_physical_storage(self, true) : self;
        auto mat2_use = tb ? get_physical_storage(mat2, true) : mat2;
        return vulkan_mm_ex(self_use, mat2_use, ta, tb);
    }

    auto self_c = self.contiguous();
    auto mat2_c = mat2.contiguous();

    check_supported_float(self_c, "mm");
    auto orig_dtype = self_c.scalar_type();

    // Logical dimensions: self_c is [M, K], mat2_c is [K, N]
    int64_t M = self.size(0);
    int64_t K = self.size(1);
    int64_t N = mat2.size(1);

    if (M == 0 || N == 0 || K == 0) {
        auto empty_out = at::zeros({M, N}, self_c.options().dtype(at::kFloat));
        return cast_from_float32(empty_out, orig_dtype);
    }

    // Check if either matrix is too large for ensure_float32.
    // Critical case: mat2_c [K, N] where K*N > kMaxCastElements → OOM.
    // Strategy: chunk along K. C[M,N] = sum_k(A[:,k:k+cs] @ B[k:k+cs,:])
    // Skip chunking for f32 inputs — no cast needed, matmul workgroups are within Vulkan limits.
    const bool already_f32 = (orig_dtype == at::kFloat);
    const bool mat2_too_large = !already_f32 && (K * N > kMaxCastElements);
    const bool self_too_large = !already_f32 && (M * K > kMaxCastElements);

    if (mat2_too_large || self_too_large) {
        // Chunked path: needs zero-initialized accumulator
        auto output_f32 = at::zeros({M, N}, self_c.options().dtype(at::kFloat));
        // Chunk size: max rows of B (or cols of A) that keep B chunk within kMaxCastElements
        int64_t max_cols = std::max(N, int64_t(1));
        int64_t k_chunk = std::max(int64_t(16),
            (kMaxCastElements / max_cols) & ~int64_t(15));

        // For self_c [M, K]: upcast once if safe, otherwise upcast inside loop
        // For the LM head backward case: M=4, K=248320, N=1024
        //   → self_too_large=false (M*K=992K), mat2_too_large=true (K*N=248M)
        //   → self_f32 = [4, 248320] f32 — upcast once (3.7MB, safe)
        //   → self_f32_t = [248320, 4] f32 — for column extraction
        //   → each k_chunk: extract self_f32_t[k:k+cs] → [k_cs, 4] → .t() → [4, k_cs]
        //   → extract mat2_c[k:k+cs] → fresh Vulkan tensor → upcast → [k_cs, N=1024]
        //   → mm: [4, k_cs] @ [k_cs, 1024] → [4, 1024] partial → accumulate

        // Upcast small matrix once outside the loop
        at::Tensor self_f32, self_f32_t, mat2_f32;
        if (!self_too_large) {
            self_f32 = ensure_float32(self_c);          // [M, K] f32 — safe
            self_f32_t = self_f32.t().contiguous();     // [K, M] f32 — for row extraction
        }
        if (!mat2_too_large) {
            mat2_f32 = ensure_float32(mat2_c);          // [K, N] f32 — safe
        }

        for (int64_t k_start = 0; k_start < K; k_start += k_chunk) {
            int64_t k_end = std::min(k_start + k_chunk, K);

            // A chunk: self_c columns [k_start:k_end] → [M, k_cs]
            at::Tensor a_chunk;
            if (self_too_large) {
                // Upcast self_c inside loop (not cached since it's too large)
                // Extract columns: upcast full → transpose → extract rows → transpose back
                // This is valid because even if self_c is large, we split K
                auto sc_f32 = ensure_float32(extract_rows(
                    ensure_float32(self_c).t().contiguous(), k_start, k_end));
                a_chunk = sc_f32.t().contiguous();
            } else {
                // Extract column-slice from cached transposed f32
                a_chunk = extract_rows(self_f32_t, k_start, k_end).t().contiguous();
            }

            // B chunk: mat2_c rows [k_start:k_end] → [k_cs, N]
            at::Tensor b_chunk;
            if (mat2_too_large) {
                b_chunk = ensure_float32(extract_rows(mat2_c, k_start, k_end));
            } else {
                b_chunk = extract_rows(mat2_f32, k_start, k_end);
            }

            // Partial result: [M, k_cs] @ [k_cs, N] = [M, N]
            auto partial = at::zeros({M, N}, at::TensorOptions().dtype(at::kFloat)
                                                                  .device(a_chunk.device()));
            mm_tiled_dispatch(a_chunk, b_chunk, partial, false, false);
            output_f32 = vulkan_add(output_f32, partial, 1);
        }
        return cast_from_float32(output_f32, orig_dtype);
    }

    // Fast path: both matrices fit within workgroup limits
    auto self_f32 = ensure_float32(self_c);
    auto mat2_f32 = ensure_float32(mat2_c);
    // Use empty (not zeros) — mm_tiled_fwd writes every output element
    auto output_f32 = at::empty({M, N}, self_c.options().dtype(at::kFloat));

    struct { uint32_t M, N, K, transpose_a, transpose_b; } params{
        static_cast<uint32_t>(M),
        static_cast<uint32_t>(N),
        static_cast<uint32_t>(K),
        0u, 0u  // transpose already handled above via self_c/mat2_c
    };
    uint32_t wg_x = (static_cast<uint32_t>(M) + 15) / 16;
    uint32_t wg_y = (static_cast<uint32_t>(N) + 15) / 16;
    dispatch_shader("matmul_mm_tiled_fwd",
                    shaders::matmul_mm_tiled_fwd, shaders::matmul_mm_tiled_fwd_size,
                    {self_f32, mat2_f32, output_f32},
                    wg_x, wg_y, 1,
                    &params, sizeof(params));
    return cast_from_float32(output_f32, orig_dtype);
}

at::Tensor vulkan_mm_ex(const at::Tensor& self, const at::Tensor& mat2,
                         bool transpose_a, bool transpose_b) {
    // Matmul with explicit transpose flags — avoids GPU permute copy from .t()
    // self is [M, K] (or [K, M] if transpose_a), mat2 is [K, N] (or [N, K] if transpose_b)
    // Both tensors must be contiguous in their stored layout.
    // Uses the optimized tiled mm shader with shared memory.

    auto self_c = self.contiguous();
    auto mat2_c = mat2.contiguous();

    check_supported_float(self_c, "mm_ex");
    auto orig_dtype = self_c.scalar_type();

    self_c = ensure_float32(self_c);
    mat2_c = ensure_float32(mat2_c);

    // Logical dimensions after transpose
    int64_t M = transpose_a ? self.size(1) : self.size(0);
    int64_t K = transpose_a ? self.size(0) : self.size(1);
    int64_t N = transpose_b ? mat2.size(0) : mat2.size(1);

    TORCH_CHECK((transpose_a ? self.size(0) : self.size(1)) ==
                (transpose_b ? mat2.size(1) : mat2.size(0)),
                "mm_ex: incompatible dimensions");

    auto output = at::empty({M, N}, self_c.options());
    if (M == 0 || N == 0 || K == 0) return output;

    struct { uint32_t M, N, K, transpose_a, transpose_b; } params{
        static_cast<uint32_t>(M),
        static_cast<uint32_t>(N),
        static_cast<uint32_t>(K),
        transpose_a ? 1u : 0u,
        transpose_b ? 1u : 0u
    };
    uint32_t wg_x = (static_cast<uint32_t>(M) + 15) / 16;
    uint32_t wg_y = (static_cast<uint32_t>(N) + 15) / 16;
    dispatch_shader("matmul_mm_tiled_fwd",
                    shaders::matmul_mm_tiled_fwd, shaders::matmul_mm_tiled_fwd_size,
                    {self_c, mat2_c, output},
                    wg_x, wg_y, 1,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

at::Tensor vulkan_addmm(
    const at::Tensor& bias,
    const at::Tensor& self,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {

    // addmm(bias, self, mat2, beta, alpha) = beta * bias + alpha * (self @ mat2)
    auto mm_result = vulkan_mm(self, mat2);

    // For the common case beta=1, alpha=1 (e.g. nn.Linear)
    if (beta.toDouble() == 1.0 && alpha.toDouble() == 1.0) {
        // Use vulkan_add which handles dtype conversion (f16/bf16 safe)
        return vulkan_add(mm_result, bias, 1);
    }

    // General case: use GPU ops for scaling
    auto alpha_mm = vulkan_mul_scalar(mm_result, alpha);
    auto beta_bias = vulkan_mul_scalar(bias, beta);

    return vulkan_add(beta_bias, alpha_mm, 1);
}

// ── Batched Matrix Multiplication ────────────────────────────────
// Check if tensor is a transpose of its last two dims (i.e., the only
// non-contiguous dims are the last two, and they are swapped).
at::Tensor vulkan_bmm(const at::Tensor& self, const at::Tensor& mat2) {
    TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3,
                "bmm: expected 3D tensors");

    // Fast path: detect zero-copy last-2-dims-transposed views from vulkan_transpose().
    // vulkan_transpose() creates metadata-only views with swapped strides for float32.
    // Recover the physical (contiguous) storage and pass transpose flags to the shader.
    bool ta = is_last2_transposed(self);
    bool tb = is_last2_transposed(mat2);
    if (ta || tb) {
        auto self_use = ta ? get_physical_storage_nd(self) : self;
        auto mat2_use = tb ? get_physical_storage_nd(mat2) : mat2;
        return vulkan_bmm_ex(self_use, mat2_use, ta, tb);
    }

    auto self_c = self.contiguous();
    auto mat2_c = mat2.contiguous();

    TORCH_CHECK(self_c.size(0) == mat2_c.size(0), "bmm: batch sizes must match");
    TORCH_CHECK(self_c.size(2) == mat2_c.size(1), "bmm: incompatible matrix sizes");
    check_supported_float(self_c, "bmm");
    auto orig_dtype = self_c.scalar_type();

    // Widen to f32 for compute
    self_c = ensure_float32(self_c);
    mat2_c = ensure_float32(mat2_c);

    int64_t B = self_c.size(0);
    int64_t M = self_c.size(1);
    int64_t K = self_c.size(2);
    int64_t N = mat2_c.size(2);

    auto output = at::empty({B, M, N}, self_c.options());

    if (B == 0 || M == 0 || N == 0 || K == 0) return output;

    struct { uint32_t batch, M, N, K, transpose_a, transpose_b; float scale; } params{
        static_cast<uint32_t>(B),
        static_cast<uint32_t>(M),
        static_cast<uint32_t>(N),
        static_cast<uint32_t>(K),
        0u, 0u,  // both inputs are contiguous row-major
        1.0f     // scale = 1.0 (no scaling)
    };

    uint32_t wg_x = (static_cast<uint32_t>(M) + 15) / 16;
    uint32_t wg_y = (static_cast<uint32_t>(N) + 15) / 16;
    uint32_t wg_z = static_cast<uint32_t>(B);

    dispatch_shader("matmul_bmm_tiled_fwd",
                    shaders::matmul_bmm_tiled_fwd, shaders::matmul_bmm_tiled_fwd_size,
                    {self_c, mat2_c, output},
                    wg_x, wg_y, wg_z,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

at::Tensor vulkan_bmm_ex(const at::Tensor& self, const at::Tensor& mat2,
                          bool transpose_a, bool transpose_b,
                          float scale) {
    // Batched matmul with explicit transpose flags and optional output scaling.
    // self is [B, M, K] (or [B, K, M] if transpose_a)
    // mat2 is [B, K, N] (or [B, N, K] if transpose_b)
    // Both tensors must be contiguous in their stored layout.
    // scale: output multiplied by this constant (fused with matmul, saves 1 dispatch vs separate mul)

    TORCH_CHECK(self.dim() == 3 && mat2.dim() == 3, "bmm_ex: expected 3D tensors");
    TORCH_CHECK(self.size(0) == mat2.size(0), "bmm_ex: batch sizes must match");

    auto self_c = self.contiguous();
    auto mat2_c = mat2.contiguous();

    check_supported_float(self_c, "bmm_ex");
    auto orig_dtype = self_c.scalar_type();

    self_c = ensure_float32(self_c);
    mat2_c = ensure_float32(mat2_c);

    int64_t B = self.size(0);
    int64_t M = transpose_a ? self.size(2) : self.size(1);
    int64_t K = transpose_a ? self.size(1) : self.size(2);
    int64_t N = transpose_b ? mat2.size(1) : mat2.size(2);

    TORCH_CHECK((transpose_a ? self.size(1) : self.size(2)) ==
                (transpose_b ? mat2.size(2) : mat2.size(1)),
                "bmm_ex: incompatible dimensions");

    auto output = at::empty({B, M, N}, self_c.options());
    if (B == 0 || M == 0 || N == 0 || K == 0) return output;

    struct { uint32_t batch, M, N, K, transpose_a, transpose_b; float scale; } params{
        static_cast<uint32_t>(B),
        static_cast<uint32_t>(M),
        static_cast<uint32_t>(N),
        static_cast<uint32_t>(K),
        transpose_a ? 1u : 0u,
        transpose_b ? 1u : 0u,
        scale
    };

    uint32_t wg_x = (static_cast<uint32_t>(M) + 15) / 16;
    uint32_t wg_y = (static_cast<uint32_t>(N) + 15) / 16;
    uint32_t wg_z = static_cast<uint32_t>(B);

    dispatch_shader("matmul_bmm_tiled_fwd",
                    shaders::matmul_bmm_tiled_fwd, shaders::matmul_bmm_tiled_fwd_size,
                    {self_c, mat2_c, output},
                    wg_x, wg_y, wg_z,
                    &params, sizeof(params));
    return cast_from_float32(output, orig_dtype);
}

// ── Scaled BMM: scale * (q @ k.T) — fuses scale multiply into bmm dispatch ──
// Saves 1 dispatch vs separate bmm + mul_scalar. Used for attention: (q @ kT) * head_dim**-0.5
at::Tensor vulkan_scaled_bmm_forward(const at::Tensor& q, const at::Tensor& k,
                                      float scale) {
    // q: [B, S, K], k: [B, K, N] (or transpose will be detected)
    // Returns scale * (q @ k.T) in a single dispatch via vulkan_bmm_ex
    bool tb = is_last2_transposed(k);
    auto k_use = tb ? get_physical_storage_nd(k) : k;
    bool ta = is_last2_transposed(q);
    auto q_use = ta ? get_physical_storage_nd(q) : q;
    return vulkan_bmm_ex(q_use, k_use, ta, tb, scale);
}

// ── Linear (F.linear) ───────────────────────────────────────────

at::Tensor vulkan_linear(const at::Tensor& input, const at::Tensor& weight,
                         const std::optional<at::Tensor>& bias_opt) {
    // linear(input, weight, bias) = input @ weight.T + bias
    // All Vulkan tensors are contiguous (opaque allocator), skip .contiguous()
    check_supported_float(input, "linear");

    int64_t in_features = input.size(-1);
    int64_t out_features = weight.size(0);
    bool needs_reshape = (input.dim() != 2);

    at::Tensor input_2d;
    if (needs_reshape) {
        int64_t batch = input.numel() / in_features;
        input_2d = input.reshape({batch, in_features});
    } else {
        input_2d = input;
    }

    at::Tensor mm_result;

    // Chunk size: largest multiple of 256 such that chunk_size*in_features <= kMaxCastElements.
    // This ensures the f32 upcast of each weight chunk fits within max Vulkan workgroup dispatch.
    const int64_t chunk_size = std::max(int64_t(256),
        (kMaxCastElements / in_features) & ~int64_t(255));  // round down to multiple of 256
    // Only chunk if (a) weight is too large for ensure_float32's 1D cast shader AND (b) not already f32.
    // For f32 weights, ensure_float32 is a no-op — no cast shader dispatch, no workgroup overflow.
    const bool already_f32 = (weight.scalar_type() == at::kFloat);
    const bool needs_chunk = !already_f32 && (out_features * in_features > kMaxCastElements);

    // Resolve weight: if it's a zero-copy transposed view (e.g., from w.t()),
    // use physical storage with tb=false (no extra transpose in shader).
    // Otherwise, weight is [out, in] contiguous and we need tb=true.
    bool weight_is_transposed = is_t_transposed(weight);
    at::Tensor weight_phys;
    bool tb_flag;
    if (weight_is_transposed) {
        // weight is view [out_features, in_features] with physical [in_features, out_features]
        // We want input @ weight_phys where weight_phys = [in_features, out_features]
        // → mm_ex(input, weight_phys, ta=false, tb=false) computes input[M,K] @ weight_phys[K,N]
        weight_phys = get_physical_storage(weight, true);  // [in_features, out_features]
        tb_flag = false;
    } else {
        weight_phys = weight;  // [out_features, in_features], needs transpose in shader
        tb_flag = true;
    }

    if (needs_chunk) {
        // Chunked matmul: split weight along out_features to avoid two issues:
        // 1. OOM: f32 upcast of full weight (e.g., 248320*1024*4 = 970MB) overflows VMA limit
        // 2. Workgroup overflow: cast shader uses 1D dispatch, max 65535 workgroups
        //
        // We need to extract weight rows [start:end] as a fresh contiguous Vulkan tensor.
        // vulkan_contiguous() returns self (all Vulkan tensors are "contiguous"), so
        // we must use VulkanBuffer::read+write to copy the row range via CPU staging.
        flush_stream();  // Ensure all pending GPU writes to weight are visible before readback
        auto& alloc = VulkanAllocator::instance();
        auto* weight_buf = alloc.get_buffer(weight.data_ptr());
        TORCH_CHECK(weight_buf, "chunked linear: cannot get Vulkan buffer for weight");

        const int64_t bytes_per_row = in_features * weight.element_size();
        // Temporary CPU buffer for staging one chunk at a time
        std::vector<uint8_t> staging(chunk_size * bytes_per_row);

        std::vector<at::Tensor> chunks;
        chunks.reserve((out_features + chunk_size - 1) / chunk_size);
        for (int64_t start = 0; start < out_features; start += chunk_size) {
            int64_t end = std::min(start + chunk_size, out_features);
            int64_t rows = end - start;

            // Read weight rows [start:end] from GPU buffer to CPU staging
            VkDeviceSize src_offset = static_cast<VkDeviceSize>(start * bytes_per_row);
            VkDeviceSize copy_bytes = static_cast<VkDeviceSize>(rows * bytes_per_row);
            weight_buf->read(staging.data(), copy_bytes, src_offset);

            // Upload chunk to a fresh Vulkan tensor
            auto weight_chunk = at::empty({rows, in_features}, weight.options());
            auto* chunk_buf = alloc.get_buffer(weight_chunk.data_ptr());
            TORCH_CHECK(chunk_buf, "chunked linear: cannot get Vulkan buffer for chunk");
            chunk_buf->write(staging.data(), copy_bytes);

            auto chunk_result = vulkan_mm_ex(input_2d, weight_chunk,
                                             /*transpose_a=*/false, /*transpose_b=*/true);
            if (bias_opt.has_value() && bias_opt->defined()) {
                // Bias is small (out_features,) — slice via CPU staging too
                auto* bias_buf = alloc.get_buffer(bias_opt->data_ptr());
                TORCH_CHECK(bias_buf, "chunked linear: cannot get Vulkan buffer for bias");
                const int64_t bias_elem_size = bias_opt->element_size();
                std::vector<uint8_t> bias_staging(rows * bias_elem_size);
                bias_buf->read(bias_staging.data(),
                               static_cast<VkDeviceSize>(rows * bias_elem_size),
                               static_cast<VkDeviceSize>(start * bias_elem_size));
                auto bias_chunk = at::empty({rows}, bias_opt->options());
                auto* bc_buf = alloc.get_buffer(bias_chunk.data_ptr());
                TORCH_CHECK(bc_buf, "chunked linear: cannot get Vulkan buffer for bias chunk");
                bc_buf->write(bias_staging.data(), rows * bias_elem_size);

                auto bias_expanded = bias_chunk.unsqueeze(0).expand_as(chunk_result);
                chunk_result = vulkan_add(chunk_result, bias_expanded.contiguous(), 1);
            }
            chunks.push_back(chunk_result);
        }
        mm_result = at::cat(chunks, /*dim=*/1);
    } else {
        // Use vulkan_mm_ex: if weight is a zero-copy transposed view, use physical
        // storage with tb=false; otherwise weight=[out,in] and we transpose in shader.
        mm_result = vulkan_mm_ex(input_2d, weight_phys, /*transpose_a=*/false, /*transpose_b=*/tb_flag);

        if (bias_opt.has_value() && bias_opt->defined()) {
            auto bias = *bias_opt;
            auto bias_expanded = bias.unsqueeze(0).expand_as(mm_result);
            // expand returns a view — on Vulkan with opaque allocator this
            // triggers a copy, but it's needed for the add.
            mm_result = vulkan_add(mm_result, bias_expanded.contiguous(), 1);
        }
    }

    if (needs_reshape) {
        auto orig_shape = input.sizes().vec();
        std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end() - 1);
        out_shape.push_back(out_features);
        return mm_result.reshape(out_shape);
    }
    return mm_result;
}

// ── Scaled Matrix Multiplication (FP8 training) ─────────────────
at::Tensor vulkan_scaled_mm(
    const at::Tensor& self, const at::Tensor& mat2,
    const at::Tensor& scale_a, const at::Tensor& scale_b,
    const std::optional<at::Tensor>& bias_opt,
    const std::optional<at::Tensor>& scale_result_opt,
    std::optional<at::ScalarType> out_dtype_opt,
    bool use_fast_accum) {

    // _scaled_mm(a_fp8, b_fp8, scale_a, scale_b) computes:
    //   result = (a * scale_a) @ (b * scale_b)
    // where scale_a/scale_b are scalar tensors.

    TORCH_CHECK(self.dim() == 2 && mat2.dim() == 2,
                "_scaled_mm: expected 2D tensors");
    TORCH_CHECK(self.size(1) == mat2.size(0),
                "_scaled_mm: incompatible dimensions");

    // Widen FP8 to f32
    auto a_f32 = ensure_float32(self.contiguous());
    auto b_f32 = ensure_float32(mat2.contiguous());

    // Apply scales (multiplicative — dequantize FP8 values)
    float sa = scale_a.cpu().item<float>();
    float sb = scale_b.cpu().item<float>();

    if (sa != 1.0f) {
        a_f32 = vulkan_mul_scalar(a_f32, at::Scalar(sa));
    }
    if (sb != 1.0f) {
        b_f32 = vulkan_mul_scalar(b_f32, at::Scalar(sb));
    }

    // Compute matmul in f32
    auto result = vulkan_mm(a_f32, b_f32);

    // Apply result scale if provided
    if (scale_result_opt.has_value() && scale_result_opt->defined()) {
        float sr = scale_result_opt->cpu().item<float>();
        if (sr != 1.0f) {
            result = vulkan_mul_scalar(result, at::Scalar(sr));
        }
    }

    // Add bias if provided
    if (bias_opt.has_value() && bias_opt->defined()) {
        result = vulkan_add(result, bias_opt->contiguous(), 1);
    }

    // Cast to output dtype
    auto target_dtype = out_dtype_opt.value_or(at::kFloat);
    return cast_from_float32(result, target_dtype);
}

}} // namespace torch_vulkan::ops
