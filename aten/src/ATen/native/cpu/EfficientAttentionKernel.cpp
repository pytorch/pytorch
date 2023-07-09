#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#endif

namespace at::native {

namespace {

template <typename scalar_t>
static inline void fill_stub(scalar_t* data, scalar_t val, int64_t size) {
  using Vec = Vectorized<scalar_t>;
  Vec data_vec = Vec(val);
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    data_vec.store(data + d);
  }
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (; d < size; d++) {
    data[d] = val;
  }
}

// NB: CPU kernel for efficient attention
//
// Note that inputs and outputs shapes are expected in physical order
//
//   query : (B, M, H, K)
//   key   : (B, N, H, K)
//   value : (B, N, H, Kv)
//
//   attn  : (B, M, H, Kv)
//   log_sumexp : (B, M, H)
//
template <typename scalar_t, int64_t kQueriesPerBlock, int64_t kKeysPerBlock>
void cpu_efficient_attention(
    const Tensor& attn,
    const Tensor& logsumexp,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool compute_logsumexp,
    bool is_causal,
    c10::optional<double> scale) {

  using Vec = vec::Vectorized<scalar_t>;
  scalar_t scaling_factor = sdp::calculate_scale(query, scale).as_float_unchecked();

  // sizes
  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t H = query.size(2);
  int64_t N = key.size(1);
  int64_t K = query.size(-1);
  int64_t Kv = value.size(-1);

  // strides
  int64_t q_strideB = query.stride(0);
  int64_t q_strideM = query.stride(1);
  int64_t q_strideH = query.stride(2);
  int64_t k_strideB = key.stride(0);
  int64_t k_strideN = key.stride(1);
  int64_t k_strideH = key.stride(2);
  int64_t v_strideB = value.stride(0);
  int64_t v_strideN = value.stride(1);
  int64_t v_strideH = value.stride(2);

  int64_t o_strideB = attn.stride(0);
  int64_t o_strideM = attn.stride(1);
  int64_t o_strideH = attn.stride(2);
  int64_t l_strideB = logsumexp.stride(0);
  int64_t l_strideM = logsumexp.stride(1);
  int64_t l_strideH = logsumexp.stride(2);

  scalar_t* q_data = query.data_ptr<scalar_t>();
  scalar_t* k_data = key.data_ptr<scalar_t>();
  scalar_t* v_data = value.data_ptr<scalar_t>();
  scalar_t* out_data = attn.data_ptr<scalar_t>();
  scalar_t* lse_data = compute_logsumexp ? logsumexp.data_ptr<scalar_t>() : nullptr;

  // number of blocks along M
  int64_t MB = divup(M, kQueriesPerBlock);

  // allocate per thread temp buffer
  int64_t size_per_thread =
      /* s_i and s_delta */ kQueriesPerBlock * kKeysPerBlock +
      /* v_prime         */ kQueriesPerBlock * Kv;

  auto buffer = at::empty({at::get_num_threads(), size_per_thread}, query.options());
  scalar_t* buf_data = buffer.data_ptr<scalar_t>();

  // TODO: try blocking on H
  // parallel on B, H, MB
  at::parallel_for(0, B * H * MB, 1, [&](int64_t begin, int64_t end) {
    int64_t b{0}, h{0}, mb{0};
    data_index_init(begin, b, B, h, H, mb, MB);

    // get thread local slices
    int tid = at::get_thread_num();
    scalar_t* buf_ptr = buf_data + tid * size_per_thread;

    // s_i and s_delta: (kQueriesPerBlock, kKeysPerBlock)
    scalar_t* s_i = buf_ptr;

    // s_delta: (kQueriesPerBlock, kKeysPerBlock)
    scalar_t* s_delta = s_i;

    // v': (kQueriesPerBlock, Kv)
    scalar_t* v_prime = s_i + kQueriesPerBlock * kKeysPerBlock;

    scalar_t s_prime[kQueriesPerBlock];
    scalar_t m_prime[kQueriesPerBlock];

    for (const auto i : c10::irange(begin, end)) {
      (void)i; // Suppress unused variable

      int64_t m = mb * kQueriesPerBlock;
      int64_t block_size_m = std::min(kQueriesPerBlock, M - m);

      scalar_t* q_ptr = q_data + b * q_strideB + m * q_strideM + h * q_strideH;

      // init v', s' and m'
      fill_stub<scalar_t>(v_prime, 0, block_size_m * Kv);
      fill_stub<scalar_t>(s_prime, 0, block_size_m);
      fill_stub<scalar_t>(m_prime, -std::numeric_limits<scalar_t>::infinity(), block_size_m);

      // loop over Q and V sequence with block size of kKeysPerBlock
      int64_t num_keys = is_causal ? std::min(m + block_size_m, N) : N;
      for (int64_t n = 0; n < num_keys; n += kKeysPerBlock) {
        int64_t block_size_n = std::min(kKeysPerBlock, N - n);
        scalar_t* k_ptr = k_data + b * k_strideB + n * k_strideN + h * k_strideH;


        // TODO: template me!
        // calculate attn: Q @ K.T
        cpublas::gemm(
            TransposeType::Transpose,
            TransposeType::NoTranspose,
            block_size_n,
            block_size_m,
            K,
            scaling_factor,
            k_ptr,
            k_strideN,
            q_ptr,
            q_strideM,
            static_cast<scalar_t>(0),
            s_i,
            kKeysPerBlock);

        // apply causal mask
        if (is_causal && num_keys - n <= kKeysPerBlock) {
          for (const auto row : c10::irange(block_size_m)) {
            int64_t last_col = m + row - n;

            // fill [last_col + 1, block_size_n) to -inf
            //
            // for (const auto col : c10::irange(block_size_n)) {
            //   if (col > last_col) {
            //     int64_t idx = row * kKeysPerBlock + col;
            //     s_i[idx] = -std::numeric_limits<scalar_t>::infinity();
            //   }
            // }
            scalar_t* row_ptr = s_i + row * kKeysPerBlock;
            fill_stub(row_ptr + last_col + 1, -std::numeric_limits<scalar_t>::infinity(), block_size_n - last_col - 1);
          }
        }

        // update the scaling coefficients
        for (const auto row : c10::irange(block_size_m)) {
          // m_i: max value per row
          scalar_t m_i = vec::reduce_all<scalar_t>(
              [](Vec& x, Vec& y) { return vec::maximum(x, y); },
              s_i + row * kKeysPerBlock,
              block_size_n);
          m_i = std::max(m_i, m_prime[row]);

          // m_delta <- exp(m' - m_i)
          scalar_t m_delta = std::exp(m_prime[row] - m_i);

          // s_delta <- exp(s_i - m_i)
          vec::map<scalar_t>(
              [m_i](Vec x) { return (x - Vec(m_i)).exp(); },
              s_delta + row * kKeysPerBlock,
              s_i + row * kKeysPerBlock,
              block_size_n);

          // s' <- s' * m_delta + sum(s_delta)
          s_prime[row] *= m_delta;
          s_prime[row] += vec::reduce_all<scalar_t>(
              [](Vec& x, Vec& y) { return x + y; },
              s_delta + row * kKeysPerBlock,
              block_size_n);

          m_prime[row] = m_i;

          // v' <- v' * m_delta
          vec::map<scalar_t>(
              [m_delta](Vec x) { return x * Vec(m_delta); },
              v_prime + row * Kv,
              v_prime + row * Kv,
              Kv);
        }

        // caculate V' <- s_delta @ V + V'
        scalar_t* v_ptr = v_data + b * v_strideB + n * v_strideN + h * v_strideH;
        cpublas::gemm(
            TransposeType::NoTranspose,
            TransposeType::NoTranspose,
            Kv,
            block_size_m,
            block_size_n,
            static_cast<scalar_t>(1),
            v_ptr,
            v_strideN,
            s_delta,
            kKeysPerBlock,
            static_cast<scalar_t>(1),
            v_prime,
            Kv);
      }

      scalar_t* out_ptr = out_data + b * o_strideB + m * o_strideM + h * o_strideH;
      for (const auto row : c10::irange(block_size_m)) {
        scalar_t s = 1 / s_prime[row];
        vec::map<scalar_t>(
            [s](Vec out) { return out * Vec(s); },
            out_ptr + row * o_strideM,
            v_prime + row * Kv,
            Kv);
      }

      if (compute_logsumexp) {
        scalar_t* lse_ptr = lse_data + b * l_strideB + m * l_strideM + h * l_strideH;
        for (const auto row : c10::irange(block_size_m)) {
          lse_ptr[row * l_strideM] = m_prime[row] + std::log(s_prime[row]);
        }
      }

      // move to the next query
      data_index_step(b, B, h, H, mb, MB);
    }
  });
}

template <typename scalar_t, int64_t kQueriesPerBlock, int64_t kKeysPerBlock>
void cpu_efficient_attention_backward(
    const Tensor& grad_q,
    const Tensor& grad_k,
    const Tensor& grad_v,
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    bool is_causal,
    c10::optional<double> scale) {

  using Vec = vec::Vectorized<scalar_t>;
  scalar_t scaling_factor = sdp::calculate_scale(query, scale).as_float_unchecked();

  // sizes
  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t H = query.size(2);
  int64_t N = key.size(1);
  int64_t K = query.size(-1);
  int64_t Kv = value.size(-1);

  // strides
  int64_t q_strideB = query.stride(0);
  int64_t q_strideM = query.stride(1);
  int64_t q_strideH = query.stride(2);
  int64_t k_strideB = key.stride(0);
  int64_t k_strideN = key.stride(1);
  int64_t k_strideH = key.stride(2);
  int64_t v_strideB = value.stride(0);
  int64_t v_strideN = value.stride(1);
  int64_t v_strideH = value.stride(2);
  int64_t o_strideB = out.stride(0);
  int64_t o_strideM = out.stride(1);
  int64_t o_strideH = out.stride(2);

  int64_t l_strideB = logsumexp.stride(0);
  int64_t l_strideM = logsumexp.stride(1);
  int64_t l_strideH = logsumexp.stride(2);

  int64_t gQ_strideB = grad_q.stride(0);
  int64_t gQ_strideM = grad_q.stride(1);
  int64_t gQ_strideH = grad_q.stride(2);
  int64_t gK_strideB = grad_k.stride(0);
  int64_t gK_strideN = grad_k.stride(1);
  int64_t gK_strideH = grad_k.stride(2);
  int64_t gV_strideB = grad_v.stride(0);
  int64_t gV_strideN = grad_v.stride(1);
  int64_t gV_strideH = grad_v.stride(2);
  int64_t gO_strideB = grad_out.stride(0);
  int64_t gO_strideM = grad_out.stride(1);
  int64_t gO_strideH = grad_out.stride(2);

  scalar_t* q_data = query.data_ptr<scalar_t>();
  scalar_t* k_data = key.data_ptr<scalar_t>();
  scalar_t* v_data = value.data_ptr<scalar_t>();
  scalar_t* out_data = out.data_ptr<scalar_t>();
  scalar_t* lse_data = logsumexp.data_ptr<scalar_t>();
  scalar_t* grad_q_data = grad_q.data_ptr<scalar_t>();
  scalar_t* grad_k_data = grad_k.data_ptr<scalar_t>();
  scalar_t* grad_v_data = grad_v.data_ptr<scalar_t>();
  scalar_t* grad_o_data = grad_out.data_ptr<scalar_t>();

  // allocate per thread temp buffer
  int64_t size_per_thread =
      /* attn_v          */ kQueriesPerBlock * kKeysPerBlock +
      /* grad_attn_v     */ kQueriesPerBlock * kKeysPerBlock;

  auto buffer = at::empty({at::get_num_threads(), size_per_thread}, query.options());
  scalar_t* buf_data = buffer.data_ptr<scalar_t>();

  at::parallel_for(0, B * H, 1, [&](int64_t begin, int64_t end) {
    int64_t b{0}, h{0};
    data_index_init(begin, b, B, h, H);

    // get thread local slices
    int tid = at::get_thread_num();
    scalar_t* buf_ptr = buf_data + tid * size_per_thread;

    // attn_v: (kQueriesPerBlock, kKeysPerBlock)
    scalar_t* attn_v = buf_ptr;

    // grad_attn_v: (kQueriesPerBlock, kKeysPerBlock)
    scalar_t* grad_attn_v = attn_v + kQueriesPerBlock * kKeysPerBlock;

    // rowsum of grad_out * out
    scalar_t sum[kQueriesPerBlock];

    for (const auto i : c10::irange(begin, end)) {
      (void)i; // Suppress unused variable

      for (int64_t m = 0; m < M; m += kQueriesPerBlock) {
        int64_t block_size_m = std::min(kQueriesPerBlock, M - m);

        scalar_t* q_ptr = q_data + b * q_strideB + m * q_strideM + h * q_strideH;
        scalar_t* lse_ptr = lse_data + b * l_strideB + m * l_strideM + h * l_strideH;
        scalar_t* out_ptr = out_data + b * o_strideB + m * o_strideM + h * o_strideH;
        scalar_t* grad_o_ptr = grad_o_data + b * gO_strideB + m * gO_strideM + h * gO_strideH;
        scalar_t* grad_q_ptr = grad_q_data + b * gQ_strideB + m * gQ_strideM + h * gQ_strideH;

        // sum <- rowsum(grad_out * out)
        for (const auto row : c10::irange(block_size_m)) {
          sum[row] = vec::map2_reduce_all<scalar_t>(
              [](Vec x, Vec y) { return x * y; },
              [](Vec x, Vec y) { return x + y; },
              grad_o_ptr + row * gO_strideM,
              out_ptr + row * o_strideM,
              Kv);
        }

        // loop over Q and V sequence with block size of kKeysPerBlock
        int64_t num_keys = is_causal ? std::min(m + block_size_m, N) : N;
        for (int64_t n = 0; n < num_keys; n += kKeysPerBlock) {
          int64_t block_size_n = std::min(kKeysPerBlock, N - n);
          scalar_t* k_ptr = k_data + b * k_strideB + n * k_strideN + h * k_strideH;

          // calculate attn_v <- q @ k.T
          cpublas::gemm(
              TransposeType::Transpose,
              TransposeType::NoTranspose,
              block_size_n,
              block_size_m,
              K,
              static_cast<scalar_t>(1),
              k_ptr,
              k_strideN,
              q_ptr,
              q_strideM,
              static_cast<scalar_t>(0),
              attn_v,
              kKeysPerBlock);

          // restore self attention after softmax from logsumexp
          // attn_v <- exp(attn_v * scale - normalizer)
          for (const auto row : c10::irange(block_size_m)) {
            scalar_t normalizer = lse_ptr[row * l_strideM];
            vec::map<scalar_t>(
                [=](Vec s) { return (s * Vec(scaling_factor) - Vec(normalizer)).exp(); },
                attn_v + row * kKeysPerBlock,
                attn_v + row * kKeysPerBlock,
                block_size_n);
          }

          // apply causal mask, fill unused with 0
          if (is_causal && num_keys - n <= kKeysPerBlock) {
            for (const auto row : c10::irange(block_size_m)) {
              int64_t last_col = m + row - n;
              scalar_t* row_ptr = attn_v + row * kKeysPerBlock;
              fill_stub<scalar_t>(row_ptr + last_col + 1, 0, block_size_n - last_col - 1);
            }
          }

          // calculate the gradient of V
          // grad_v <- grad_v + attn_v.T @ grad_out
          scalar_t* grad_v_ptr = grad_v_data + b * gV_strideB + n * gV_strideN + h * gV_strideH;
          cpublas::gemm(
              TransposeType::NoTranspose,
              TransposeType::Transpose,
              Kv,
              block_size_n,
              block_size_m,
              static_cast<scalar_t>(1),
              grad_o_ptr,
              gO_strideM,
              attn_v,
              kKeysPerBlock,
              static_cast<scalar_t>(1),
              grad_v_ptr,
              gV_strideN);

          // calculate the gradient of self attention after softmax
          // grad_attn_v <- grad_out @ v.T
          scalar_t* v_ptr = v_data + b * v_strideB + n * v_strideN + h * v_strideH;
          cpublas::gemm(
              TransposeType::Transpose,
              TransposeType::NoTranspose,
              block_size_n,
              block_size_m,
              Kv,
              static_cast<scalar_t>(1),
              v_ptr,
              v_strideN,
              grad_o_ptr,
              gO_strideM,
              static_cast<scalar_t>(0),
              grad_attn_v,
              kKeysPerBlock);

          // grad_attn_v <- attn_v * (grad_attn_v - rowsum)
          for (const auto row : c10::irange(block_size_m)) {
            scalar_t s = sum[row];
            vec::map2<scalar_t>(
                [s](Vec attn, Vec grad_attn) { return attn * (grad_attn - Vec(s)); },
                grad_attn_v + row * kKeysPerBlock,
                attn_v + row * kKeysPerBlock,
                grad_attn_v + row * kKeysPerBlock,
                block_size_n);
          }

          // calculate the gradient of Q
          // grad_q <- grad_q + scale * grad_attn_v @ k
          // buf <- attn_v @ k
          cpublas::gemm(
              TransposeType::NoTranspose,
              TransposeType::NoTranspose,
              K,
              block_size_m,
              block_size_n,
              scaling_factor,
              k_ptr,
              k_strideN,
              grad_attn_v,
              kKeysPerBlock,
              static_cast<scalar_t>(1),
              grad_q_ptr,
              gQ_strideM);

          // grad_k <- grad_k + scale * grad_attn_v.T @ q
          scalar_t* grad_k_ptr = grad_k_data + b * gK_strideB + n * gK_strideN + h * gK_strideH;
          cpublas::gemm(
              TransposeType::NoTranspose,
              TransposeType::Transpose,
              K,
              block_size_n,
              block_size_m,
              scaling_factor,
              q_ptr,
              q_strideM,
              grad_attn_v,
              kKeysPerBlock,
              static_cast<scalar_t>(1),
              grad_k_ptr,
              gK_strideN);
        } // N loop
      } // M loop

      // move to the next index
      data_index_step(b, B, h, H);
    }
  });
}

void efficient_attention_kernel_impl(
    const Tensor& attn,
    const Tensor& logsumexp,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool compute_logsumexp,
    bool is_causal,
    c10::optional<double> scale) {
  // TODO: add bfloat16 and float16
  AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "efficient_attention", [&] {
    cpu_efficient_attention<scalar_t, 128, 256>(attn, logsumexp, query, key, value, compute_logsumexp, is_causal, scale);
  });
}

void efficient_attention_backward_kernel_impl(
    const Tensor& grad_q,
    const Tensor& grad_k,
    const Tensor& grad_v,
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    bool is_causal,
    c10::optional<double> scale) {
  // make sure grad_out has no zero strides (broadcasted dimensions)
  // since we are going to call gemm next
  // zero stride in leading dimension would lead to slow impl for gemm
  auto grad_out_contig = grad_out.contiguous();

  AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "efficient_attention_backward", [&] {
    cpu_efficient_attention_backward<scalar_t, 128, 256>(grad_q, grad_k, grad_v, grad_out_contig, query, key, value, out, logsumexp, is_causal, scale);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(efficient_attention_kernel, &efficient_attention_kernel_impl);
REGISTER_DISPATCH(efficient_attention_backward_kernel, &efficient_attention_backward_kernel_impl)

} // at::native
