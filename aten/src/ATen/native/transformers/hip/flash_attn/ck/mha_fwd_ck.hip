/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <ATen/native/transformers/hip/flash_attn/flash_common_hip.hpp>
#include <fmha_fwd.hpp>
#include <mask.hpp>



namespace pytorch_flash {


fmha_fwd_traits get_ck_fmha_fwd_traits(const mask_info &mask,
                                       std::string dtype,
                                       int head_size,
                                       bool has_dropout,
                                       bool has_lse,
                                       bool enable_alibi)
{
    return fmha_fwd_traits{head_size,
                           head_size,
                           dtype,
                           false, // is_group_mode
                           true,  // is_v_rowmajor
                           mask.type,
                           enable_alibi ? bias_enum::alibi : bias_enum::no_bias,
                           has_lse,
                           has_dropout,
                           false}; // do_fp8_static_quant
}

fmha_fwd_args get_ck_fmha_fwd_args(bool has_lse,
                                   bool has_dropout_randval,
                                   const mask_info &mask,
                                   // sizes
                                   const int b,
                                   const int seqlen_q,
                                   const int seqlen_k,
                                   const int h,
                                   const int h_k,
                                   const int d,
                                   // device pointers
                                   const at::Tensor q,
                                   const at::Tensor k,
                                   const at::Tensor v,
                                   std::optional<at::Tensor> &alibi_slopes_,
                                   at::Tensor out,
                                   at::Tensor softmax_lse,
                                   at::Tensor dropout_randval,
                                   float softmax_scale,
                                   float p_dropout,
                                   std::pair<uint64_t*, uint64_t*> drop_seed_offset)
{
    // q: (batch_size, seqlen_q, nheads, d)
    // k: (batch_size, seqlen_k, nheads_k, d)
    // v: (batch_size, seqlen_k, nheads_k, d)
    // o: (batch_size, seqlen_q, nheads, d)

    // alibi_slopes:(batch_size, nheads) or (nhead)
    // lse: (batch_size, nheads, seqlen_q)
    // randval: (batch_size, nheads, seqlen_q, seqlen_k)

    ck_tile::index_t stride_q = q.stride(1);
    ck_tile::index_t stride_k = k.stride(1);
    ck_tile::index_t stride_v = v.stride(1);
    ck_tile::index_t stride_o = out.stride(1);
    ck_tile::index_t stride_randval = has_dropout_randval ? dropout_randval.stride(2) : 0;

    ck_tile::index_t nhead_stride_q = q.stride(2);
    ck_tile::index_t nhead_stride_k = k.stride(2);
    ck_tile::index_t nhead_stride_v = v.stride(2);
    ck_tile::index_t nhead_stride_o = out.stride(2);
    ck_tile::index_t nhead_stride_lse = has_lse ? softmax_lse.stride(1) : 0;
    ck_tile::index_t nhead_stride_randval = has_dropout_randval ? dropout_randval.stride(1) : 0;

    ck_tile::index_t batch_stride_q = q.stride(0);
    ck_tile::index_t batch_stride_k = k.stride(0);
    ck_tile::index_t batch_stride_v = v.stride(0);
    ck_tile::index_t batch_stride_o = out.stride(0);

    ck_tile::index_t batch_stride_lse = has_lse ? softmax_lse.stride(0) : 0;
    ck_tile::index_t batch_stride_randval = has_dropout_randval ? dropout_randval.stride(0) : 0;

    void *alibi_slopes_ptr = nullptr;
    ck_tile::index_t stride_alibi_slopes = 0;

    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == at::IntArrayRef({h}) || alibi_slopes.sizes() == at::IntArrayRef({b, h}));
        alibi_slopes_ptr = alibi_slopes.data_ptr();
        stride_alibi_slopes = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    }

    return fmha_fwd_args{q.data_ptr(),
                         k.data_ptr(),
                         v.data_ptr(),
                         alibi_slopes_ptr, // bias
                         has_dropout_randval ? dropout_randval.data_ptr() : nullptr,
                         has_lse ? softmax_lse.data_ptr() : nullptr,
                         out.data_ptr(),
                         nullptr, // seqstart_q
                         nullptr, // seqstart_k
                         nullptr,
                         seqlen_q,
                         seqlen_k,
                         b,
                         seqlen_q,      // max_seqlen_q
                         d,             // hdim_q
                         d,             // hdim_v
                         h,             // nhead
                         h_k,           // nhead_k
                         softmax_scale, // scale_s
                         1,             // scale_p
                         1,             // scale_o
                         stride_q,
                         stride_k,
                         stride_v,
                         stride_alibi_slopes,
                         stride_randval,
                         stride_o,
                         nhead_stride_q,
                         nhead_stride_k,
                         nhead_stride_v,
                         0, // nhead_stride_bias, FA without bias
                         nhead_stride_randval,
                         nhead_stride_lse,
                         nhead_stride_o,
                         batch_stride_q,
                         batch_stride_k,
                         batch_stride_v,
                         0, // batch_stride_bias, FA without bias
                         batch_stride_randval,
                         batch_stride_lse,
                         batch_stride_o,
                         mask.left,
                         mask.right,
                         static_cast<ck_tile::index_t>(mask.type),
                         p_dropout,
                         has_dropout_randval,
                         drop_seed_offset};
}
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_fwd_ck(const at::Tensor &q,                      // batch_size x seqlen_q x num_heads x head_size
           const at::Tensor &k,                      // batch_size x seqlen_k x num_heads_k x head_size
           const at::Tensor &v,                      // batch_size x seqlen_k x num_heads_k x head_size
           c10::optional<at::Tensor> &out_,          // batch_size x seqlen_q x num_heads xhead_size
           c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
           const float p_dropout,
           const float softmax_scale,
           bool is_causal,
           int window_size_left,
           int window_size_right,
           const bool return_dropout_randval,
           c10::optional<at::Generator> gen_)
{
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    std::string q_dtype_str = q_dtype == at::kHalf ? "fp16" : "bf16";

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }

    mask_info mask;
    if (is_causal) {
        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        window_size_right = 0;
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + "0";
        mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // casual
    }
    else if (window_size_left == -1 && window_size_right == -1) {
        mask = mask_info::decode("0", seqlen_q, seqlen_k); // no mask
    }
    else {
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + std::to_string(window_size_right);
        mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // local
    }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size % 8 == 0 && !alibi_slopes_.has_value();
    const int ngroups = num_heads / num_heads_k;
    at::Tensor temp_q = q;
    if (seqlenq_ngroups_swapped) {
        temp_q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    CHECK_SHAPE(temp_q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);


    at::Tensor q_padded, k_padded, v_padded;
    if (head_size % 8 != 0) {
        q_padded = at::pad(temp_q, {0, 8 - head_size % 8});
        k_padded = at::pad(k, {0, 8 - head_size % 8});
        v_padded = at::pad(v, {0, 8 - head_size % 8});
    }
    else {
        q_padded = temp_q;
        k_padded = k;
        v_padded = v;
    }


    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size);
        if (seqlenq_ngroups_swapped) {
            out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        }
        if (head_size % 8 != 0) { out = at::empty_like(q_padded); };
    }
    else {
        out = at::empty_like(q);
    }

    auto round_multiple = [](int x, int m) { return (x + m -1) / m*m;};
    const int head_size_8x = round_multiple(head_size, 8);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::hip::HIPGuardMasqueradingAsCUDA device_guard{(char)q.get_device()};

    auto opts = q.options();
    bool has_lse = true;
    bool has_dropout = p_dropout > 0.0f;

    at::Tensor softmax_lse;
    // TODO - check gradient, only training require lse
    softmax_lse = at::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    at::Tensor p;
    if (return_dropout_randval) {
        TORCH_CHECK(has_dropout, "return_dropout_randval require p_dropout > 0");
        p = at::empty({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(at::kByte));
    }
    else {
        p = at::empty({ 0 }, opts);
    }

    int64_t counter_offset = batch_size * num_heads * ck_tile::get_warp_size();
    auto rng_state = at::empty({2}, opts.dtype(at::kLong));
    auto rng_state_ptr = reinterpret_cast<uint64_t*>(rng_state.data_ptr());



    at::Tensor seed_t, offset_t;

    if (p_dropout > 0.0)  {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);

        auto philox_args = gen->philox_cuda_state(counter_offset);



        hipLaunchKernelGGL(
            flash::ParsePhiloxCudaState, dim3(1), dim3(64), 0, at::hip::getCurrentHIPStreamMasqueradingAsCUDA(), philox_args, rng_state_ptr);
        seed_t = at::scalar_tensor(at::Scalar(static_cast<uint64_t>(rng_state_ptr[0])), at::dtype(at::kLong));
        offset_t = at::scalar_tensor(at::Scalar(static_cast<uint64_t>(rng_state_ptr[1])), at::dtype(at::kLong));
    }
    else
    {
        seed_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
        offset_t = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
    }

    if (seqlen_k > 0) {
        auto drop_seed_offset = std::make_pair(rng_state_ptr, rng_state_ptr + 1);
        auto stream = at::cuda::getCurrentHIPStream().stream();
        ck_tile::stream_config stream_config{stream};

        auto traits =
            get_ck_fmha_fwd_traits(
                mask,
                q_dtype_str,
                head_size_8x,
                has_dropout,
                has_lse,
                alibi_slopes_.has_value());

        auto args =
            get_ck_fmha_fwd_args(
                has_lse,
                return_dropout_randval,
                mask,
                batch_size,
                seqlen_q,
                seqlen_k,
                num_heads,
                num_heads_k,
                head_size_8x,
                q,
                k,
                v,
                alibi_slopes_,
                out,
                softmax_lse,
                p,
                softmax_scale,
                p_dropout,
                drop_seed_offset);
#if (defined(__gfx90a__) || defined(__gfx942__))
        float t = fmha_fwd(traits, args, stream_config);
        TORCH_CHECK(t >= 0, "invalid argument for fmha_fwd");
#endif
    }
    else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        q_padded = q_padded.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    return {out, q_padded, k_padded, v_padded, softmax_lse, seed_t, offset_t, p};
}
} //namespace pytorch_flash
