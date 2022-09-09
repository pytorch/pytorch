// Copied from https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 rotary_embedding_coefficient(const int zid, const int rot_embed_dim, const float t_step)
{
    const float inv_freq = t_step / pow(10000.0f, zid / (float)rot_embed_dim);
    return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float2 rotary_embedding_transform(const float2 v, const float2 coef)
{
    float2 rot_v;
    rot_v.x = coef.x * v.x - coef.y * v.y;
    rot_v.y = coef.x * v.y + coef.y * v.x;
    return rot_v;
}

inline __device__ uint32_t rotary_embedding_transform(const uint32_t v, const float2 coef)
{
    float2 fv = half2_to_float2(v);
    float2 rot_fv = rotary_embedding_transform(fv, coef);
    return float2_to_half2(rot_fv);
}

#ifdef ENABLE_BF16
inline __device__ __nv_bfloat162 rotary_embedding_transform(const __nv_bfloat162 v, const float2 coef)
{
    float2 fv = bf1622float2(v);
    float2 rot_fv = rotary_embedding_transform(fv, coef);
    return __floats2bfloat162_rn(rot_fv.x, rot_fv.y);
}
#endif

inline __device__ void apply_rotary_embedding(float& q, int zid, int rot_embed_dim, int t_step)
{
    return;
}

inline __device__ void apply_rotary_embedding(float& q, float& k, int zid, int rot_embed_dim, int t_step)
{
    return;
}

inline __device__ void apply_rotary_embedding(float2& q, int tid, int rot_embed_dim, int t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
    q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(float2& q, float2& k, int tid, int rot_embed_dim, int t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
    q = rotary_embedding_transform(q, coef);
    k = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(float4& q, int tid, int rot_embed_dim, int t_step)
{
    if (4 * tid >= rot_embed_dim) {
        return;
    }

    Float4_& q_ = *reinterpret_cast<Float4_*>(&q);
    const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
    q_.x = rotary_embedding_transform(q_.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
    q_.y = rotary_embedding_transform(q_.y, coef1);
}

inline __device__ void apply_rotary_embedding(float4& q, float4& k, int tid, int rot_embed_dim, int t_step)
{
    if (4 * tid >= rot_embed_dim) {
        return;
    }

    Float4_& q_ = *reinterpret_cast<Float4_*>(&q);
    Float4_& k_ = *reinterpret_cast<Float4_*>(&k);
    const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
    q_.x = rotary_embedding_transform(q_.x, coef0);
    k_.x = rotary_embedding_transform(k_.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
    q_.y = rotary_embedding_transform(q_.y, coef1);
    k_.y = rotary_embedding_transform(k_.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint32_t& q, int tid, int rot_embed_dim, int t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
    q = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(uint32_t& q, uint32_t& k, int tid, int rot_embed_dim, int t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
    q = rotary_embedding_transform(q, coef);
    k = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(uint2& q, int tid, int rot_embed_dim, int t_step)
{
    if (4 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
    q.x = rotary_embedding_transform(q.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
    q.y = rotary_embedding_transform(q.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint2& q, uint2& k, int tid, int rot_embed_dim, int t_step)
{
    if (4 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
    q.x = rotary_embedding_transform(q.x, coef0);
    k.x = rotary_embedding_transform(k.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
    q.y = rotary_embedding_transform(q.y, coef1);
    k.y = rotary_embedding_transform(k.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint4& q, int tid, int rot_embed_dim, int t_step)
{
    if (8 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
    q.x = rotary_embedding_transform(q.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
    q.y = rotary_embedding_transform(q.y, coef1);
    const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
    q.z = rotary_embedding_transform(q.z, coef2);
    const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
    q.w = rotary_embedding_transform(q.w, coef3);
}

inline __device__ void apply_rotary_embedding(uint4& q, uint4& k, int tid, int rot_embed_dim, int t_step)
{
    if (8 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
    q.x = rotary_embedding_transform(q.x, coef0);
    k.x = rotary_embedding_transform(k.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
    q.y = rotary_embedding_transform(q.y, coef1);
    k.y = rotary_embedding_transform(k.y, coef1);
    const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
    q.z = rotary_embedding_transform(q.z, coef2);
    k.z = rotary_embedding_transform(k.z, coef2);
    const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
    q.w = rotary_embedding_transform(q.w, coef3);
    k.w = rotary_embedding_transform(k.w, coef3);
}

#ifdef ENABLE_BF16
inline __device__ void apply_rotary_embedding(__nv_bfloat162& q, int tid, int rot_embed_dim, int t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
    q = rotary_embedding_transform(q, coef);
}

inline __device__ void
apply_rotary_embedding(__nv_bfloat162& q, __nv_bfloat162& k, int tid, int rot_embed_dim, int t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
    q = rotary_embedding_transform(q, coef);
    k = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(bf16_4_t& q, int tid, int rot_embed_dim, int t_step)
{
    if (4 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
    q.x = rotary_embedding_transform(q.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
    q.y = rotary_embedding_transform(q.y, coef1);
}

inline __device__ void apply_rotary_embedding(bf16_4_t& q, bf16_4_t& k, int tid, int rot_embed_dim, int t_step)
{
    if (4 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
    q.x = rotary_embedding_transform(q.x, coef0);
    k.x = rotary_embedding_transform(k.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
    q.y = rotary_embedding_transform(q.y, coef1);
    k.y = rotary_embedding_transform(k.y, coef1);
}

inline __device__ void apply_rotary_embedding(bf16_8_t& q, int tid, int rot_embed_dim, int t_step)
{
    if (8 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
    q.x = rotary_embedding_transform(q.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
    q.y = rotary_embedding_transform(q.y, coef1);
    const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
    q.z = rotary_embedding_transform(q.z, coef2);
    const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
    q.w = rotary_embedding_transform(q.w, coef3);
}

inline __device__ void apply_rotary_embedding(bf16_8_t& q, bf16_8_t& k, int tid, int rot_embed_dim, int t_step)
{
    if (8 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
    q.x = rotary_embedding_transform(q.x, coef0);
    k.x = rotary_embedding_transform(k.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
    q.y = rotary_embedding_transform(q.y, coef1);
    k.y = rotary_embedding_transform(k.y, coef1);
    const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
    q.z = rotary_embedding_transform(q.z, coef2);
    k.z = rotary_embedding_transform(k.z, coef2);
    const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
    q.w = rotary_embedding_transform(q.w, coef3);
    k.w = rotary_embedding_transform(k.w, coef3);
}
#endif  // ENABLE_BF16

}  // namespace mmha