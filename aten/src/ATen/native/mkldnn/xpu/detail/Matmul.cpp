#include <c10/xpu/XPUFunctions.h>

#include <ATen/ATen.h>

#include <Attr.h>
#include <ATen/native/mkldnn/xpu/detail/DnnlExt.h>
#include <Utils.h>

#include <c10/core/ScalarType.h>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <unordered_map>
#include <utility>

namespace at::native::onednn {

namespace {

using dim_t = dnnl::memory::dim;

constexpr int64_t kFpMatmulPrimitiveCacheKeyVersion = 1;
constexpr size_t kFpMatmulPrimitiveCacheCapacity = 512;

struct FpMatmulCachedPrimitive {
  dnnl::matmul::primitive_desc pd;
  primitive_ext prim;

  explicit FpMatmulCachedPrimitive(dnnl::matmul::primitive_desc&& p)
      : pd(std::move(p)), prim(dnnl::matmul(pd)) {}
};

using FpMatmulCacheValue = std::shared_ptr<FpMatmulCachedPrimitive>;
using FpMatmulLruCache = lru_cache<dnnl::memory::dims, FpMatmulCacheValue>;

FpMatmulLruCache& get_fpmatmul_primitive_cache(int device_id) {
  static thread_local std::unordered_map<int, FpMatmulLruCache> caches;
  auto [it, _] = caches.try_emplace(device_id, FpMatmulLruCache());
  auto& c = it->second;
  if (c.max_size() == 0) {
    c.resize(kFpMatmulPrimitiveCacheCapacity);
  }
  return c;
}

// build_* and make_* must keep identical parameter lists; lookup forwards the
// same pack to both so arity/type mismatches fail at compile time.
dnnl::memory::dims build_fpmatmul_primitive_cache_key(
    int device_id,
    int64_t rank,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t mb,
    [[maybe_unused]] const dnnl::memory::dims& m1_dims,
    [[maybe_unused]] const dnnl::memory::dims& m2_dims,
    [[maybe_unused]] const dnnl::memory::dims& dst_dims,
    const dnnl::memory::dims& m1_strides,
    const dnnl::memory::dims& m2_strides,
    const dnnl::memory::dims& dst_strides,
    dnnl::memory::data_type m1_usr_dt,
    dnnl::memory::data_type m2_usr_dt,
    dnnl::memory::data_type dst_usr_dt,
    dnnl::memory::data_type m1_dt,
    dnnl::memory::data_type m2_dt,
    dnnl::memory::data_type dst_dt,
    bool m2_trans,
    bool with_bias,
    const dnnl::memory::dims& bias_dims,
    dnnl::memory::data_type bias_dt,
    const dnnl::memory::dims& bias_strides,
    bool deterministic,
    bool allow_tf32_oneapi,
    Attr& attr,
    dnnl::engine&,
    const at::Tensor&) {
  dnnl::memory::dims key;
  auto append_dims = [&](const dnnl::memory::dims& d) {
    key.insert(key.end(), d.begin(), d.end());
  };

  key.push_back(kFpMatmulPrimitiveCacheKeyVersion);
  key.push_back(static_cast<dim_t>(device_id));
  key.push_back(rank);
  key.push_back(m);
  key.push_back(n);
  key.push_back(k);
  key.push_back(mb);
  append_dims(m1_strides);
  append_dims(m2_strides);
  append_dims(dst_strides);
  key.push_back(static_cast<dim_t>(static_cast<int>(m1_usr_dt)));
  key.push_back(static_cast<dim_t>(static_cast<int>(m2_usr_dt)));
  key.push_back(static_cast<dim_t>(static_cast<int>(dst_usr_dt)));
  key.push_back(static_cast<dim_t>(static_cast<int>(m1_dt)));
  key.push_back(static_cast<dim_t>(static_cast<int>(m2_dt)));
  key.push_back(static_cast<dim_t>(static_cast<int>(dst_dt)));
  key.push_back(m2_trans ? 1 : 0);
  key.push_back(with_bias ? 1 : 0);
  if (with_bias) {
    append_dims(bias_dims);
    key.push_back(static_cast<dim_t>(static_cast<int>(bias_dt)));
    append_dims(bias_strides);
  }
  key.push_back(deterministic ? 1 : 0);
  key.push_back(allow_tf32_oneapi ? 1 : 0);
  append_dims(attr.matmul_primitive_cache_key_extra());
  return key;
}

FpMatmulCacheValue make_fpmatmul_cached_primitive(
    [[maybe_unused]] int device_id,
    [[maybe_unused]] int64_t rank,
    [[maybe_unused]] int64_t m,
    [[maybe_unused]] int64_t n,
    [[maybe_unused]] int64_t k,
    [[maybe_unused]] int64_t mb,
    const dnnl::memory::dims& m1_dims,
    const dnnl::memory::dims& m2_dims,
    const dnnl::memory::dims& dst_dims,
    const dnnl::memory::dims& m1_strides,
    const dnnl::memory::dims& m2_strides,
    const dnnl::memory::dims& dst_strides,
    [[maybe_unused]] dnnl::memory::data_type m1_usr_dt,
    [[maybe_unused]] dnnl::memory::data_type m2_usr_dt,
    [[maybe_unused]] dnnl::memory::data_type dst_usr_dt,
    dnnl::memory::data_type m1_dt,
    dnnl::memory::data_type m2_dt,
    dnnl::memory::data_type dst_dt,
    [[maybe_unused]] bool m2_trans,
    bool with_bias,
    const dnnl::memory::dims& bias_dims,
    dnnl::memory::data_type bias_dt,
    const dnnl::memory::dims& bias_strides,
    bool deterministic,
    bool allow_tf32_oneapi,
    Attr& attr,
    dnnl::engine& engine,
    const at::Tensor& dst) {
  dnnl::post_ops po = attr.extract_post_ops(dst);

  // STEP1: create memory desc
  dnnl::memory::desc m1_md(m1_dims, m1_dt, m1_strides);
  dnnl::memory::desc m2_md(m2_dims, m2_dt, m2_strides);
  dnnl::memory::desc dst_md(dst_dims, dst_dt, dst_strides);

  // STEP2: creat attribute
  dnnl::primitive_attr pattr;
  pattr.set_post_ops(po);

#if ONEDNN_SUPPORT_DETERMINISTIC
  if (deterministic) {
    pattr.set_deterministic(true);
  }
#endif

  // scratchpad
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  if (m1_dt == dnnl::memory::data_type::f32) {
    if (allow_tf32_oneapi) {
      pattr.set_fpmath_mode(dnnl::fpmath_mode::tf32);
    } else {
      pattr.set_fpmath_mode(dnnl::fpmath_mode::strict);
    }
  }

  // STEP3: create primitive
  dnnl::matmul::primitive_desc matmul_pd =
      with_bias ? dnnl::matmul::primitive_desc(
                        engine,
                        m1_md,
                        m2_md,
                        dnnl::memory::desc(
                            bias_dims, bias_dt, bias_strides),
                        dst_md,
                        pattr)
                  : dnnl::matmul::primitive_desc(
                        engine, m1_md, m2_md, dst_md, pattr);

  return std::make_shared<FpMatmulCachedPrimitive>(std::move(matmul_pd));
}

template <typename... Args>
FpMatmulCacheValue lookup_or_build_fpmatmul_cached_primitive(
    FpMatmulLruCache& primitive_cache,
    Args&&... args) {
  dnnl::memory::dims cache_key = build_fpmatmul_primitive_cache_key(
      std::forward<Args>(args)...);
  auto cache_it = primitive_cache.find(cache_key);
  if (cache_it != primitive_cache.end()) {
    return cache_it->second;
  }
  FpMatmulCacheValue entry =
      make_fpmatmul_cached_primitive(std::forward<Args>(args)...);
  primitive_cache.insert({std::move(cache_key), entry});
  return entry;
}

} // namespace

sycl::event matmul(
    at::Tensor& result,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Tensor& b_raw,
    bool m2_trans,
    Attr attr,
    const std::vector<sycl::event>& deps) {
  // m2_trans means mat2 is transposed from the nn.Linear perspective.
  // m2_trans==true means mat2 is [k, n] layout.
  // m2_trans==false means mat2 is [n, k] layout, aka, the default layout in
  // nn.Linear.
  int64_t dims = result.dim();
  TORCH_CHECK(
      dims == 2 || dims == 3,
      "oneDNN matmul only works with 2D or 3D, got ",
      dims);
  TORCH_CHECK(
      dims == mat1.dim() && dims == mat2.dim(),
      "oneDNN input matrixes must have the same ranks");
  TORCH_CHECK(result.defined(), "oneDNN matmul result should be defined");

  auto& engine = GpuEngineManager::Instance().get_engine();
  auto& stream = GpuStreamManager::Instance().get_stream();

  at::Tensor m1 = mat1;
  at::Tensor m2 = mat2;

  undo_broadcast_on_batch(m1, m2);

  m1 = is_onednn_matmul_strides(m1) ? m1 : m1.contiguous();
  m2 = is_onednn_matmul_strides(m2) ? m2 : m2.contiguous();
  at::Tensor dst =
      is_onednn_matmul_strides(result) ? result : result.contiguous();

  int64_t m = dst.size(-2);
  int64_t n = dst.size(-1);
  int64_t k = m1.size(-1);
  int64_t mb = 1;

  if (dims == 3) {
    mb = dst.size(0);
    TORCH_CHECK(
        mb == mat1.size(0) && mb == mat2.size(0),
        "batch size mismatch, dst mb: ",
        mb,
        "m1 mb",
        mat1.size(0),
        " m2 mb: ",
        mat2.size(0));
  }

  // validate bias and make it compatible with oneDNN implementation
  bool with_bias = false;
  at::Tensor b = b_raw;
  if (b.defined()) {
    with_bias = true;
    if (b.dim() == 1) {
      TORCH_CHECK(
          b.size(0) == n || b.size(0) == 1,
          "matmul supports [n] or [1] when bias dim is 1 ...");
      if (b.size(0) == 0) {
        with_bias = false;
      } else if (m1.dim() == 3) {
        b = b.expand({mb, m, n}).contiguous();
      } else if (m1.dim() == 2) {
        b = b.expand({1, n}).contiguous();
      }
    } else if (b.dim() == 2) {
      TORCH_CHECK(
          (b.size(0) == m && b.size(1) == n) ||
              (b.size(0) == 1 && b.size(1) == n) ||
              (b.size(0) == m && b.size(1) == 1) ||
              (b.size(0) == 1 && b.size(1) == 1),
          "matmul supports [m, n] or [1, n] or [m, 1] or [1, 1] when bias dim is 2 ...");
      if (b.size(0) == 1 && b.size(1) == 1)
        b = b.expand({1, n}).contiguous();
    } else if (b.dim() == 3) {
      TORCH_CHECK(
          at::are_expandable({mb, m, n}, b.sizes()),
          "matmul bias must be expandable to:",
          dst.sizes(),
          " but got:",
          b.sizes());
      b = b.expand({mb, m, n}).contiguous();
    } else if (b.dim() == 0) {
      TORCH_CHECK(
          b.numel() == 1, "matmul supports 1 numel when bias dim is [] ...");
      if (m1.dim() == 3) {
        b = b.expand({mb, m, n}).contiguous();
      } else {
        b = b.expand({1, n}).contiguous();
      }
    } else {
      TORCH_CHECK(0, "unsupported bias dim in matmul ...");
    }
  }

  b = b.contiguous(); // avoid reorder 2 times

  // xpu matmul support both ab/ba shape for m2 tensor, we don't check any more
  auto m1_usr_dt = get_onednn_dtype_include_double(m1);
  auto m2_usr_dt = get_onednn_dtype_include_double(m2);
  auto dst_usr_dt = get_onednn_dtype_include_double(dst);

  auto m1_dt = m1_usr_dt;
  auto m2_dt = m2_usr_dt;
  auto dst_dt = dst_usr_dt;
  dnnl::memory::data_type bias_dt = dnnl::memory::data_type::undef;

  // Naive Master weight
  if (m1_dt == dnnl::memory::data_type::bf16 &&
      m2_dt == dnnl::memory::data_type::f32) {
    m2_dt = dnnl::memory::data_type::bf16;
    dst_dt = dnnl::memory::data_type::bf16;
  } else if (
      m1_dt == dnnl::memory::data_type::f32 &&
      m2_dt == dnnl::memory::data_type::bf16) {
    m1_dt = dnnl::memory::data_type::bf16;
    dst_dt = dnnl::memory::data_type::bf16;
  }

  dnnl::memory::dims m1_dims, m2_dims, dst_dims, bias_dims;
  dnnl::memory::dims m1_strides, m2_strides, dst_strides, bias_strides;
  if (dims == 2) {
    m1_dims = {m, k};
    m2_dims = {k, n};
    dst_dims = {m, n};

    m1_strides = {m1.stride(0), m1.stride(1)};
    if (m2_trans) {
      m2_strides = {m2.stride(0), m2.stride(1)};
    } else {
      m2_strides = {m2.stride(1), m2.stride(0)};
    }
    dst_strides = {dst.stride(0), dst.stride(1)};
  } else {
    m1_dims = {m1.size(0), m, k};
    m2_dims = {m2.size(0), k, n};
    dst_dims = {mb, m, n};

    m1_strides = {m1.stride(0), m1.stride(1), m1.stride(2)};
    if (m2_trans) {
      m2_strides = {m2.stride(0), m2.stride(1), m2.stride(2)};
    } else {
      m2_strides = {m2.stride(0), m2.stride(2), m2.stride(1)};
    }
    dst_strides = {dst.stride(0), dst.stride(1), dst.stride(2)};
  }

  if (with_bias) {
    bias_dims = get_onednn_dims(b);
    bias_dt = get_onednn_dtype_include_double(b);
    bias_strides = get_onednn_strides(b);
  }

  bool deterministic = false;
#if ONEDNN_SUPPORT_DETERMINISTIC
  deterministic = at::globalContext().deterministicAlgorithms() ||
      at::globalContext().deterministicMkldnn();
#endif
  const bool allow_tf32_oneapi =
      m1_dt == dnnl::memory::data_type::f32 &&
      at::globalContext().allowTF32OneDNN();

  const int device_id = c10::xpu::current_device();
  auto& primitive_cache = get_fpmatmul_primitive_cache(device_id);
  FpMatmulCacheValue entry = lookup_or_build_fpmatmul_cached_primitive(
      primitive_cache,
      device_id,
      dims,
      m,
      n,
      k,
      mb,
      m1_dims,
      m2_dims,
      dst_dims,
      m1_strides,
      m2_strides,
      dst_strides,
      m1_usr_dt,
      m2_usr_dt,
      dst_usr_dt,
      m1_dt,
      m2_dt,
      dst_dt,
      m2_trans,
      with_bias,
      bias_dims,
      bias_dt,
      bias_strides,
      deterministic,
      allow_tf32_oneapi,
      attr,
      engine,
      dst);

  // STEP4: create memory
  dnnl::memory::desc m1_usr_md(m1_dims, m1_usr_dt, m1_strides);
  dnnl::memory::desc m2_usr_md(m2_dims, m2_usr_dt, m2_strides);
  dnnl::memory::desc dst_usr_md(dst_dims, dst_usr_dt, dst_strides);

  auto m1_usr_m = make_onednn_memory(m1_usr_md, engine, m1.data_ptr());
  auto m2_usr_m = make_onednn_memory(m2_usr_md, engine, m2.data_ptr());
  auto dst_usr_m = make_onednn_memory(dst_usr_md, engine, dst.data_ptr());

  dnnl::memory m1_m = m1_usr_m, m2_m = m2_usr_m, dst_m = dst_usr_m;

  std::unordered_map<int, dnnl::memory> args;
  if (attr.with_binary()) {
    attr.construct_post_binary(entry->pd, args);
  }

  const size_t scratchpad_size = entry->pd.scratchpad_desc().get_size();
  at::Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)},
      m1.options().dtype(at::kByte),
      std::nullopt);
  auto scratchpad_memory = make_onednn_memory(
      entry->pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});

  args.insert({DNNL_ARG_SRC, m1_m});
  args.insert({DNNL_ARG_WEIGHTS, m2_m});
  args.insert({DNNL_ARG_DST, dst_m});
  if (with_bias) {
    dnnl::memory::desc bias_md(bias_dims, bias_dt, bias_strides);
    auto bias_m = make_onednn_memory(bias_md, engine, b.data_ptr());
    args.insert({DNNL_ARG_BIAS, bias_m});
  }

  sycl::event matmul_event =
      dnnl::sycl_interop::execute(entry->prim, stream, args, deps);

  if (!dst.is_same(result))
    result.copy_(dst);

  return matmul_event;
}

} // namespace at::native::onednn
