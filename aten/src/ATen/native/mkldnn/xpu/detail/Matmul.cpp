
#include <c10/xpu/XPUFunctions.h>

#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <Attr.h>
#include <Utils.h>

#include <c10/core/ScalarType.h>
#include <oneapi/dnnl/dnnl.hpp>

namespace at::native::onednn {
class BiasHandler {
 public:
  BiasHandler(at::Tensor& bias, int64_t mb, int64_t m, int64_t k, int64_t n)
      : b(bias), mb(mb), m(m), k(k), n(n), with_bias(bias.defined()) {}

  void handle0D() {
    TORCH_CHECK(
        b.numel() == 1, "matmul supports 1 numel when bias dim is [] ...");
    if (gemm_dim == 3) {
      b = b.expand({mb, m, n}).contiguous();
    } else {
      b = b.expand({1, n}).contiguous();
    }
  }

  void handle1D() {
    TORCH_CHECK(
        b.size(0) == n || b.size(0) == 1,
        "matmul supports [n] or [1] when bias dim is 1...");
    if (b.size(0) == 0) {
      with_bias = false;
    } else if (gemm_dim == 3) {
      b = b.expand({mb, m, n}).contiguous();
    } else if (gemm_dim == 2) {
      b = b.expand({1, n}).contiguous();
    }
  }

  void handle2D() {
    TORCH_CHECK(
        (b.size(0) == m && b.size(1) == n) ||
            (b.size(0) == 1 && b.size(1) == n) ||
            (b.size(0) == m && b.size(1) == 1) ||
            (b.size(0) == 1 && b.size(1) == 1),
        "matmul supports [m, n] or [1, n] or [m, 1] or [1, 1] when bias dim is 2 ...");
    if (b.size(0) == 1 && b.size(1) == 1)
      b = b.expand({1, n}).contiguous();
  }

  void handle3D() {
    TORCH_CHECK(
        at::are_expandable({mb, m, n}, b.sizes()),
        "matmul bias must be expandable to:",
        "{mb, m, n} where mb is the batch size, m is the number of rows, and n is the number of columns.",
        " but got:",
        b.sizes());
    b = b.expand({mb, m, n}).contiguous();
  }

  bool is_with_bias() {
    return with_bias;
  }

  void handle() {
    with_bias = b.defined();
    if (!with_bias)
      return;
    using HandlerFn = void (BiasHandler::*)();
    std::unordered_map<int, HandlerFn> handler_map = {
        {0, &BiasHandler::handle0D},
        {1, &BiasHandler::handle1D},
        {2, &BiasHandler::handle2D},
        {3, &BiasHandler::handle3D}};

    auto iter = handler_map.find(b.dim());
    TORCH_CHECK(iter != handler_map.end(), "invalid bias dim:", b.dim());
    (this->*(iter->second))();
  }

 private:
  at::Tensor& b;
  int mb, m, k, n, gemm_dim;
  bool with_bias;
};
class GEMMMemoryCreator {
 public:
  GEMMMemoryCreator(
      int64_t dim,
      int64_t m,
      int64_t k,
      int64_t n,
      int64_t mb,
      int64_t bsA,
      int64_t bsB,
      bool m2_trans,
      bool with_bias)
      : ndim(dim),
        m(m),
        k(k),
        n{n},
        mb{mb},
        bsA(bsA),
        bsB(bsB),
        m2_trans(m2_trans),
        with_bias(with_bias) {}

  using arg_map_t = std::unordered_map<int, dnnl::memory>;
  using md_t = dnnl::memory::desc;

  void handle2D(
      const at::Tensor& m1,
      const at::Tensor& m2,
      const at::Tensor& dst) {
    m1_dims = {m, k};
    m2_dims = {k, n};
    dst_dims = {m, n};

    m1_strides = {m1.stride(0), m1.stride(1)};

    m2_strides = m2_trans ? std::vector<int64_t>{m2.stride(0), m2.stride(1)}
                          : std::vector<int64_t>{m2.stride(1), m2.stride(0)};

    dst_strides = {dst.stride(0), dst.stride(1)};
  }

  void handle3D(
      const at::Tensor& m1,
      const at::Tensor& m2,
      const at::Tensor& dst) {
    m1_dims = dnnl::memory::dims({bsA, m, k});
    m2_dims = dnnl::memory::dims({bsB, k, n});
    dst_dims = dnnl::memory::dims({mb, m, n});

    m1_strides = {m1.stride(0), m1.stride(1), m1.stride(2)};
    m2_strides = m2_trans
        ? std::vector<int64_t>{m2.stride(0), m2.stride(1), m2.stride(2)}
        : std::vector<int64_t>{m2.stride(0), m2.stride(2), m2.stride(1)};

    dst_strides = {dst.stride(0), dst.stride(1), dst.stride(2)};
  }

  void initialize_md(
      const at::Tensor& m1,
      const at::Tensor& m2,
      const at::Tensor& dst) {
    m1_md = create_md(m1, m1_dims, m1_strides);
    m2_md = create_md(m2, m2_dims, m2_strides);
    dst_md = create_md(dst, dst_dims, dst_strides);
  }

  std::tuple<md_t, md_t, md_t, md_t> query_md() {
    return {m1_md, m2_md, dst_md, bias_md};
  }

  void initialize(
      const at::Tensor& m1,
      const at::Tensor& m2,
      const at::Tensor& dst,
      const at::Tensor& bias) {
    if (ndim == 2) {
      handle2D(m1, m2, dst);
    } else if (ndim == 3) {
      handle3D(m1, m2, dst);
    } else {
      TORCH_CHECK(false, "only support 2D or 3D matmul, got ndim:", ndim);
    }
    initialize_md(m1, m2, dst);

    if (with_bias) {
      bias_dims = get_onednn_dims(bias);
      bias_strides = get_onednn_strides(bias);
      bias_md = dnnl::memory::desc(
          bias_dims, get_onednn_dtype_include_double(bias), bias_strides);
    } else {
      bias_md = nullptr;
    }
  }

  arg_map_t create_memory(
      const at::Tensor& m1,
      const at::Tensor& m2,
      const at::Tensor& dst,
      const at::Tensor& bias) {
    std::unordered_map<int, dnnl::memory> args;
    auto& eng = GpuEngineManager::Instance().get_engine();
    args.insert({DNNL_ARG_SRC, dnnl::memory(m1_md, eng, m1.data_ptr())});
    args.insert({DNNL_ARG_WEIGHTS, dnnl::memory(m2_md, eng, m2.data_ptr())});
    args.insert({DNNL_ARG_DST, dnnl::memory(dst_md, eng, dst.data_ptr())});
    if (with_bias) {
      auto bias_m = make_onednn_memory(bias_md, eng, bias.data_ptr());
      args.insert({DNNL_ARG_BIAS, bias_m});
    }
    return args;
  }

 private:
  dnnl::memory::desc create_md(
      const at::Tensor& t,
      dnnl::memory::dims dims,
      dnnl::memory::dims strides) {
    return dnnl::memory::desc(
        dims, get_onednn_dtype_include_double(t), strides);
  }

  dnnl::memory::dims m1_dims, m2_dims, dst_dims;
  dnnl::memory::dims m1_strides, m2_strides, dst_strides;
  dnnl::memory::dims bias_dims, bias_strides;
  dnnl::memory::desc m1_md, m2_md, dst_md, bias_md;
  int64_t ndim, m, k, n, mb, bsA, bsB;
  bool m2_trans, with_bias;
};

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

  auto matmul_stride_handler = [](const at::Tensor& t) -> at::Tensor {
    return is_onednn_matmul_strides(t) ? t : t.contiguous();
  };
  m1 = matmul_stride_handler(m1);
  m2 = matmul_stride_handler(m2);
  at::Tensor dst = matmul_stride_handler(result);

  int64_t m = dst.size(-2);
  int64_t n = dst.size(-1);
  int64_t k = m1.size(-1);
  int64_t mb = dims == 3 ? dst.size(0) : 1;
  if (dims == 3) {
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
  // Step1: handle shape, and create mds
  at::Tensor b = b_raw;
  BiasHandler bias_handler(b, mb, m, k, n);
  bias_handler.handle();
  bool with_bias = bias_handler.is_with_bias();
  b = b.contiguous();
  GEMMMemoryCreator memory_creator(
      dims, m, k, n, mb, m1.size(0), m2.size(0), m2_trans, with_bias);
  memory_creator.initialize(m1, m2, dst, b);
  auto [m1_md, m2_md, dst_md, bias_md] = memory_creator.query_md();

  dnnl::post_ops po = attr.extract_post_ops(dst);
  dnnl::primitive_attr pattr;
  pattr.set_post_ops(po);

  // Step2: primitive attributes
  dnnl::memory::data_type m1_dt = get_onednn_dtype_include_double(m1);
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  if (m1_dt == dnnl::memory::data_type::f32) {
    bool allow_tf32 = at::globalContext().allowTF32OneDNN();
    if (allow_tf32) {
      pattr.set_fpmath_mode(dnnl::fpmath_mode::tf32);
    } else {
      pattr.set_fpmath_mode(dnnl::fpmath_mode::strict);
    }
  }

#if ONEDNN_SUPPORT_DETERMINISTIC
  if (at::globalContext().deterministicAlgorithms() ||
      at::globalContext().deterministicMkldnn())
    pattr.set_deterministic(true);
#endif

  // Step3: pd, primitive creation
  dnnl::matmul matmul_p;
  dnnl::matmul::primitive_desc matmul_pd;
  if (with_bias) {
    matmul_pd = dnnl::matmul::primitive_desc(
        engine, m1_md, m2_md, bias_md, dst_md, pattr);
  } else {
    matmul_pd =
        dnnl::matmul::primitive_desc(engine, m1_md, m2_md, dst_md, pattr);
  }
  matmul_p = dnnl::matmul(matmul_pd);

  // Step4: create memory args and execution
  std::unordered_map<int, dnnl::memory> args =
      memory_creator.create_memory(m1, m2, dst, b);
  if (attr.with_binary())
    attr.construct_post_binary(matmul_pd, args);

  size_t scratchpad_size = matmul_pd.scratchpad_desc().get_size();
  at::Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)},
      m1.options().dtype(at::kByte),
      std::nullopt);
  auto scratchpad_memory = make_onednn_memory(
      matmul_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});

  sycl::event matmul_event =
      dnnl::sycl_interop::execute(matmul_p, stream, args, deps);

  if (!dst.is_same(result))
    result.copy_(dst);

  return matmul_event;
}

} // namespace at::native::onednn
