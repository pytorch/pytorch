#pragma once

#include <ATen/ATen.h>

#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>

#include <oneapi/dnnl/dnnl.hpp>

namespace at::native::onednn {

class BiasHandler {
  // This class handles the bias tensor for matmul operations.
  // It checks the dimensions of the bias tensor and expands it if necessary
  // for broadcasting.
 public:
  BiasHandler() = delete;
  BiasHandler(
      at::Tensor& bias,
      int64_t mb,
      int64_t m,
      int64_t k,
      int64_t n,
      int64_t gemm_dim)
      : b(bias),
        mb(mb),
        m(m),
        k(k),
        n(n),
        gemm_dim(gemm_dim),
        with_bias(bias.defined()) {}

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
  // This class creates memory descriptors and memory objects for GEMM
  // operations.
 public:
  GEMMMemoryCreator() = delete;
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
      bias_md = dnnl::memory::desc();
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

} // namespace at::native::onednn
