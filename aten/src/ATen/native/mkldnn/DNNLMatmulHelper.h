#ifndef DNNL_HELPER_HPP
#define DNNL_HELPER_HPP

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include "oneapi/dnnl/dnnl.hpp"

namespace {
template <typename T>
struct DNNLType {
  static constexpr dnnl::memory::data_type type =
      dnnl::memory::data_type::undef;
};

template <>
struct DNNLType<int8_t> {
  static constexpr dnnl::memory::data_type type = dnnl::memory::data_type::s8;
};

template <>
struct DNNLType<int32_t> {
  static constexpr dnnl::memory::data_type type = dnnl::memory::data_type::s32;
};

template <>
struct DNNLType<float> {
  static constexpr dnnl::memory::data_type type = dnnl::memory::data_type::f32;
};

template <>
struct DNNLType<c10::BFloat16> {
  static constexpr dnnl::memory::data_type type = dnnl::memory::data_type::bf16;
};

template <>
struct DNNLType<c10::Half> {
  static constexpr dnnl::memory::data_type type = dnnl::memory::data_type::f16;
};

template <typename T>
constexpr inline dnnl::memory::data_type get_dnnl_type() {
  return DNNLType<std::decay_t<T>>::type;
}
};  // namespace

class DNNLPrimitiveHelper {
 public:
  // A: [M, K], row-major/column-major (defined by a_stride_0 and a_stride_1)
  // B: [K, N], row-major/column-major (defined by b_stride_0 and b_stride_1)
  // C: [M, N], row-major
  template <typename OutputT>
  static void gemm_s8s8_dnnl(const int8_t* a, const int8_t* b, OutputT* c,
                            dnnl_dim_t M, dnnl_dim_t N,
                            dnnl_dim_t K, dnnl_dim_t a_stride_0, dnnl_dim_t a_stride_1, dnnl_dim_t b_stride_0, dnnl_dim_t b_stride_1) {
    auto&& OutputType = get_dnnl_type<OutputT>();
    dnnl::memory::desc a_md({M, K}, dnnl::memory::data_type::s8, {a_stride_0, a_stride_1});

    dnnl::memory::desc b_md({K, N}, dnnl::memory::data_type::s8, {b_stride_0, b_stride_1});
    dnnl::memory::desc c_md({M, N}, OutputType, {N, 1});

    dnnl::matmul::primitive_desc matmul_pd;
    auto mat_src_md = dnnl::memory::desc({M, K}, dnnl::memory::data_type::s8,
                                         dnnl::memory::format_tag::any);
    auto mat_weights_md = dnnl::memory::desc(
        {K, N}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::any);
    auto mat_dst_md =
        dnnl::memory::desc({M, N}, OutputType, dnnl::memory::format_tag::any);

    matmul_pd = dnnl::matmul::primitive_desc(
          default_engine(), mat_src_md, mat_weights_md, mat_dst_md);
    dnnl::matmul matmul(matmul_pd);

    auto& engine = default_engine();

    dnnl::memory a_m(a_md, engine, (void*)a);
    dnnl::memory b_m(b_md, engine, (void*)b);
    dnnl::memory c_m(c_md, engine, (void*)c);

    auto& stream = default_stream();

    auto mat_src_mem = a_m;
    auto mat_weights_mem = b_m;
    auto mat_dst_mem = c_m;

    if (matmul_pd.src_desc() != a_m.get_desc()) {
      mat_src_mem = dnnl::memory(matmul_pd.src_desc(), engine);
      dnnl::reorder(a_m, mat_src_mem).execute(stream, a_m, mat_src_mem);
    }
    
    if (matmul_pd.weights_desc() != b_m.get_desc()) {
      mat_weights_mem = dnnl::memory(matmul_pd.weights_desc(), engine);
      dnnl::reorder(b_m, mat_weights_mem).execute(stream, b_m, mat_weights_mem);
    }

    matmul.execute(
                stream, {
                            {DNNL_ARG_SRC, mat_src_mem},
                            {DNNL_ARG_WEIGHTS, mat_weights_mem},
                            {DNNL_ARG_DST, mat_dst_mem},
                        });
        stream.wait();
    }

 private:
  static dnnl::engine& default_engine() {
    static dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    return engine;
  }

  static dnnl::stream& default_stream() {
    static dnnl::stream stream(default_engine());
    return stream;
  }
};

#endif