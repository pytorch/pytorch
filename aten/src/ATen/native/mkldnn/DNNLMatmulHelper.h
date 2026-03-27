#ifndef DNNL_HELPER_HPP
#define DNNL_HELPER_HPP

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <oneapi/dnnl/dnnl.hpp>

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
    static void gemm_s8s8_dnnl(
        const int8_t* a,
        const int8_t* b,
        OutputT* c,
        dnnl_dim_t M,
        dnnl_dim_t N,
        dnnl_dim_t K,
        dnnl_dim_t a_stride_0,
        dnnl_dim_t a_stride_1,
        dnnl_dim_t b_stride_0,
        dnnl_dim_t b_stride_1) {

      auto&& OutputType = get_dnnl_type<OutputT>();
      auto& engine = default_engine();
      auto& stream = default_stream();

      // ---- fast-path detection ----
      bool is_a_row_major = (a_stride_0 == K && a_stride_1 == 1);
      bool is_b_col_major = (b_stride_0 == 1 && b_stride_1 == K);
      bool use_fast_path = is_a_row_major && is_b_col_major;

      // ---- Primitive cache key ----
      struct PrimitiveKey {
        dnnl_dim_t M, N, K;
        dnnl::memory::data_type out_dt;

        bool operator==(const PrimitiveKey& o) const {
          return M == o.M && N == o.N && K == o.K &&
                out_dt == o.out_dt;
        }
      };

      struct PrimitiveHash {
        size_t operator()(const PrimitiveKey& k) const {
          size_t h = std::hash<int64_t>()(k.M);
          h ^= std::hash<int64_t>()(k.N) << 1;
          h ^= std::hash<int64_t>()(k.K) << 2;
          return h;
        }
      };

      struct PrimitiveValue {
        dnnl::matmul primitive;
        dnnl::matmul::primitive_desc pd;
      };

      static std::unordered_map<PrimitiveKey, PrimitiveValue, PrimitiveHash> primitive_cache;
      static std::mutex primitive_mutex;

      dnnl::matmul matmul;
      dnnl::matmul::primitive_desc matmul_pd;

      if (use_fast_path) {
        PrimitiveKey pkey{M, N, K, OutputType};

        std::lock_guard<std::mutex> guard(primitive_mutex);

        auto it = primitive_cache.find(pkey);
        if (it == primitive_cache.end()) {
          auto mat_src_md = dnnl::memory::desc(
              {M, K}, dnnl::memory::data_type::s8,
              dnnl::memory::format_tag::any);

          auto mat_weights_md = dnnl::memory::desc(
              {K, N}, dnnl::memory::data_type::s8,
              dnnl::memory::format_tag::any);

          auto mat_dst_md = dnnl::memory::desc(
              {M, N}, OutputType,
              dnnl::memory::format_tag::any);

          auto pd = dnnl::matmul::primitive_desc(
              engine, mat_src_md, mat_weights_md, mat_dst_md);

          auto prim = dnnl::matmul(pd);

          it = primitive_cache.emplace(pkey, PrimitiveValue{prim, pd}).first;
        }

        matmul = it->second.primitive;
        matmul_pd = it->second.pd;

      } else {
        // fallback: no primitive cache
        auto mat_src_md = dnnl::memory::desc(
            {M, K}, dnnl::memory::data_type::s8,
            dnnl::memory::format_tag::any);

        auto mat_weights_md = dnnl::memory::desc(
            {K, N}, dnnl::memory::data_type::s8,
            dnnl::memory::format_tag::any);

        auto mat_dst_md = dnnl::memory::desc(
            {M, N}, OutputType,
            dnnl::memory::format_tag::any);

        matmul_pd = dnnl::matmul::primitive_desc(
            engine, mat_src_md, mat_weights_md, mat_dst_md);

        matmul = dnnl::matmul(matmul_pd);
      }

      // ---- user memory ----
      dnnl::memory::desc a_md({M, K}, dnnl::memory::data_type::s8,
                              {a_stride_0, a_stride_1});
      dnnl::memory::desc b_md({K, N}, dnnl::memory::data_type::s8,
                              {b_stride_0, b_stride_1});
      dnnl::memory::desc c_md({M, N}, OutputType, {N, 1});

      dnnl::memory a_m(a_md, engine, (void*)a);
      dnnl::memory b_m(b_md, engine, (void*)b);
      dnnl::memory c_m(c_md, engine, (void*)c);

      auto mat_src_mem = a_m;
      auto mat_weights_mem = b_m;
      auto mat_dst_mem = c_m;

      // ---- weight cache (ONLY fast path) ----
      if (use_fast_path) {
        struct WeightKey {
          void* ptr;
          dnnl_dim_t K;
          dnnl_dim_t N;
          dnnl_dim_t stride_0;
          dnnl_dim_t stride_1;
          const void* pd; // primitive descriptor identity

          bool operator==(const WeightKey& o) const {
            return ptr == o.ptr &&
                  K == o.K &&
                  N == o.N &&
                  stride_0 == o.stride_0 &&
                  stride_1 == o.stride_1 &&
                  pd == o.pd;
          }
        };

        struct WeightHash {
          size_t operator()(const WeightKey& k) const {
            size_t h = std::hash<void*>()(k.ptr);
            h ^= std::hash<int64_t>()(k.K) << 1;
            h ^= std::hash<int64_t>()(k.N) << 2;
            h ^= std::hash<int64_t>()(k.stride_0) << 3;
            h ^= std::hash<int64_t>()(k.stride_1) << 4;
            h ^= std::hash<const void*>()(k.pd) << 5;
            return h;
          }
        };

        static std::unordered_map<WeightKey, dnnl::memory, WeightHash> weight_cache;
        static std::mutex weight_mutex;

        auto expected_w_md = matmul_pd.weights_desc();
        WeightKey wkey{(void*)b, K, N, b_stride_0, b_stride_1,matmul_pd.get()};

        std::lock_guard<std::mutex> guard(weight_mutex);

        auto it = weight_cache.find(wkey);
        if (it == weight_cache.end()) {
          auto packed = dnnl::memory(expected_w_md, engine);

          dnnl::reorder(b_m, packed).execute(stream, b_m, packed);
          stream.wait();

          it = weight_cache.emplace(wkey, packed).first;
        }

        mat_weights_mem = it->second;

      } else {
        // fallback: no weight cache
        if (matmul_pd.weights_desc() != b_m.get_desc()) {
          mat_weights_mem = dnnl::memory(matmul_pd.weights_desc(), engine);
          dnnl::reorder(b_m, mat_weights_mem).execute(stream, b_m, mat_weights_mem);
        }
      }

      // ---- src reorder ----
      if (matmul_pd.src_desc() != a_m.get_desc()) {
        mat_src_mem = dnnl::memory(matmul_pd.src_desc(), engine);
        dnnl::reorder(a_m, mat_src_mem).execute(stream, a_m, mat_src_mem);
      }

      matmul.execute(
          stream,
          {
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