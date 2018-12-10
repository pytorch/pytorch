#ifndef _ABSTRACT_TYPES_HPP_
#define _ABSTRACT_TYPES_HPP_

#include <string>
#include <map>
#include <vector>
#include <mkldnn.h>
#include <mkldnn.hpp>

namespace ideep {

#if defined (__GNUC__)
#define IDEEP_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define IDEEP_DEPRECATED __declspec(deprecated)
#else
#define IDEEP_DEPRECATED
#endif

#ifdef _WIN32
#define IDEEP_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define IDEEP_EXPORT __attribute__((__visibility__("default")))
#else
#define IDEEP_EXPORT
#endif

#define IDEEP_ENFORCE(condition, message) \
  do {  \
    error::wrap_c_api((condition) \
        ? mkldnn_success : mkldnn_invalid_arguments, (message));  \
  } while(false) \

#define IDEEP_STD_ALL_EQ(v, i) \
  std::all_of(v.begin(), v.end(), []( \
        std::remove_reference<decltype(v)>::type::value_type k){return k == i;})

#define IDEEP_STD_ANY_LE(v, i) \
  std::any_of(v.begin(), v.end(), []( \
        std::remove_reference<decltype(v)>::type::value_type k){return k <= i;})

#define IDEEP_STD_EACH_SUB(v, i) \
  for (auto it = v.begin(); it != v.end(); it++) {*it -= i;}

#define IDEEP_CROSS_EQUAL(v1, v2, i1, i2) \
  (((v1 == i1) && (v2 == i2)) || ((v1 == i2) && (v2 == i1)))

// For 2D convolution with grouped weights, the ndims must be 5 (goihw)
#define IDEEP_IS_GROUPED_4DIMS(d) (((d).size() == 5) ? 1 : 0)

#define IDEEP_MOD_PTR(ptr, bytes) (((uintptr_t)(ptr)) & ((bytes) - 1))
#define IDEEP_IS_ALIGNED_PTR(ptr, bytes) ((IDEEP_MOD_PTR(ptr, bytes)) == 0)

struct error: public std::exception {
    mkldnn_status_t status;
    const char *message;

    error(mkldnn_status_t astatus, const char* amessage)
        : status(astatus), message(amessage) {}

    static void wrap_c_api(mkldnn_status_t status, const char * message) {
      if (status != mkldnn_success) {
        throw error(status, message);
      }
    }
};

/// Same class for resource management, except public default constructor
/// Movable support for better performance
template <typename T, typename traits = mkldnn::handle_traits<T>>
class c_wrapper :
  public std::shared_ptr<typename std::remove_pointer<T>::type> {
  using super = std::shared_ptr<typename std::remove_pointer<T>::type>;
public:
  /// Constructs a C handle wrapper.
  /// @param t The C handle to wrap.
  /// @param weak A flag to specify whether to construct a weak wrapper.
  c_wrapper(T t = nullptr, bool weak = false): super(t, [weak]() {
    auto dummy = [](T) {
      return decltype(traits::destructor(0))(0);
    };
    return weak? dummy : traits::destructor;
  }()) {}

  using super::super;

  /// Resets the value of a C handle.
  /// @param t The new value of the C handle.
  /// @param weak A flag to specify whether the wrapper should be weak.
  void reset(T t, bool weak = false) {
    auto dummy_destructor = [](T) {
      return decltype(traits::destructor(0))(0);
    };
    super::reset(t, weak ? dummy_destructor : traits::destructor);
  }
};

using batch_normalization_flag = mkldnn::batch_normalization_flag;
using query = mkldnn::query;
using scale_t = std::vector<float>;
using round_mode = mkldnn::round_mode;

#define IDEEP_OP_SCALE_MASK(scale_size) (((scale_size) > 1) ? 2 : 0)
#define IDEEP_TENSOR_SCALE_MASK(scale_size, grouped) \
  (((scale_size) > 1) ? ((grouped) ? 3 : 1) : 0)

const scale_t IDEEP_DEF_SCALE {1.0f};

constexpr int IDEEP_U8_MAX = 0xFF;
constexpr int IDEEP_S8_MAX = 0x7F;
constexpr int IDEEP_S32_MAX = 0x7FFFFFFF;
const std::map<mkldnn::memory::data_type, int> dt_max_map
{
  {mkldnn::memory::data_type::s32, IDEEP_S32_MAX},
  {mkldnn::memory::data_type::s8, IDEEP_S8_MAX},
  {mkldnn::memory::data_type::u8, IDEEP_U8_MAX}
};

/// hide other formats
enum format {
  format_undef = mkldnn_format_undef,
  any = mkldnn_any,
  blocked = mkldnn_blocked,
  x = mkldnn_x,
  nc = mkldnn_nc,
  io = mkldnn_io,
  oi = mkldnn_oi,
  nchw = mkldnn_nchw,
  nhwc = mkldnn_nhwc,
  chwn = mkldnn_chwn,
  ncdhw = mkldnn_ncdhw,
  ndhwc = mkldnn_ndhwc,
  oihw = mkldnn_oihw,
  ihwo = mkldnn_ihwo,
  hwio = mkldnn_hwio,
  oidhw = mkldnn_oidhw,
  goihw = mkldnn_goihw,
  hwigo = mkldnn_hwigo,
  ntc = mkldnn_ntc,
  tnc = mkldnn_tnc,
  iohw = mkldnn_format_last + 1,
  format_last = iohw + 1
};

/// cpu execution engine only.
struct engine: public mkldnn::engine {
  explicit engine(const mkldnn_engine_t& aengine) = delete;
  engine(engine const &) = delete;
  void operator =(engine const &) = delete;

  /// Singleton CPU engine for all primitives
  static IDEEP_EXPORT engine &cpu_engine();

  /// Put this global engine in only one library
  #define INIT_GLOBAL_ENGINE \
  ideep::engine &ideep::engine::cpu_engine() { \
    static engine cpu_engine; \
    return cpu_engine; \
  }

  inline static format default_format(int ndims) {
    switch(ndims) {
    case 1:
      return format::x;
    case 2:
      return format::nc;
    case 3:
      return format::blocked;
    case 4:
      return format::nchw;
    case 5:
      return format::ncdhw;
    default:
      return format::format_undef;
    }
  }

private:
  /// Constructs an engine.
  ///
  /// @param akind The kind of engine to construct.
  /// @param dformat The default data type of the engine.

  engine(kind akind = kind::cpu)
    :mkldnn::engine(akind, 0) {
  }
};

/// A default stream
struct stream: public mkldnn::stream {
  using mkldnn::stream::stream;
  static stream default_stream() {
    return stream(mkldnn::stream::kind::eager);
  }
};

using key_t = std::string;

using kind = mkldnn::primitive::kind;
using prop_kind = mkldnn::prop_kind;
using algorithm = mkldnn::algorithm;
using padding_kind = mkldnn::padding_kind;
}

#endif
