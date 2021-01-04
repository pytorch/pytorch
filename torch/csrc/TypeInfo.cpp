#include <torch/csrc/TypeInfo.h>

#include <torch/csrc/Dtype.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <c10/util/variant.h>

#include <type_traits>
#include <limits>
#include <sstream>

namespace torch {
using std::numeric_limits;
using at::ScalarType;

namespace {

constexpr double pow2(int x) {
  double y = 1;
  if (x > 0) {
    for (int i = 0; i < x; i++) {
      y *= 2;
    }
  } else {
    for (int i = 0; i > x; i--) {
      y /= 2;
    }
  }
  return y;
}

template <class T, class = std::enable_if_t<std::is_unsigned<T>::value>>
constexpr double generic_float_to_double(T x, int num_frac_bits) {
  int num_bits = sizeof(T)*8;
  T exponent_mask = T(T(-1) << (num_frac_bits + 1)) >> 1;
  T exponent_special = exponent_mask >> num_frac_bits;
  int exponent_offset = exponent_special >> 1;
  T fraction_mask = T(-1) >> (num_bits - num_frac_bits);

  T sign = x >> (num_bits - 1);
  T exponent_bits = (x & exponent_mask) >> num_frac_bits;
  T fraction = x & fraction_mask;
  
  if (exponent_bits == exponent_special) {
    throw std::runtime_error("conversion of Inf/NaN is not implemented");
  }
  int exponent = exponent_bits - exponent_offset;
  double fp64 = pow2(exponent) + fraction * pow2(exponent - num_frac_bits);
  if (sign) {
    fp64 = -fp64;
  }

  return fp64;
}

template <class T>
inline constexpr double to_double(T num) {
  return (double)num;
}

template <>
inline constexpr double to_double(at::Half num) {
  return generic_float_to_double(num.x, 10);
}

template <>
inline constexpr double to_double(at::BFloat16 num) {
  return generic_float_to_double(num.x, 7);
}

struct IInfo {
  ScalarType scalar_type;
  uint16_t bits;
  int64_t min;
  int64_t max;
};

struct FInfo {
  ScalarType scalar_type;
  uint16_t bits;
  uint16_t digits10;
  double min;
  double max;
  double eps;
  double tiny;

  inline double resolution() const {
    return std::pow(10, -this->digits10);
  }
};

struct UnderlyingInfo {
  ScalarType scalar_type;
  ScalarType underlying_type;
};

#define PY_IINFO(scalar_type, type) \
  (IInfo { \
    /* scalar_type = */ ScalarType::scalar_type, \
    /* bits = */ sizeof(type)*8, \
    /* min = */ numeric_limits<type>::lowest(), \
    /* max = */ numeric_limits<type>::max() \
  })
#define PY_FINFO(scalar_type, type) \
  (FInfo { \
    /* scalar_type = */ ScalarType::scalar_type, \
    /* bits = */ sizeof(type)*8, \
    /* digits10 = */ numeric_limits<type>::digits10, \
    /* min = */ to_double(numeric_limits<type>::lowest()), \
    /* max = */ to_double(numeric_limits<type>::max()), \
    /* eps = */ to_double(numeric_limits<type>::epsilon()), \
    /* tiny = */ to_double(numeric_limits<type>::min()) \
  })
#define PY_UNDERLYING_INFO(scalar_type, underlying_type) \
  (UnderlyingInfo { \
    /* scalar_type = */ ScalarType::scalar_type, \
    /* underlying_type = */ ScalarType::underlying_type \
  })
#define PY_NOINFO(_scalar_type) (c10::monostate {})

typedef c10::variant<c10::monostate, IInfo, FInfo, UnderlyingInfo> DtypeInfo;

constexpr size_t NUM_DTYPES = static_cast<size_t>(ScalarType::NumOptions);

constexpr DtypeInfo dtype_info_registry[NUM_DTYPES] = {
  PY_IINFO(Byte, uint8_t),
  PY_IINFO(Char, int8_t),
  PY_IINFO(Short, int16_t),
  PY_IINFO(Int, int32_t),
  PY_IINFO(Long, int64_t),
  PY_FINFO(Half, at::Half),
  PY_FINFO(Float, float),
  PY_FINFO(Double, double),
  PY_UNDERLYING_INFO(ComplexHalf, Half),
  PY_UNDERLYING_INFO(ComplexFloat, Float),
  PY_UNDERLYING_INFO(ComplexDouble, Double),
  PY_NOINFO(Bool),
  PY_UNDERLYING_INFO(QInt8, Char),
  PY_UNDERLYING_INFO(QUInt8, Byte),
  PY_UNDERLYING_INFO(QInt32, Int),
  PY_FINFO(BFloat16, at::BFloat16)
};

inline c10::string_view dtypeName(ScalarType scalar_type) {
  return getPyDtype(scalar_type).primary_name;
}

} // namespace

void initTypeInfoBindings(PyObject* module) {
  py::options options;
  options.disable_user_defined_docstrings();
  options.disable_function_signatures();

  py::class_<IInfo>(module, "iinfo", py::is_final())
      .def_static(
          "__new__",
          [](ScalarType scalar_type) {
            HANDLE_TH_ERRORS
            const DtypeInfo& info = dtype_info_registry[static_cast<size_t>(scalar_type)];
            const DtypeInfo* target_info;
            if (c10::holds_alternative<UnderlyingInfo>(info)) {
              auto underlying_type = c10::get<UnderlyingInfo>(info).underlying_type;
              target_info = &dtype_info_registry[static_cast<size_t>(underlying_type)];
            } else {
              target_info = &info;
            }

            auto dtype_name = dtypeName(scalar_type);
            TORCH_CHECK_TYPE(
              !c10::holds_alternative<c10::monostate>(*target_info),
              "torch.", dtype_name, " is not supported by torch.iinfo"
            );
            TORCH_CHECK_TYPE(
              !c10::holds_alternative<FInfo>(*target_info),
              "torch.iinfo() requires an integer input type. "
              "Use torch.finfo to handle torch.", dtype_name
            );

            return c10::get<IInfo>(*target_info);
            END_HANDLE_TH_ERRORS_PYBIND
          },
          py::return_value_policy::reference,
          py::arg("type"))
      .def("__repr__",
          [](const IInfo& iinfo) {
            std::ostringstream oss;
            oss << "iinfo(min=" << iinfo.min;
            oss << ", max=" << iinfo.max;
            oss << ", dtype=" << dtypeName(iinfo.scalar_type) << ")";
            return oss.str();
          })
      .def_property_readonly(
          "dtype",
          [](const IInfo& iinfo) {
            return dtypeName(iinfo.scalar_type);
          })
      .def_readonly("bits", &IInfo::bits)
      .def_readonly("min", &IInfo::min)
      .def_readonly("max", &IInfo::max);

  py::class_<FInfo>(module, "finfo", py::is_final())
      .def_static(
          "__new__",
          [](c10::optional<ScalarType> type) {
            HANDLE_TH_ERRORS
            auto scalar_type = type.value_or(tensors::get_default_scalar_type());

            const DtypeInfo& info = dtype_info_registry[static_cast<size_t>(scalar_type)];
            const DtypeInfo* target_info;
            if (c10::holds_alternative<UnderlyingInfo>(info)) {
              auto underlying_type = c10::get<UnderlyingInfo>(info).underlying_type;
              target_info = &dtype_info_registry[static_cast<size_t>(underlying_type)];
            } else {
              target_info = &info;
            }

            auto dtype_name = dtypeName(scalar_type);
            TORCH_CHECK_TYPE(
              !c10::holds_alternative<c10::monostate>(*target_info),
              "torch.", dtype_name, " is not supported by torch.finfo"
            );
            TORCH_CHECK_TYPE(
              !c10::holds_alternative<IInfo>(*target_info),
              "torch.finfo() requires a floating point input type. "
              "Use torch.iinfo to handle torch.", dtype_name
            );

            return c10::get<FInfo>(*target_info);
            END_HANDLE_TH_ERRORS_PYBIND
          },
          py::return_value_policy::reference,
          py::arg("type") = c10::nullopt)
      .def("__repr__",
          [](const FInfo& finfo) {
            std::ostringstream oss;
            oss << "finfo(resolution=" << finfo.resolution();
            oss << ", min=" << finfo.min;
            oss << ", max=" << finfo.max;
            oss << ", eps=" << finfo.eps;
            oss << ", tiny=" << finfo.tiny;
            oss << ", dtype=" << dtypeName(finfo.scalar_type) << ")";
            return oss.str();
          })
      .def_property_readonly(
          "dtype",
          [](const FInfo& finfo) {
            return dtypeName(finfo.scalar_type);
          })
      .def_property_readonly("resolution", &FInfo::resolution)
      .def_readonly("bits", &FInfo::bits)
      .def_readonly("min", &FInfo::min)
      .def_readonly("max", &FInfo::max)
      .def_readonly("eps", &FInfo::eps)
      .def_readonly("tiny", &FInfo::tiny);
}

} // namespace torch
