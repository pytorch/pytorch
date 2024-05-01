#include <torch/csrc/tensor/fake_tensor.h>

#include <pybind11/pybind11.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/pybind.h>

using namespace pybind11::literals;

namespace std {
template <>
struct hash<c10::SymInt> {
  size_t operator()(const c10::SymInt& si) const noexcept {
    if (auto mi = si.maybe_as_int()) {
      return std::hash<int64_t>{}(*mi);
    } else {
      // Use the SymNode address as the hash.
      auto b = si.toSymNodeImplUnowned();
      return std::hash<void*>{}(b);
    }
  }
};

} // namespace std

namespace {

void serialize(std::string& result, uint64_t value) {
  result.push_back((char)value);
  result.push_back((char)(value >> 8));
  result.push_back((char)(value >> 16));
  result.push_back((char)(value >> 24));
  result.push_back((char)(value >> 32));
  result.push_back((char)(value >> 40));
  result.push_back((char)(value >> 48));
  result.push_back((char)(value >> 56));
}

void serialize(std::string& result, int8_t value) {
  result.push_back(value);
}

void serialize(std::string& result, int16_t value) {
  result.push_back((char)value);
  result.push_back((char)(value >> 8));
}

void serialize(std::string& result, int32_t value) {
  result.push_back((char)value);
  result.push_back((char)(value >> 8));
  result.push_back((char)(value >> 16));
  result.push_back((char)(value >> 24));
}

void serialize(std::string& result, int64_t value) {
  serialize(result, (uint64_t)value);
}

void serializePointer(std::string& result, void* value) {
  serialize(result, (uint64_t)value);
}

// General enum serializer.
template <
    typename T,
    typename = typename std::enable_if<std::is_enum<T>::value, T>::type>
void serialize(std::string& result, T value) {
  serialize(result, std::underlying_type_t<T>(value));
}

void serialize(std::string& result, const c10::SymInt& value) {
  if (auto i_ = value.maybe_as_int()) {
    int64_t i = *i_;
    // This is really only valid in little endian - but we actually just have to
    // be approximate since we never deserialize this.
    if ((i & 0xFF) == 0xFF) {
      result.push_back(0);
    }
    serialize(result, i);
  } else {
    result.push_back(-1);
    serializePointer(result, value.toSymNodeImplUnowned());
  }
}

void serialize(std::string& result, c10::Device value) {
  result += value.str();
}

template <typename T>
void serialize(std::string& result, const c10::ArrayRef<T>& value) {
  serialize(result, value.size());
  for (const auto& v : value) {
    serialize(result, v);
  }
}

bool is_true(py::handle h) {
  return PyObject_IsTrue(h.ptr());
}

template <typename T>
py::object pybind_wrap(T x) {
  return py::reinterpret_steal<py::object>(torch::autograd::utils::wrap(x));
}

template <typename T, typename U = std::hash<T>>
void hash_combine_with(size_t& value, T with) {
  value = c10::hash_combine(value, U{}(with));
}

template <typename T>
void hash_combine_with(size_t& value, const c10::ArrayRef<T>& array) {
  for (const auto& v : array) {
    hash_combine_with(value, v);
  }
}

struct fake_tensor_t {
  const py::module __module__;

  const py::type FakeTensor;
  const py::type TensorMetadata;
  const py::type _DispatchCacheKey;
  const py::object _UNHASHABLE;
  const py::function extract_tensor_metadata;

  fake_tensor_t(const py::module& _subclasses)
      : __module__(_subclasses.attr("fake_tensor")),
        FakeTensor(__module__.attr("FakeTensor")),
        TensorMetadata(__module__.attr("TensorMetadata")),
        _DispatchCacheKey(__module__.attr("_DispatchCacheKey")),
        _UNHASHABLE(__module__.attr("_UNHASHABLE")),
        extract_tensor_metadata(__module__.attr("extract_tensor_metadata")) {}
};

struct meta_utils_t {
  const py::module __module__;

  const py::function is_sparse_any;

  meta_utils_t(const py::module& _subclasses)
      : __module__(_subclasses.attr("meta_utils")),
        is_sparse_any(__module__.attr("is_sparse_any")) {}
};

struct _subclasses_t {
  const py::module __module__;

  const fake_tensor_t fake_tensor;
  const meta_utils_t meta_utils;

  _subclasses_t(const py::module& torch)
      : __module__(torch.attr("_subclasses")),
        fake_tensor(__module__),
        meta_utils(__module__) {}
};

struct _prims_common_t {
  const py::module __module__;

  const py::function suggest_memory_format;

  _prims_common_t(const py::module& torch)
      : __module__(torch.attr("_prims_common")),
        suggest_memory_format(__module__.attr("suggest_memory_format")) {}
};

struct torch_t {
  const py::module __module__;

  const _prims_common_t _prims_common;
  const _subclasses_t _subclasses;

  const py::type SymBool;
  const py::type SymFloat;
  const py::type SymInt;
  const py::object strided;

  torch_t()
      : __module__(py::module_::import("torch")),
        _prims_common(__module__),
        _subclasses(__module__),
        SymBool(__module__.attr("SymBool")),
        SymFloat(__module__.attr("SymFloat")),
        SymInt(__module__.attr("SymInt")),
        strided(__module__.attr("strided")) {}
};

struct Globals {
  torch_t torch;

  static const Globals& singleton() {
    // Purposely leak this object!
    static Globals* instance = new Globals();
    return *instance;
  }

  Globals(const Globals&) = delete;
  Globals& operator=(Globals&) = delete;

 private:
  Globals() = default;
};

py::tuple into_tuple(std::vector<py::object> v) {
  auto tup = py::tuple(v.size());
  for (ssize_t i = (ssize_t)v.size() - 1; i >= 0; i--) {
    tup[i] = std::move(v[i]);
  }
  return tup;
}

void prep_args_for_hash(
    std::vector<py::object>& output,
    py::handle args,
    py::handle self);
py::object extract_tensor_metadata(py::handle t);

void prep_arg_for_hash(
    std::vector<py::object>& output,
    py::handle arg,
    py::handle self) {
  auto& torch = Globals::singleton().torch;
  auto& fake_tensor = torch._subclasses.fake_tensor;

  if (py::isinstance(arg, fake_tensor.FakeTensor)) {
    if (is_true(arg.attr("_has_symbolic_sizes_strides")) ||
        arg.attr("fake_mode").ptr() != self.ptr()) {
      output.push_back(fake_tensor._UNHASHABLE);
    } else {
      output.push_back(extract_tensor_metadata(arg));
    }
  } else if (
      py::isinstance(arg, torch.SymBool) || py::isinstance(arg, torch.SymInt) ||
      py::isinstance(arg, torch.SymFloat)) {
    output.push_back(fake_tensor._UNHASHABLE);
  } else if (
      PyDict_Check(arg.ptr()) || PyList_Check(arg.ptr()) ||
      PyTuple_Check(arg.ptr())) {
    prep_args_for_hash(output, arg, self);
  } else {
    output.push_back(py::type::of(arg));
    output.push_back(arg.cast<py::object>());
  }
}

void prep_args_for_hash(
    std::vector<py::object>& output,
    py::handle args,
    py::handle self) {
  if (PyDict_Check(args.ptr())) {
    ssize_t ppos = 0;
    PyObject *key = nullptr, *value = nullptr;
    while (PyDict_Next(args.ptr(), &ppos, &key, &value)) {
      prep_arg_for_hash(output, key, self);
      prep_arg_for_hash(output, value, self);
    }
    return;
  }

  for (const auto& item : args) {
    prep_arg_for_hash(output, item, self);
  }
}

py::object compute_cache_key(
    py::handle self,
    py::handle func,
    py::handle args,
    py::handle kwargs) {
  auto& torch = Globals::singleton().torch;

  std::vector<py::object> values;
  values.reserve(10);
  values.push_back(func.cast<py::object>());

  values.push_back(torch.__module__.attr("get_default_dtype")());

  values.push_back(torch.__module__.attr("_C").attr("_get_default_device")());
  // values.push_back(THPModule_getDefaultDevice(nullptr, nullptr));

  values.push_back(torch.__module__.attr("is_inference_mode_enabled")());

  auto shape_env = self.attr("shape_env");
  if (!shape_env.is_none()) {
    values.push_back(shape_env.attr("settings"));
  } else {
    values.push_back(py::none());
  }

  prep_args_for_hash(values, args, self);
  prep_args_for_hash(values, kwargs, self);

  return torch._subclasses.fake_tensor._DispatchCacheKey(into_tuple(values));
}

c10::MemoryFormat suggest_memory_format(const at::Tensor& tensor) {
  // if x.layout != torch.strided:
  //       return torch.contiguous_format
  if (tensor.layout() != c10::Layout::Strided) {
    return c10::MemoryFormat::Contiguous;
  }

  // if are_strides_like_channels_last(x.shape, x.stride()):
  //       return torch.channels_last if x.ndim == 4 else torch.channels_last_3d
  auto shape = tensor.sizes();
  auto strides = tensor.strides();
  if (c10::is_channels_last_strides_2d(shape, strides)) {
    return c10::MemoryFormat::ChannelsLast;
  }
  if (c10::is_channels_last_strides_3d(shape, strides)) {
    return c10::MemoryFormat::ChannelsLast3d;
  }

  return c10::MemoryFormat::Contiguous;
}

py::object extract_tensor_metadata(py::handle t) {
  // auto& torch = Globals::singleton().torch;

  const at::Tensor& tensor = THPVariable_Unpack(t.ptr());
  std::string result;

  bool is_sparse = tensor.is_sparse();
  auto layout_ = tensor.layout();

  // auto memory_format = torch._prims_common.suggest_memory_format(t);
  //
  c10::MemoryFormat memory_format = suggest_memory_format(tensor);
  if (!tensor.is_sparse() && tensor.is_contiguous(memory_format)) {
    // NumOptions used as a sentinal
    memory_format = c10::MemoryFormat::NumOptions;
  }
  serialize(result, memory_format);

  // Do we need to do this?
  // if (torch::check_has_torch_function(t.ptr())) {
  //   fprintf(stderr, "check_has_torch_function returned true\n");
  // }

  // dtype
  serialize(result, tensor.scalar_type());

  // py::object shape =
  // py::reinterpret_steal<py::object>(THPSize_NewFromSymSizes(tensor));
  auto sym_sizes = tensor.sizes();
  for (auto i : c10::irange(sym_sizes.size())) {
    serialize(result, i);
  }

  // py::object layout = pybind_wrap(torch::getTHPLayout(layout_));
  serialize(result, layout_);

  // py::object stride = (layout_ == c10::Layout::Strided) ? t.attr("stride")()
  // : py::tuple(0);
  if (layout_ == c10::Layout::Strided) {
    serialize(result, tensor.sym_strides());
  }

  // py::object device =
  // py::reinterpret_steal<py::object>(THPDevice_New(tensor.device()));
  serialize(result, t.attr("fake_device").cast<c10::Device>());

  // py::object storage_offset = py::cast(tensor.sym_storage_offset());
  serialize(result, tensor.sym_storage_offset());

  // TODO: py::object requires_grad = t.attr("requires_grad");

  // py::object is_quantized = pybind_wrap(tensor.is_quantized());
  serialize(result, tensor.is_quantized());

  // py::object is_conj = pybind_wrap(tensor.is_conj());
  serialize(result, tensor.is_conj());

  // py::object is_neg = pybind_wrap(tensor.is_neg());
  serialize(result, tensor.is_neg());

  // py::object is_inference = pybind_wrap(tensor.is_inference());
  serialize(result, tensor.is_inference());

  if (is_sparse) {
    // py::object is_coalesced = pybind_wrap(is_sparse &&
    // tensor.is_coalesced());
    serialize(result, tensor.is_coalesced());

    // py::object dense_dim = pybind_wrap(is_sparse ? tensor.dense_dim() : 0);
    serialize(result, tensor.dense_dim());

    // py::object sparse_dim = pybind_wrap(is_sparse ? tensor.sparse_dim() : 0);
    serialize(result, tensor.sparse_dim());
  }

  // return pybind_wrap((int64_t)value);
  return py::bytes(result);

  //-- py::object dtype = pybind_wrap(torch::getTHPDtype(tensor.scalar_type()));
  //-- py::object shape =
  // py::reinterpret_steal<py::object>(THPSize_NewFromSymSizes(tensor));

  //-- py::object layout = pybind_wrap(torch::getTHPLayout(layout_));
  //-- py::object stride = (layout_ == c10::Layout::Strided) ?
  // t.attr("stride")() : py::tuple(0);
  //-- py::object device =
  // py::reinterpret_steal<py::object>(THPDevice_New(tensor.device()));
  //-- py::object storage_offset = py::cast(tensor.sym_storage_offset());
  //-- py::object requires_grad = t.attr("requires_grad");
  //-- py::object is_quantized = pybind_wrap(tensor.is_quantized());
  //-- py::object is_conj = pybind_wrap(tensor.is_conj());
  //-- py::object is_neg = pybind_wrap(tensor.is_neg());
  //-- py::object is_inference = pybind_wrap(tensor.is_inference());
  //-- py::object is_coalesced = pybind_wrap(is_sparse &&
  // tensor.is_coalesced());
  //-- py::object dense_dim = pybind_wrap(is_sparse ? tensor.dense_dim() : 0);
  //-- py::object sparse_dim = pybind_wrap(is_sparse ? tensor.sparse_dim() : 0);

  //-- return torch._subclasses.fake_tensor.TensorMetadata(
  //--   dtype,
  //--   shape,
  //--   stride,
  //--   device,
  //--   layout,
  //--   memory_format,
  //--   storage_offset,
  //--   requires_grad,
  //--   is_quantized,
  //--   is_conj,
  //--   is_neg,
  //--   is_inference,
  //--   pybind_wrap(is_sparse),
  //--   is_coalesced,
  //--   dense_dim,
  //--   sparse_dim
  //-- );
}

} // anonymous namespace

void torch::fake_tensor::initialize(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("_FakeTensor_compute_cache_key", compute_cache_key);
  m.def("_FakeTensor_extract_tensor_metadata", extract_tensor_metadata);
}
