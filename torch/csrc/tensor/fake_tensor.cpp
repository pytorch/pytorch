#include <torch/csrc/tensor/fake_tensor.h>

#include <pybind11/pybind11.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/core/TensorBase.h>
#include <torch/csrc/utils/python_arg_parser.h>

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

using SerializeBuffer = std::vector<uint8_t>;

void serialize(SerializeBuffer& result, uint64_t value) {
  result.push_back((uint8_t)value);
  result.push_back((uint8_t)(value >> 8));
  result.push_back((uint8_t)(value >> 16));
  result.push_back((uint8_t)(value >> 24));
  result.push_back((uint8_t)(value >> 32));
  result.push_back((uint8_t)(value >> 40));
  result.push_back((uint8_t)(value >> 48));
  result.push_back((uint8_t)(value >> 56));
}

void serialize(SerializeBuffer& result, int8_t value) {
  result.push_back((uint8_t)value);
}

void serialize(SerializeBuffer& result, int16_t value) {
  result.push_back((uint8_t)value);
  result.push_back((uint8_t)(value >> 8));
}

void serialize(SerializeBuffer& result, int32_t value) {
  result.push_back((uint8_t)value);
  result.push_back((uint8_t)(value >> 8));
  result.push_back((uint8_t)(value >> 16));
  result.push_back((uint8_t)(value >> 24));
}

void serialize(SerializeBuffer& result, int64_t value) {
  serialize(result, (uint64_t)value);
}

void serializePointer(SerializeBuffer& result, void* value) {
  serialize(result, (uint64_t)value);
}

// General enum serializer.
template <
    typename T,
    typename = typename std::enable_if<std::is_enum<T>::value, T>::type>
void serialize(SerializeBuffer& result, T value) {
  serialize(result, std::underlying_type_t<T>(value));
}

void serialize(SerializeBuffer& result, const c10::SymInt& value) {
  if (auto i = value.maybe_as_int()) {
    result.push_back(0);
    serialize(result, *i);
  } else {
    result.push_back(1);
    serializePointer(result, value.toSymNodeImplUnowned());
  }
}

void serialize(SerializeBuffer& result, const std::string& s) {
  result.insert(result.end(), s.begin(), s.end());
}

void serialize(SerializeBuffer& result, c10::Device value) {
  serialize(result, value.str());
}

template <typename T>
void serialize(SerializeBuffer& result, const c10::ArrayRef<T>& value) {
  serialize(result, value.size());
  for (const auto& v : value) {
    serialize(result, v);
  }
}

struct Globals {
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

void serialize_stride(SerializeBuffer& result, const at::Tensor& tensor) {
  auto layout = tensor.layout();
  if (layout != c10::Layout::Strided) {
    return;
  }

  serialize(result, tensor.sym_strides());
}

bool is_sparse_any(const at::Tensor& tensor) {
  // return is_sparse_coo(tensor) || is_sparse_compressed(tensor);
  auto layout = tensor.layout();
  return layout == c10::Layout::Sparse || layout == c10::Layout::SparseCsr ||
      layout == c10::Layout::SparseCsc || layout == c10::Layout::SparseBsr ||
      layout == c10::Layout::SparseBsc;
}

void serialize_memory_format(
    SerializeBuffer& result,
    const at::Tensor& tensor) {
  if (is_sparse_any(tensor)) {
    serialize(result, 0);
    return;
  }

  //  memory_format: Optional[torch.memory_format] = suggest_memory_format(t)
  //  if is_sparse_any(t) or not t.is_contiguous(memory_format=memory_format):
  //      memory_format = None

  auto memory_format = tensor.suggest_memory_format();

  bool is_contiguous = tensor.is_contiguous(memory_format);
  if (!is_contiguous) {
    serialize(result, 0);
    return;
  }

  serialize(result, memory_format);
}

py::object extract_tensor_metadata(py::handle t_) {
  const at::Tensor& t = THPVariable_Unpack(t_.ptr());

  // TODO: Do we need to check_has_torch_function()? Is that ever true for a
  // FakeTensor?

  SerializeBuffer result;

  // t.dtype
  serialize(result, t.scalar_type());
  // t.shape
  serialize(result, t.sym_sizes());
  // t.stride() if t.layout == torch.strided else ()
  serialize_stride(result, t);
  // t.device
  serialize(result, t.device());
  // t.layout
  serialize(result, t.layout());
  // memory_format
  serialize_memory_format(result, t);
  // t.storage_offset()
  serialize(result, t.sym_storage_offset());
  // t.requires_grad
  serialize(result, t.requires_grad());
  // t.is_quantized
  serialize(result, t.is_quantized());
  // t.is_conj()
  serialize(result, t.is_conj());
  // t.is_neg()
  serialize(result, t.is_neg());
  // t.is_inference()
  serialize(result, t.is_inference());
  // t.is_sparse
  bool is_sparse = t.is_sparse();
  serialize(result, is_sparse);
  if (is_sparse) {
    // t.is_coalesced() if t.is_sparse else None
    serialize(result, t.is_coalesced());
    // t.dense_dim() if t.is_sparse else None
    serialize(result, t.dense_dim());
    // t.sparse_dim() if t.is_sparse else None
    serialize(result, t.sparse_dim());
  }

  return py::bytes((const char*)result.data(), result.size());
}

} // anonymous namespace

void torch::fake_tensor::initialize(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("_FakeTensor_extract_tensor_metadata", extract_tensor_metadata);
}
