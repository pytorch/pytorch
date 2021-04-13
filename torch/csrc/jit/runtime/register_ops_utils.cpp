#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <torch/csrc/jit/runtime/slice_indices_adjust.h>

#include <c10/util/irange.h>

namespace torch {
namespace jit {

template <>
c10::impl::GenericList make_result_list<IValue>(const TypePtr& elemType) {
  return c10::impl::GenericList(elemType);
}

template <>
void listIndex<at::Tensor>(Stack* stack) {
  at::Tensor elem = pop(stack).to<at::Tensor>();
  c10::List<at::Tensor> list = pop(stack).to<c10::List<at::Tensor>>();

  auto pos =
      std::find_if(list.begin(), list.end(), [elem](const at::Tensor& b) {
        const auto cmp_result = elem.eq(b);
        return cmp_result.is_nonzero();
      });

  if (pos != list.end()) {
    push(stack, static_cast<int64_t>(std::distance(list.begin(), pos)));
  } else {
    AT_ERROR("'", elem, "' is not in list");
  }
}

template <>
void listCount<at::Tensor>(Stack* stack) {
  at::Tensor elem = pop(stack).to<at::Tensor>();
  c10::List<at::Tensor> list = pop(stack).to<c10::List<at::Tensor>>();

  const int64_t count =
      std::count_if(list.begin(), list.end(), [&](const at::Tensor& b) {
        const auto cmp_result = elem.eq(b);
        return cmp_result.is_nonzero();
      });
  push(stack, count);
}

template <>
void listEq<at::Tensor>(Stack* stack) {
  c10::List<at::Tensor> b = pop(stack).to<c10::List<at::Tensor>>();
  c10::List<at::Tensor> a = pop(stack).to<c10::List<at::Tensor>>();
  push(stack, tensor_list_equal(a, b));
}

template <>
void listNe<at::Tensor>(Stack* stack) {
  c10::List<at::Tensor> b = pop(stack).to<c10::List<at::Tensor>>();
  c10::List<at::Tensor> a = pop(stack).to<c10::List<at::Tensor>>();
  push(stack, !tensor_list_equal(a, b));
}

template <>
void listSort<at::Tensor>(Stack* stack) {
  bool reverse = pop(stack).toBool();
  c10::List<at::Tensor> list = pop(stack).toTensorList();
  std::sort(
      list.begin(),
      list.end(),
      [reverse](const at::Tensor& a, const at::Tensor& b) -> bool {
        // "strict weak ordering" issue - see other sort
        if (a.getIntrusivePtr() == b.getIntrusivePtr()) {
          return false;
        }
        return (a.lt(b).is_nonzero()) ^ reverse;
      });
}

template <>
void listCopyAndSort<at::Tensor>(Stack* stack) {
  c10::List<at::Tensor> list = pop(stack).toTensorList();
  auto list_copied = list.copy();
  std::sort(
      list_copied.begin(),
      list_copied.end(),
      [](const at::Tensor& a, const at::Tensor& b) {
        return a.lt(b).is_nonzero();
      });
  push(stack, list_copied);
}

template <>
void listRemove<at::Tensor>(Stack* stack) {
  at::Tensor elem = pop(stack).to<at::Tensor>();
  c10::List<at::Tensor> list = pop(stack).to<c10::List<at::Tensor>>();

  auto pos = std::find_if(list.begin(), list.end(), [&](const at::Tensor& b) {
    const auto cmp_result = elem.eq(b);
    return cmp_result.is_nonzero();
  });

  if (pos != list.end()) {
    list.erase(pos);
  } else {
    AT_ERROR("list.remove(x): x not in list");
  }
}

void checkImplicitTensorToNum(const at::Tensor& t, bool toInt) {
  if (t.requires_grad()) {
    throw std::runtime_error(
        "Cannot input a tensor that requires grad as a scalar argument");
  }
  if (t.sizes().size() != 0) {
    throw std::runtime_error(
        "Cannot input a tensor of dimension other than 0 as a scalar argument");
  }
  if (toInt && !isIntegralType(t.scalar_type(), /*includeBool=*/false)) {
    std::stringstream ss;
    ss << "Cannot input a tensor of type " << t.scalar_type()
       << " as an integral argument";
    throw std::runtime_error(ss.str());
  }
}

IValue tensorToListRecursive(
    char* data,
    int64_t cur_dim,
    int64_t num_tensor_dims,
    TypePtr ty,
    at::ScalarType scalar_ty,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    size_t element_size) {
  // If ty is a ListType, get the element type.
  if (auto list_type = ty->cast<ListType>()) {
    ty = list_type->getElementType();
  } else {
    // If the output type is a scalar, read and push one scalar of
    // the right type onto the stack.
    if (ty == IntType::get()) {
      int64_t scalar = *(int64_t*)data;
      return IValue(scalar);
    } else if (ty == FloatType::get()) {
      TORCH_INTERNAL_ASSERT(
          scalar_ty == at::ScalarType::Float ||
              scalar_ty == at::ScalarType::Double,
          "Unexpected scalar type for Tensor");
      double scalar =
          scalar_ty == at::ScalarType::Float ? *(float*)data : *(double*)data;
      return IValue(scalar);
    } else if (ty == ComplexType::get()) {
      TORCH_INTERNAL_ASSERT(
          scalar_ty == at::ScalarType::ComplexFloat ||
              scalar_ty == at::ScalarType::ComplexDouble,
          "Unexpected scalar type for Tensor");
      c10::complex<double> scalar = scalar_ty == at::ScalarType::ComplexFloat
          ? *(c10::complex<float>*)data
          : *(c10::complex<double>*)data;
      return IValue(scalar);
    } else if (ty == BoolType::get()) {
      bool scalar = *(bool*)data;
      return IValue(scalar);
    } else {
      TORCH_CHECK(
          false,
          ty->repr_str(),
          " is not one of the supported types for tolist: int, float, bool");
    }
  }

  // Make the result list consisting of elements of type ty. Since this
  // invocation is processing dimension cur_dim, there will be sizes[cur_dim]
  // output elements.
  auto result = c10::impl::GenericList(ty);
  result.reserve(sizes[cur_dim]);

  // Since ty was a list type, tensorToListRecursive needs to be called
  // recursively on each slice of the tensor in the current dimension.
  for (int64_t i = 0, e = sizes[cur_dim]; i < e; ++i) {
    auto inner_result = tensorToListRecursive(
        data,
        cur_dim + 1,
        num_tensor_dims,
        ty,
        scalar_ty,
        sizes,
        strides,
        element_size);

    if (inner_result.isList()) {
      result.emplace_back(inner_result.toList());
    } else if (inner_result.isComplexDouble()) {
      result.emplace_back(inner_result.toComplexDouble());
    } else if (inner_result.isDouble()) {
      result.emplace_back(inner_result.toDouble());
    } else if (inner_result.isInt()) {
      result.emplace_back(inner_result.toInt());
    } else if (inner_result.isBool()) {
      result.emplace_back(inner_result.toBool());
    } else {
      TORCH_INTERNAL_ASSERT("Unknown return type for tensorToListRecursive");
    }

    data += strides[cur_dim] * element_size;
  }

  return result;
}

void checkDoubleInRange(double a) {
  if (std::isnan(a) || std::isinf(a) ||
      a > double(std::numeric_limits<int64_t>::max()) ||
      a < double(std::numeric_limits<int64_t>::min())) {
    throw c10::Error(
        "Cannot convert float " + c10::to_string(a) + " to integer", "");
    return;
  }
}

int64_t partProduct(int n, int m) {
  if (m <= (n + 1))
    return (int64_t)n;
  if (m == (n + 2))
    return (int64_t)n * m;
  auto k = n + (m - n) / 2; // Overflow-safe midpoint
  if ((k & 1) != 1)
    k = k - 1;
  return partProduct(n, k) * partProduct(k + 2, m);
}

void loop(int n, int64_t& p, int64_t& r) {
  if (n <= 2)
    return;
  loop(n / 2, p, r);
  p = p * partProduct(n / 2 + 1 + ((n / 2) & 1), n - 1 + (n & 1));
  r = r * p;
}

int nminussumofbits(int v) {
  long w = (long)v;
  w -= (0xaaaaaaaa & w) >> 1; // NOLINT
  w = (w & 0x33333333) + ((w >> 2) & 0x33333333); // NOLINT
  w = (w + (w >> 4)) & 0x0f0f0f0f; // NOLINT
  w += w >> 8; // NOLINT
  w += w >> 16; // NOLINT
  return v - (int)(w & 0xff); // NOLINT
}

int64_t factorial(int n) {
  if (n < 0) {
    throw std::runtime_error("factorial() not defined for negative values");
  }
  int64_t p = 1, r = 1;
  loop(n, p, r);
  return r << nminussumofbits(n);
}

double degrees(double x) {
  return x * radToDeg;
}
double radians(double x) {
  return x * degToRad;
}

int64_t normalizeIndex(int64_t idx, int64_t list_size) {
  if (idx < 0) {
    // Handle negative indexing
    idx = list_size + idx;
  }
  return idx;
}

void listAppend(Stack* stack) {
  IValue el = pop(stack).to<IValue>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  list.push_back(std::move(el));
  push(stack, std::move(list));
}

void listReverse(Stack* stack) {
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  std::reverse(list.begin(), list.end());
}

void listPopImpl(Stack* stack, const char* empty_message) {
  int64_t idx = pop(stack).to<int64_t>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  const int64_t list_size = list.size();
  const int64_t normalized_idx = normalizeIndex(idx, list_size);

  if (list_size == 0) {
    AT_ERROR(empty_message);
  }

  push(stack, getItem(list, idx));
  list.erase(list.begin() + normalized_idx);
}

void listPop(Stack* stack) {
  return listPopImpl(stack, "pop from empty list");
}

void listClear(Stack* stack) {
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  list.clear();
}

void listDelete(Stack* stack) {
  listPopImpl(stack, "pop index out of range");
  pop(stack);
}

void listInsert(Stack* stack) {
  IValue elem = pop(stack).to<IValue>();
  int64_t idx = pop(stack).to<int64_t>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  const int64_t list_size = list.size();
  const int64_t normalized_idx = normalizeIndex(idx, list_size);

  if (normalized_idx < 0 || normalized_idx >= list_size) {
    if (normalized_idx < 0) {
      list.insert(list.begin(), elem);
    } else {
      list.push_back(elem);
    }
  } else {
    list.insert(list.begin() + normalized_idx, elem);
  }
}

void listExtend(Stack* stack) {
  c10::List<IValue> b = pop(stack).to<c10::List<IValue>>();
  c10::List<IValue> a = pop(stack).to<c10::List<IValue>>();

  a.reserve(a.size() + b.size());
  for (size_t i = 0; i < b.size(); ++i) {
    a.push_back(b.get(i));
  }
}

void listCopy(Stack* stack) {
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();
  push(stack, list.copy());
}

void listSelect(Stack* stack) {
  int64_t idx = pop(stack).to<int64_t>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  auto element = getItem(list, idx);
  push(stack, std::move(element));
}

void listLen(Stack* stack) {
  c10::List<IValue> a = pop(stack).to<c10::List<IValue>>();

  const int64_t size = a.size();
  push(stack, size);
}

void listList(Stack* stack) {
  c10::List<IValue> a = pop(stack).to<c10::List<IValue>>();
  push(stack, a.copy());
}

void listAdd(Stack* stack) {
  c10::List<IValue> b = pop(stack).to<c10::List<IValue>>();
  c10::List<IValue> a = pop(stack).to<c10::List<IValue>>();

  c10::List<IValue> ret = make_result_list<IValue>(a.elementType());

  if (a.use_count() == 1) {
    ret = std::move(a);
  } else {
    ret = a.copy();
  }

  ret.append(std::move(b));

  push(stack, std::move(ret));
}

void listInplaceAdd(Stack* stack) {
  c10::List<IValue> b = pop(stack).to<List<IValue>>();
  c10::List<IValue> a = pop(stack).to<List<IValue>>();
  a.append(std::move(b));
  push(stack, std::move(a));
}

void listMulIntLeftInPlace(Stack* stack) {
  int64_t n = pop(stack).to<int64_t>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();
  if (n <= 0) {
    list.clear();
  } else if (n > 1) {
    size_t list_size = list.size();
    for (int64_t i = 1; i < n; i++) {
      for (size_t j = 0; j < list_size; j++) {
        list.push_back(list.get(j));
      }
    }
  }

  push(stack, std::move(list));
}

void listMulIntLeft(Stack* stack) {
  int64_t n = pop(stack).to<int64_t>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  c10::List<IValue> ret = make_result_list<IValue>(list.elementType());
  const auto size = list.size() * n;
  ret.reserve(size);

  for (const auto i : c10::irange(n)) {
    (void)i; // Suppress unused variable warning
    for (IValue e : list) {
      ret.push_back(std::move(e));
    }
  }

  push(stack, std::move(ret));
}

void listMulIntRight(Stack* stack) {
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();
  int64_t n = pop(stack).to<int64_t>();

  c10::List<IValue> ret = make_result_list<IValue>(list.elementType());
  const auto size = list.size() * n;
  ret.reserve(size);

  for (const auto i : c10::irange(n)) {
    (void)i; // Suppress unused variable warning
    for (IValue e : list) {
      ret.push_back(std::move(e));
    }
  }

  push(stack, std::move(ret));
}

void listSlice(Stack* stack) {
  int64_t step = pop(stack).to<int64_t>();
  int64_t end = pop(stack).to<int64_t>();
  int64_t start = pop(stack).to<int64_t>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  const int64_t list_size = list.size();

  c10::List<IValue> sliced_list = make_result_list<IValue>(list.elementType());
  const int64_t num_values =
      slice_indices_adjust(list_size, &start, &end, step);
  sliced_list.reserve(num_values);

  int i = start;
  for (int j = 0; j < num_values; ++j) {
    sliced_list.push_back(list.get(i));
    i += step;
  }

  push(stack, std::move(sliced_list));
}

void listSetItem(Stack* stack) {
  IValue value = pop(stack).to<IValue>();
  int64_t idx = pop(stack).to<int64_t>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  setItem(list, idx, std::move(value));

  push(stack, std::move(list));
}
} // namespace jit
} // namespace torch
