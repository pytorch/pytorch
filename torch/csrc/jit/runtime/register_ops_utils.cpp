#include <torch/csrc/jit/runtime/register_ops_utils.h>
#include <torch/csrc/jit/runtime/slice_indices_adjust.h>
#include <limits>

#include <c10/util/irange.h>

namespace torch::jit {

template <>
c10::impl::GenericList make_result_list<IValue>(const TypePtr& elemType) {
  return c10::impl::GenericList(elemType);
}

template <>
void listIndex<at::Tensor>(Stack& stack) {
  at::Tensor elem = pop(stack).to<at::Tensor>();
  c10::List<at::Tensor> list = pop(stack).to<c10::List<at::Tensor>>();

  auto pos =
      std::find_if(list.begin(), list.end(), [elem](const at::Tensor& b) {
        const auto cmp_result = elem.eq(b);
        return at::native::is_nonzero(cmp_result);
      });

  if (pos != list.end()) {
    push(stack, static_cast<int64_t>(std::distance(list.begin(), pos)));
  } else {
    AT_ERROR("'", elem, "' is not in list");
  }
}

template <>
void listCount<at::Tensor>(Stack& stack) {
  at::Tensor elem = pop(stack).to<at::Tensor>();
  c10::List<at::Tensor> list = pop(stack).to<c10::List<at::Tensor>>();

  const int64_t count =
      std::count_if(list.begin(), list.end(), [&](const at::Tensor& b) {
        const auto cmp_result = elem.eq(b);
        return at::native::is_nonzero(cmp_result);
      });
  push(stack, count);
}

template <>
void listEq<at::Tensor>(Stack& stack) {
  c10::List<at::Tensor> b = pop(stack).to<c10::List<at::Tensor>>();
  c10::List<at::Tensor> a = pop(stack).to<c10::List<at::Tensor>>();
  push(stack, tensor_list_equal(a, b));
}

template <>
void listNe<at::Tensor>(Stack& stack) {
  c10::List<at::Tensor> b = pop(stack).to<c10::List<at::Tensor>>();
  c10::List<at::Tensor> a = pop(stack).to<c10::List<at::Tensor>>();
  push(stack, !tensor_list_equal(a, b));
}

template <>
void listSort<at::Tensor>(Stack& stack) {
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
        return (at::native::is_nonzero(a.lt(b))) ^ reverse;
      });
}

template <>
void listCopyAndSort<at::Tensor>(Stack& stack) {
  c10::List<at::Tensor> list = pop(stack).toTensorList();
  auto list_copied = list.copy();
  std::sort(
      list_copied.begin(),
      list_copied.end(),
      [](const at::Tensor& a, const at::Tensor& b) {
        return at::native::is_nonzero(a.lt(b));
      });
  push(stack, list_copied);
}

template <>
void listRemove<at::Tensor>(Stack& stack) {
  at::Tensor elem = pop(stack).to<at::Tensor>();
  c10::List<at::Tensor> list = pop(stack).to<c10::List<at::Tensor>>();

  auto pos = std::find_if(list.begin(), list.end(), [&](const at::Tensor& b) {
    const auto cmp_result = elem.eq(b);
    return at::native::is_nonzero(cmp_result);
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
  if (!t.sizes().empty()) {
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

void listAppend(Stack& stack) {
  IValue el = pop(stack).to<IValue>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  list.push_back(std::move(el));
  push(stack, std::move(list));
}

void listReverse(Stack& stack) {
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  std::reverse(list.begin(), list.end());
}

void listPopImpl(Stack& stack, const char* empty_message) {
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

void listPop(Stack& stack) {
  return listPopImpl(stack, "pop from empty list");
}

void listClear(Stack& stack) {
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  list.clear();
}

void listDelete(Stack& stack) {
  listPopImpl(stack, "pop index out of range");
  pop(stack);
}

void listInsert(Stack& stack) {
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

void listExtend(Stack& stack) {
  c10::List<IValue> b = pop(stack).to<c10::List<IValue>>();
  c10::List<IValue> a = pop(stack).to<c10::List<IValue>>();

  a.reserve(a.size() + b.size());
  for (const auto i : c10::irange(b.size())) {
    a.push_back(b.get(i));
  }
}

void listCopy(Stack& stack) {
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();
  push(stack, list.copy());
}

void listSelect(Stack& stack) {
  int64_t idx = pop(stack).to<int64_t>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  auto element = getItem(list, idx);
  push(stack, std::move(element));
}

void listLen(Stack& stack) {
  c10::List<IValue> a = pop(stack).to<c10::List<IValue>>();

  const int64_t size = a.size();
  push(stack, size);
}

void listList(Stack& stack) {
  c10::List<IValue> a = pop(stack).to<c10::List<IValue>>();
  push(stack, a.copy());
}

void listAdd(Stack& stack) {
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

void listInplaceAdd(Stack& stack) {
  c10::List<IValue> b = pop(stack).to<c10::List<IValue>>();
  c10::List<IValue> a = pop(stack).to<c10::List<IValue>>();
  a.append(std::move(b));
  push(stack, std::move(a));
}

void listMulIntLeftInPlace(Stack& stack) {
  int64_t n = pop(stack).to<int64_t>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();
  if (n <= 0) {
    list.clear();
  } else if (n > 1) {
    size_t list_size = list.size();
    for (const auto i : c10::irange(1, n)) {
      (void)i; // Suppress unused variable warning
      for (const auto j : c10::irange(list_size)) {
        list.push_back(list.get(j));
      }
    }
  }

  push(stack, std::move(list));
}

void listMulIntLeft(Stack& stack) {
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

void listMulIntRight(Stack& stack) {
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

void listSlice(Stack& stack) {
  auto step_val = pop(stack);
  auto end_val = pop(stack);
  auto start_val = pop(stack);

  // By default, both start and end will be None.
  // By python convention, they will be translated into
  // INT64_MAX. If the step size is not given, it will be 1.
  int64_t step = step_val.isInt() ? step_val.to<int64_t>() : 1;
  int64_t end = end_val.isInt() ? end_val.to<int64_t>()
                                : std::numeric_limits<int64_t>::max();
  int64_t start = start_val.isInt() ? start_val.to<int64_t>()
                                    : std::numeric_limits<int64_t>::max();

  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  const int64_t list_size = list.size();

  c10::List<IValue> sliced_list = make_result_list<IValue>(list.elementType());
  const int64_t num_values =
      slice_indices_adjust(list_size, &start, &end, step);
  sliced_list.reserve(num_values);

  int i = start;
  for (const auto j : c10::irange(num_values)) {
    (void)j; // Suppress unused variable warning
    sliced_list.push_back(list.get(i));
    i += step;
  }

  push(stack, std::move(sliced_list));
}

void listSetItem(Stack& stack) {
  IValue value = pop(stack).to<IValue>();
  int64_t idx = pop(stack).to<int64_t>();
  c10::List<IValue> list = pop(stack).to<c10::List<IValue>>();

  setItem(list, idx, std::move(value));

  push(stack, std::move(list));
}
} // namespace torch::jit
