#pragma once

#include "ATen/Tensor.h"
#include <functional>
#include <sstream>

namespace at {

// avoid copy-construction of Tensor by using a reference_wrapper.
inline void check_defined(std::initializer_list<std::reference_wrapper<const Tensor>> tensors, const char *api_name) {
  for (auto& t : tensors) {
    if (!t.get().defined()) {
      runtime_error("%s(...) called with an undefined Tensor", api_name);
    }
  }
}

inline std::tuple<Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand) {
  if (tensor.sizes().equals(to_expand.sizes())) {
    return std::make_tuple(to_expand);
  }

  return std::make_tuple(to_expand.expand(tensor.sizes()));
}

inline std::tuple<Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand, const char *api_name) {
  check_defined({tensor, to_expand}, api_name);
  return expand_inplace(tensor, to_expand);
}

inline std::tuple<Tensor, Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand1, const Tensor &to_expand2) {
  if (tensor.sizes().equals(to_expand1.sizes()) && tensor.sizes().equals((to_expand2.sizes()))) {
    return std::make_tuple(to_expand1, to_expand2);
  }

  return std::make_tuple(to_expand1.expand(tensor.sizes()), to_expand2.expand(tensor.sizes()));
}

inline std::tuple<Tensor, Tensor> expand_inplace(const Tensor &tensor, const Tensor &to_expand1, const Tensor &to_expand2,
                                                 const char *api_name) {
  check_defined({tensor, to_expand1, to_expand2}, api_name);
  return expand_inplace(tensor, to_expand1, to_expand2);
}

inline std::vector<int64_t> infer_size2(IntList a, IntList b) {
  auto dimsA = a.size();
  auto dimsB = b.size();
  ptrdiff_t ndim = dimsA > dimsB ? dimsA : dimsB;
  std::vector<int64_t> expandedSizes(ndim);

  for (long i = ndim - 1; i >= 0; --i) {
    long offset = ndim - 1 - i;
    long dimA = dimsA - 1 - offset;
    long dimB = dimsB - 1 - offset;
    long sizeA = (dimA >= 0) ? a[dimA] : 1;
    long sizeB = (dimB >= 0) ? b[dimB] : 1;
    if (sizeA == sizeB || sizeA == 1 || sizeB == 1) {
      expandedSizes[i] = std::max(sizeA, sizeB);
    } else {
      std::ostringstream oss;
      oss << "The size of tensor a (" << sizeA << ") must match the size of tensor b ("
          << sizeB << ") at non-singleton dimension " << i;
      throw std::runtime_error(oss.str());
    }
  }

  return expandedSizes;
}

inline std::tuple<Tensor, Tensor> expand_outplace(const Tensor &to_expand1, const Tensor &to_expand2) {
  if (to_expand1.sizes().equals(to_expand2.sizes())) {
    return std::make_tuple(to_expand1, to_expand2);
  }

  auto expanded_size = infer_size2(to_expand1.sizes(), to_expand2.sizes());
  return std::make_tuple(to_expand1.expand(expanded_size), to_expand2.expand(expanded_size));
}

inline std::tuple<Tensor, Tensor> expand_outplace(const Tensor &to_expand1, const Tensor &to_expand2, const char *api_name) {
  check_defined({to_expand1, to_expand2}, api_name);
  return expand_outplace(to_expand1, to_expand2);
}

inline std::tuple<Tensor, Tensor, Tensor> expand_outplace(const Tensor &to_expand1,
                                                          const Tensor &to_expand2,
                                                          const Tensor &to_expand3) {
  if (to_expand1.sizes().equals(to_expand2.sizes()) && to_expand1.sizes().equals(to_expand3.sizes())) {
    return std::make_tuple(to_expand1, to_expand2, to_expand3);
  }

  auto expanded_size12 = infer_size2(to_expand1.sizes(), to_expand2.sizes());
  auto expanded_size = infer_size2(expanded_size12, to_expand3.sizes());
  return std::make_tuple(to_expand1.expand(expanded_size), to_expand2.expand(expanded_size), to_expand3.expand(expanded_size));
}

inline std::tuple<Tensor, Tensor, Tensor> expand_outplace(const Tensor &to_expand1,
                                                          const Tensor &to_expand2,
                                                          const Tensor &to_expand3,
                                                          const char *api_name) {
  check_defined({to_expand1, to_expand2, to_expand3}, api_name);
  return expand_outplace(to_expand1, to_expand2, to_expand3);
}

inline std::tuple<Tensor> expand_size(const Tensor &to_expand, IntList sizes) {
  if(to_expand.sizes().equals(sizes)) {
    return std::make_tuple(to_expand);
  }

  return std::make_tuple(to_expand.expand(sizes));
}

inline std::tuple<Tensor> expand_size(const Tensor &to_expand, IntList sizes, const char *api_name) {
  check_defined({to_expand}, api_name);
  return expand_size(to_expand, sizes);
}

}
