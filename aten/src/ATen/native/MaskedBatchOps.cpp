#include "ATen/ATen.h"

namespace at { namespace native {

// elementwise operators

#define IMPLEMENT_MASKEDBATCH_OPS_ELEMENTWISE_UNARY(op, opfn)  \
  std::tuple<Tensor, Tensor, Tensor> b_##op(TensorList self) { \
    if (self.size() != 3) {                                    \
      throw std::invalid_argument(                             \
          "3 tensor in input TensorList expected, found " +    \
          std::to_string(self.size()));                        \
    }                                                          \
    Tensor data = opfn(self[0]);                               \
    return std::make_tuple(data, self[1], self[2]);            \
  }

IMPLEMENT_MASKEDBATCH_OPS_ELEMENTWISE_UNARY(tanh, at::tanh)
IMPLEMENT_MASKEDBATCH_OPS_ELEMENTWISE_UNARY(relu, at::relu)
IMPLEMENT_MASKEDBATCH_OPS_ELEMENTWISE_UNARY(sigmoid, at::sigmoid)

#define IMPLEMENT_MASKEDBATCH_OPS_ELEMENTWISE_BINARY(op, opfn)             \
  std::tuple<Tensor, Tensor, Tensor> b_##op(                               \
      TensorList self, TensorList other) {                                 \
    if (self.size() != 3) {                                                \
      throw std::invalid_argument(                                         \
          "3 tensor in input TensorList expected, found " +                \
          std::to_string(self.size()));                                    \
    }                                                                      \
    if (other.size() == 3) {                                               \
      auto data = opfn(self[0], other[0]);                                 \
      auto mask = self[1] * other[1];                                      \
      auto dims = self[2].__or__(other[2]);                                \
      return std::make_tuple(data, mask, dims);                            \
    } else if (other.size() == 1) {                                        \
      auto data = opfn(self[0], other[0].unsqueeze(0).expand_as(self[0])); \
      return std::make_tuple(data, self[1], self[2]);                      \
    } else {                                                               \
      throw std::invalid_argument(                                         \
          "1 or 3 tensor in input 'other' TensorList expected, found " +   \
          std::to_string(other.size()));                                   \
    }                                                                      \
  }

IMPLEMENT_MASKEDBATCH_OPS_ELEMENTWISE_BINARY(add, at::add)
IMPLEMENT_MASKEDBATCH_OPS_ELEMENTWISE_BINARY(sub, at::sub)
IMPLEMENT_MASKEDBATCH_OPS_ELEMENTWISE_BINARY(mul, at::mul)

#define IMPLEMENT_MASKEDBATCH_OPS_ELEMENTWISE_BATCH_VALUE(op, opfn)          \
  std::tuple<Tensor, Tensor, Tensor> b_##op(TensorList self, Scalar other) { \
    if (self.size() != 3) {                                                  \
      throw std::invalid_argument(                                           \
          "3 tensor in input TensorList expected, found " +                  \
          std::to_string(self.size()));                                      \
    }                                                                        \
    auto data = opfn(self[0], other);                                        \
    return std::make_tuple(data, self[1], self[2]);                          \
  }

IMPLEMENT_MASKEDBATCH_OPS_ELEMENTWISE_BATCH_VALUE(add, at::add)
IMPLEMENT_MASKEDBATCH_OPS_ELEMENTWISE_BATCH_VALUE(sub, at::sub)
IMPLEMENT_MASKEDBATCH_OPS_ELEMENTWISE_BATCH_VALUE(mul, at::mul)

// tersor math operators

std::tuple<Tensor, Tensor, Tensor> b_mm(TensorList self, TensorList other) {
  if (self.size() != 3) {
    throw std::invalid_argument(
        "3 tensor in input TensorList expected, found " +
        std::to_string(self.size()));
  }
  if (other.size() == 3) {
    auto data = at::bmm(self[0] * self[1], other[0] * other[1]);
    auto mask = at::bmm(self[1].narrow(2, 0, 1), other[1].narrow(1, 0, 1));
    auto dims = at::cat(
        std::vector<at::Tensor>({self[2].narrow(0, 0, 1),
                                 other[2].narrow(0, 1, other[2].size(0) - 1)}));
    return std::make_tuple(data, mask, dims);
  } else if (other.size() == 1) {
    auto data = at::bmm(
        self[0] * self[1],
        other[0].unsqueeze(0).expand(
            {self[0].size(0), other[0].size(0), other[0].size(1)}));
    auto mask = self[1].narrow(2, 0, 1);
    auto dims = at::cat(std::vector<at::Tensor>(
        {self[2].narrow(0, 0, 1), self[2].type().tensor({1})}));
    return std::make_tuple(data, mask, dims);
  }
  throw std::invalid_argument(
      "1 or 3 tensor in input 'other' TensorList expected, found " +
      std::to_string(other.size()));
}

std::tuple<Tensor, Tensor, Tensor> b_matmul(TensorList self, TensorList other) {
  if (self.size() != 3) {
    throw std::invalid_argument(
        "3 tensor in input TensorList expected, found " +
        std::to_string(self.size()));
  }
  if (other.size() != 3) {
    throw std::invalid_argument(
        "3 tensor in input 'other' TensorList expected, found " +
        std::to_string(other.size()));
  }
  int d1 = self[2].size(0);
  int d2 = other[2].size(0);
  auto data1 = self[0] * self[1];
  auto data2 = other[0] * other[1];
  if (d1 == 1)
    data1 = data1.unsqueeze(-2);
  if (d2 == 1)
    data2 = data2.unsqueeze(-1);
  auto data = at::bmm(data1, data2);
  Tensor mask, dims;
  if (d1 == 1 && d2 == 1) {
    data = data.squeeze(-1).squeeze(-1);
    mask = self[1].narrow(1, 0, 1).squeeze(-1);
    dims = self[2].type().tensor();
  } else if (d1 == 2 && d2 == 1) {
    data = data.squeeze(-1);
    mask =
        at::bmm(self[1].narrow(2, 0, 1), other[1].narrow(1, 0, 1).unsqueeze(-1))
            .squeeze(-1);
    dims = self[2].narrow(0, 0, 1);
  } else if (d1 == 1 && d2 == 2) {
    data = data.squeeze(-2);
    mask =
        at::bmm(self[1].narrow(1, 0, 1).unsqueeze(-2), other[1].narrow(1, 0, 1))
            .squeeze(-2);
    dims = other[2].narrow(0, 1, d2 - 1);
  } else if (d1 == 2 and d2 == 2) {
    mask = at::bmm(self[1].narrow(2, 0, 1), other[1].narrow(1, 0, 1));
    dims = at::cat(std::vector<at::Tensor>(
        {self[2].narrow(0, 0, 1), other[2].narrow(0, 1, d2 - 1)}));
  } else {
    throw std::runtime_error(
        "matmul not implemented with batches of 3+D tensors");
  }
  return std::make_tuple(data, mask, dims);
}

// tensor shape operators

std::tuple<Tensor, Tensor, Tensor> b_contiguous(TensorList self) {
  if (self.size() != 3) {
    throw std::invalid_argument(
        "3 tensor in input TensorList expected, found " +
        std::to_string(self.size()));
  }
  return std::make_tuple(self[0].contiguous(), self[1].contiguous(), self[2]);
}

// assumption: (sizes[i] == -1) === ith dimension is dynamic
std::tuple<Tensor, Tensor, Tensor> b_view(TensorList self, IntList sizes_new) {
  if (self.size() != 3) {
    throw std::invalid_argument(
        "3 tensor in input TensorList expected, found " +
        std::to_string(self.size()));
  }
  int bs = self[0].size(0);
  if (sizes_new[0] != 1 && sizes_new[0] != -1 && sizes_new[0] != bs)
    throw std::runtime_error("first dim in view must be 1, -1, or batch size");
  std::vector<long long> data_sizes = {bs}, mask_sizes = {bs};
  Tensor dims_new = at::zeros(self[2].type(), (sizes_new.size() - 1));
  for (int i = 1; i < int(sizes_new.size()); i++) {
    data_sizes.push_back(sizes_new[i]);
    mask_sizes.push_back(sizes_new[i] == -1 ? self[0].size(i) : 1);
    dims_new[i - 1] = (sizes_new[i] == -1);
  }
  return std::make_tuple(
      self[0].view(data_sizes), self[1].view(mask_sizes), dims_new);
}

// special operators

std::tuple<Tensor, Tensor, Tensor> b_update(TensorList self, TensorList other) {
  if (self.size() != 3) {
    throw std::invalid_argument(
        "3 tensor in input TensorList expected, found " +
        std::to_string(self.size()));
  }
  if (other.size() != 3) {
    throw std::invalid_argument(
        "3 tensor in input 'other' TensorList expected, found " +
        std::to_string(other.size()));
  }
  auto update_mask = other[1]; // TODO: maybe third arguments needed
  auto data = at::where(update_mask, other[0], self[0]);
  return std::make_tuple(data, update_mask, other[2]);
}

std::tuple<Tensor, Tensor, Tensor> b_synchronize(TensorList self) {
  if (self.size() != 3) {
    throw std::invalid_argument(
        "3 tensor in input TensorList expected, found " +
        std::to_string(self.size()));
  }
  auto mask_new = self[1] + (1 - self[1]);
  return std::make_tuple(self[0], mask_new, self[2]);
}
}} // namespace at::native
