#pragma once

#include <torch/torch.h>

const auto list_construct_script = R"JIT(
  def forward(self, a, b):
    return [a, b]
)JIT";

const auto list_construct_script_2 = R"JIT(
  def forward(self, a, b):
    c = a + a
    return [c, c]
)JIT";

const auto list_construct_script_3 = R"JIT(
  def forward(self, a, b):
    c = a + a
    return [c, c.flatten()]
)JIT";

const auto list_unpack_script = R"JIT(
  def forward(self, a, b):
    c = [a, b]
    x, y = c
    z = x + y
    return z
)JIT";

const auto list_unpack_script_2 = R"JIT(
  def forward(self, a, b):
    c = [a, b]
    x, y = c
    z = (x, y)
    return z
)JIT";

const auto tuple_construct_script = R"JIT(
  def forward(self, a, b):
    return (a, b)
)JIT";

const auto tuple_construct_script_2 = R"JIT(
  def forward(self, a, b):
    return (a.flatten(), b)
)JIT";

const auto add_script = R"JIT(
  def forward(self, a, b):
      return a + b
)JIT";

const auto reshape_script_1 = R"JIT(
  def forward(self, a: Tensor, shape: List[int]):
      b = a.reshape(shape)
      return b + b
)JIT";

const auto reshape_script_2 = R"JIT(
  def forward(self, a: Tensor, shape: List[int]):
      b = a.transpose(0, 1)
      return b.reshape(shape)
)JIT";

const auto reshape_script_3 = R"JIT(
  def forward(self, inp: Tensor, shape: List[int]):
      a = inp + inp
      b = a.reshape(shape)
      c = a.reshape(shape)
      d = c + c
      e = d + d
      f = e * e
      g = f * f
      return b.reshape(shape), g
)JIT";

// exercise reshape_copy and flatten_copy
const auto reshape_script_4 = R"JIT(
  def forward(self, inp: Tensor, shape: List[int]):
      k = inp + inp
      a = k + k
      b = a.reshape(shape)
      c = a.flatten().reshape(shape)
      return b + c
)JIT";

// exercise reshape_copy
const auto reshape_script_5 = R"JIT(
  def forward(self, inp: Tensor, shape: List[int]):
      a = inp + inp
      b = a.reshape(shape)
      c = a.reshape(shape).relu()
      d = c + c
      e = d + d
      f = e * e
      g = f * f
      return g
)JIT";

const auto reshape_inplace_script = R"JIT(
  def forward(self, inp: Tensor, shape: List[int]):
      a = inp + inp
      b = a.reshape(shape)
      c = b.sigmoid_()
      d = c + c
      e = a + a
      f = b + b
      return (d, e, f)
)JIT";

const auto sigmoid_inplace_script = R"JIT(
  def forward(self, inp: Tensor, shape: List[int]):
      a = torch.sigmoid(inp, out=inp)
      return (a)
)JIT";

const auto sigmoid_out_script = R"JIT(
  def forward(self, inp: Tensor, shape: List[int]):
      a = inp + inp
      b = torch.sigmoid(inp, out=a)
      return (b)
)JIT";

const auto logit_script_1 = R"JIT(
  def forward(self, inp: Tensor):
      a = torch.logit(inp)
      return (a)
)JIT";

const auto logit_script_2 = R"JIT(
  def forward(self, inp: Tensor):
      a = torch.logit(inp, 1e-6)
      return (a)
)JIT";

const auto logit_script_3 = R"JIT(
  def forward(self, inp: Tensor, eps: float):
      a = torch.logit(inp, eps)
      return (a)
)JIT";

// b is in_contiguous
const auto reshape_incontiguous_script = R"JIT(
  def forward(self, a: Tensor, shape: List[int]):
      b = a.transpose(0, 1)
      c = b.reshape(shape)
      c = c.relu()
      return (c)
)JIT";

// exercise flatten_copy
const auto flatten_script_1 = R"JIT(
  def forward(self, a: Tensor, start_dim: int, end_dim: int):
      b = a * a
      c = torch.flatten(b, start_dim, end_dim)
      d = torch.relu(c)
      return d
)JIT";

const auto flatten_script_2 = R"JIT(
  def forward(self, a: Tensor, start_dim: int, end_dim: int):
      b = a.transpose(0, 1)
      return torch.flatten(b, start_dim, end_dim)
)JIT";

const auto clone_script_0 = R"JIT(
  def forward(self, input):
      return torch.clone(input)
)JIT";

const auto clone_script_1 = R"JIT(
  def forward(self, input: Tensor, memory_format: int):
      return torch.clone(input, memory_format=memory_format)
)JIT";

const auto aten_sum = R"JIT(
  def forward(self, input):
      return torch.sum(input)
)JIT";

const auto aten_sum_0 = R"JIT(
  def forward(self, input):
      return torch.sum(input, 0)
)JIT";

const auto aten_sum_1 = R"JIT(
  def forward(self, input):
      return torch.sum(input, 1)
)JIT";

const auto aten_sum_0_true = R"JIT(
  def forward(self, input):
      return torch.sum(input, 0, True)
)JIT";

const auto aten_sum_1_true = R"JIT(
  def forward(self, input):
      return torch.sum(input, 1, True)
)JIT";

const auto pow_script_ten_sca = R"JIT(
  def forward(self, input : Tensor, exponent : int):
      return torch.pow(input, exponent)
)JIT";

const auto pow_script_ten_ten = R"JIT(
  def forward(self, input : Tensor, exponent : Tensor):
      return torch.pow(input, exponent)
)JIT";

const auto pow_script_sca_ten = R"JIT(
  def forward(self, input : int, exponent : Tensor):
      return torch.pow(input, exponent)
)JIT";

const auto to_script_0 = R"JIT(
  def forward(self, input: Tensor, dtype: int, non_blocking: bool, copy: bool, memory_format: int):
      return torch.to(input, dtype, non_blocking, copy, memory_format)
)JIT";

const auto to_script_1 = R"JIT(
  def forward(self, input:Tensor, dtype: int, non_blocking: bool, copy: bool):
      return torch.to(input, dtype, non_blocking, copy)
)JIT";

const auto to_script_2 = R"JIT(
  def forward(self, input:Tensor, other: Tensor, non_blocking: bool, copy: bool, memory_format: int):
      return torch.to(input, other, non_blocking, copy, memory_format)
)JIT";

const std::string embedding_bag_default = R"JIT(
  def forward(self, a: Tensor, b: Tensor, c: Tensor):
      return torch.embedding_bag(a, b, c)
)JIT";

const std::string embedding_bag_mean = R"JIT(
  def forward(self, a: Tensor, b: Tensor, c: Tensor):
      return torch.embedding_bag(a, b, c, False, 1)
)JIT";

const std::string embedding_bag_max = R"JIT(
  def forward(self, a: Tensor, b: Tensor, c: Tensor):
      return torch.embedding_bag(a, b, c, False, 2)
)JIT";

const std::string embedding_bag_sum_last_offset = R"JIT(
  def forward(self, a: Tensor, b: Tensor, c: Tensor):
      return torch.embedding_bag(a, b, c, False, 0, False, None, True)
)JIT";

const std::string embedding_bag_mean_last_offset = R"JIT(
  def forward(self, a: Tensor, b: Tensor, c: Tensor):
      return torch.embedding_bag(a, b, c, False, 1, False, None, True)
)JIT";

const std::string embedding_bag_max_last_offset = R"JIT(
  def forward(self, a: Tensor, b: Tensor, c: Tensor):
      return torch.embedding_bag(a, b, c, False, 2, False, None, True)
)JIT";

const auto div_tensor = R"JIT(
  def forward(self, a: Tensor, b: Tensor):
      return torch.div(a, b)
)JIT";

const auto div_scalar = R"JIT(
  def forward(self, a: Tensor, b: int):
      return torch.div(a, b)
)JIT";

const auto div_tensor_mode = R"JIT(
  def forward(self, a: Tensor, b: Tensor, c: str):
      return torch.div(a, b, rounding_mode=c)
)JIT";

const auto div_scalar_mode = R"JIT(
  def forward(self, a: Tensor, b: float, c: str):
      return torch.div(a, b, rounding_mode=c)
)JIT";

const auto sub_tensor = R"JIT(
  def forward(self, a: Tensor, b: Tensor):
      return torch.sub(a, b)
)JIT";

const auto sub_scalar = R"JIT(
  def forward(self, a: Tensor, b: int):
      return torch.sub(a, b)
)JIT";

const auto sub_tensor_alpha = R"JIT(
  def forward(self, a: Tensor, b: Tensor, c: float):
      return torch.sub(a, b, alpha=c)
)JIT";

const auto sub_scalar_alpha = R"JIT(
  def forward(self, a: Tensor, b: float, c: int):
      return torch.sub(a, b, alpha=c)
)JIT";

const std::string layer_norm_with_weights = R"JIT(
  def forward(self, input: Tensor, normalized_shape: List[int], weight: Tensor, bias: Tensor):
      return torch.layer_norm(input, normalized_shape, weight, bias, 1e-05, False)
)JIT";

const std::string layer_norm_without_weights = R"JIT(
  def forward(self, input: Tensor, normalized_shape: List[int]):
      return torch.layer_norm(input, normalized_shape, None, None, 1e-05, False)
)JIT";

const auto norm_2arg = R"JIT(
  def forward(self, a: Tensor, p: int):
      return torch.norm(a, p)
)JIT";

const auto norm_3arg = R"JIT(
  def forward(self, a: Tensor, p: int, dtype: int):
      return torch.norm(a, p, dtype=dtype)
)JIT";

const auto norm_4arg = R"JIT(
  def forward(self, a: Tensor, p: int, dim: List[int], keepdim: bool):
      return torch.norm(a, p, dim, keepdim)
)JIT";

const auto norm_5arg = R"JIT(
  def forward(self, a: Tensor, p: int, dim: List[int], keepdim: bool, dtype: int):
      return torch.norm(a, p, dim, keepdim, dtype=dtype)
)JIT";

const auto aten_matmul = R"JIT(
  def forward(self, a: Tensor, b: Tensor):
      return torch.matmul(a, b)
)JIT";

const std::string repeat = R"JIT(
  def forward(self, a: Tensor, repeats: List[int]):
      return torch.repeat(a, repeats)
)JIT";

const auto clamp_script_1 = R"JIT(
  def forward(self, inp: Tensor, min: int, max: int):
      a = torch.clamp(inp, min, max)
      return (a)
)JIT";

const auto clamp_script_2 = R"JIT(
  def forward(self, inp: Tensor, min: Tensor, max: Tensor):
      a = torch.clamp(inp, min, max)
      return (a)
)JIT";
