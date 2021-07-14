#pragma once

#include <torch/torch.h>

/*
 When adding a test for an operator implemented in static runtime, there are
 several things that you need to pay attention to: 1) if the op is an out
 variant, in the test script of the op,
 instead of:
    def forward(self, input):
      return myop(input)

  do:
    def forward(self, input):
      return myop(input).clone()

 This makes sure that the output of myop is managed by the memory planner and
 exercise the code path in the op impl that otherwise doesn't get exercised. The
 output of the model is not managed by the memory planner, because it needs to
 be returned to the client.

 2) for view ops such as aten::reshape or aten::to, if you want it to be
 replaced by the copy version with the ReplaceWithCopy pass in passes.h, you
 also want to make sure its output is not returned as the model output. The
 reason is that ReplaceWithCopy only replaces the op whoes output is not an
 alias of the model output.

*/
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
      c = a + b
      return (c.clone())
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
  def forward(self, inp: Tensor):
      a = torch.sigmoid(inp, out=inp).clone()
      return (a)
)JIT";

const auto sigmoid_out_script = R"JIT(
  def forward(self, inp: Tensor):
      a = inp + inp
      b = torch.sigmoid(inp, out=a).clone()
      return (b)
)JIT";

const auto sigmoid_script = R"JIT(
  def forward(self, inp: Tensor):
      b = torch.sigmoid(inp).clone()
      return (b)
)JIT";

// no nnc
const auto logit_script_1 = R"JIT(
  def forward(self, inp: Tensor):
      a = torch.logit(inp).clone()
      return (a)
)JIT";

// with nnc
const auto logit_script_2 = R"JIT(
  def forward(self, inp: Tensor):
      a = torch.logit(inp, 1e-6).clone()
      return (a)
)JIT";

// no nnc
const auto logit_script_3 = R"JIT(
  def forward(self, inp: Tensor, eps: float):
      a = torch.logit(inp, eps).clone()
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
      return torch.flatten(b, start_dim, end_dim).clone()
)JIT";

const auto clone_script_0 = R"JIT(
  def forward(self, input):
      a = torch.clone(input)
      return (a + a)
)JIT";

const auto clone_script_1 = R"JIT(
  def forward(self, input: Tensor, memory_format: int):
      a = torch.clone(input, memory_format=memory_format)
      return (a + a)
)JIT";

const auto aten_sum = R"JIT(
  def forward(self, input):
      return torch.sum(input).clone()
)JIT";

const auto aten_sum_0 = R"JIT(
  def forward(self, input):
      return torch.sum(input, 0).clone()
)JIT";

const auto aten_sum_1 = R"JIT(
  def forward(self, input):
      return torch.sum(input, 1).clone()
)JIT";

const auto aten_sum_0_true = R"JIT(
  def forward(self, input):
      return torch.sum(input, 0, True).clone()
)JIT";

const auto aten_sum_1_true = R"JIT(
  def forward(self, input):
      return torch.sum(input, 1, True).clone()
)JIT";

const auto pow_script_ten_sca = R"JIT(
  def forward(self, input : Tensor, exponent : int):
      return torch.pow(input, exponent).clone()
)JIT";

const auto pow_script_ten_ten = R"JIT(
  def forward(self, input : Tensor, exponent : Tensor):
      return torch.pow(input, exponent).clone()
)JIT";

const auto pow_script_sca_ten = R"JIT(
  def forward(self, input : int, exponent : Tensor):
      return torch.pow(input, exponent).clone()
)JIT";

// to.dtype
const auto to_script_0 = R"JIT(
  def forward(self, input: Tensor, dtype: int, non_blocking: bool, copy: bool, memory_format: int):
      a = input + input
      return torch.to(a, dtype, non_blocking, copy, memory_format).clone()
)JIT";

// to.dtype, strided
const auto to_script_1 = R"JIT(
  def forward(self, input: Tensor, dtype: int, non_blocking: bool, copy: bool, memory_format: int):
      b = input.permute(0, 2, 3, 1)
      return torch.to(b, dtype, non_blocking, copy, memory_format).clone()
)JIT";

// to.prim_dtype
const auto to_script_2 = R"JIT(
  def forward(self, input:Tensor, dtype: int, non_blocking: bool, copy: bool):
      a = input + input
      return torch.to(a, dtype, non_blocking, copy).clone()
)JIT";

// to.other
const auto to_script_3 = R"JIT(
  def forward(self, input:Tensor, other: Tensor, non_blocking: bool, copy: bool, memory_format: int):
      a = input + input
      return torch.to(a, other, non_blocking, copy, memory_format).clone()
)JIT";

// if input is float tensor, b could be alias of a
const auto to_script_4 = R"JIT(
  def forward(self, input:Tensor):
      a = input + input
      b = a.float()
      c = b * b
      return (c)
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

const auto sign_tensor = R"JIT(
  def forward(self, input: Tensor):
      return torch.sign(input).clone()
)JIT";

const auto div_tensor = R"JIT(
  def forward(self, a: Tensor, b: Tensor):
      return torch.div(a, b).clone()
)JIT";

const auto div_scalar = R"JIT(
  def forward(self, a: Tensor, b: int):
      return torch.div(a, b).clone()
)JIT";

const auto div_tensor_mode = R"JIT(
  def forward(self, a: Tensor, b: Tensor, c: str):
      return torch.div(a, b, rounding_mode=c).clone()
)JIT";

const auto div_scalar_mode = R"JIT(
  def forward(self, a: Tensor, b: float, c: str):
      return torch.div(a, b, rounding_mode=c).clone()
)JIT";

const auto log_tensor = R"JIT(
  def forward(self, inp: Tensor):
      a = torch.log(inp).clone()
      return (a)
)JIT";

const auto sub_tensor = R"JIT(
  def forward(self, a: Tensor, b: Tensor):
      return torch.sub(a, b).clone()
)JIT";

const auto sub_scalar = R"JIT(
  def forward(self, a: Tensor, b: int):
      return torch.sub(a, b).clone()
)JIT";

const auto sub_tensor_alpha = R"JIT(
  def forward(self, a: Tensor, b: Tensor, c: float):
      return torch.sub(a, b, alpha=c).clone()
)JIT";

const auto sub_scalar_alpha = R"JIT(
  def forward(self, a: Tensor, b: float, c: int):
      return torch.sub(a, b, alpha=c).clone()
)JIT";

const std::string layer_norm_with_weights = R"JIT(
  def forward(self, input: Tensor, normalized_shape: List[int], weight: Tensor, bias: Tensor):
      return torch.layer_norm(input, normalized_shape, weight, bias, 1e-05, False).clone()
)JIT";

const std::string layer_norm_without_weights = R"JIT(
  def forward(self, input: Tensor, normalized_shape: List[int]):
      return torch.layer_norm(input, normalized_shape, None, None, 1e-05, False).clone()
)JIT";

const auto norm_2arg = R"JIT(
  def forward(self, a: Tensor, p: int):
      return torch.norm(a, p).clone()
)JIT";

const auto norm_3arg = R"JIT(
  def forward(self, a: Tensor, p: int, dtype: int):
      return torch.norm(a, p, dtype=dtype).clone()
)JIT";

const auto norm_4arg = R"JIT(
  def forward(self, a: Tensor, p: int, dim: List[int], keepdim: bool):
      return torch.norm(a, p, dim, keepdim).clone()
)JIT";

const auto norm_5arg = R"JIT(
  def forward(self, a: Tensor, p: int, dim: List[int], keepdim: bool, dtype: int):
      return torch.norm(a, p, dim, keepdim, dtype=dtype).clone()
)JIT";

const auto aten_matmul = R"JIT(
  def forward(self, a: Tensor, b: Tensor):
      return torch.matmul(a, b).clone()
)JIT";

const std::string repeat = R"JIT(
  def forward(self, a: Tensor, repeats: List[int]):
      return torch.repeat(a, repeats).clone()
)JIT";

const auto clamp_script_1 = R"JIT(
  def forward(self, inp: Tensor, min: int, max: int):
      a = torch.clamp(inp, min, max).clone()
      return (a)
)JIT";

const auto clamp_script_2 = R"JIT(
  def forward(self, inp: Tensor, min: Tensor, max: Tensor):
      a = torch.clamp(inp, min, max).clone()
      return (a)
)JIT";

const auto full_like_script = R"JIT(
  def forward(self,
              a: Tensor,
              fill_value: int,
              dtype: Optional[int],
              layout: Optional[int],
              device: Optional[Device],
              pin_memory: Optional[bool],
              memory_format: Optional[int]):
      b = torch.full_like(a,
                          fill_value,
                          dtype=dtype,
                          layout=layout,
                          device=device,
                          pin_memory=pin_memory,
                          memory_format=memory_format)
      return (b.clone())
)JIT";

// dict of tuple of list
const auto nested_output_script_0 = R"JIT(
  def forward(self, a, b):
    c = (a + b).relu().half().float()
    d = a.flatten().half() * b.flatten().half()
    e = d.float().relu()
    f = ([c], [d])
    g = ([e], [f])
    return ({"prediction":(f, d)})
)JIT";

// tuple of lists
const auto nested_output_script_1 = R"JIT(
  def forward(self, a, b):
    c = (a + b).relu().half().float()
    d = a.flatten().half() * b.flatten().half()
    e = d.float().relu()
    f = [c]
    g = [e]
    return (f, g)
)JIT";

// list of tuple of dict
const auto nested_output_script_2 = R"JIT(
  def forward(self, a, b):
    c = (a + b).relu().half().float()
    d = b * c
    e = a.flatten().half() * b.flatten().half()
    f = e.float().relu()
    g = ({"d": d}, {"b": b})
    h = ({"e": e}, {"f": f})
    return [g, h]
)JIT";

// lit of dict
const auto nested_output_script_3 = R"JIT(
  def forward(self, a, b):
    c = (a + b).relu().half().float()
    d = b * c
    e = a.flatten().half() * b.flatten().half()
    f = e.float().relu()
    g = {"d": d, "b": b}
    h = {"e": e, "f": f}
    return [g, h]
)JIT";

const auto bmm_script = R"JIT(
  def forward(self, inp: Tensor, mat2: Tensor):
   return torch.bmm(inp, mat2).clone()
)JIT";

const auto addmm_script = R"JIT(
  def forward(self, inp: Tensor, mat1: Tensor, mat2: Tensor, beta: float, alpha: float):
   return torch.addmm(inp, mat1, mat2, alpha=alpha, beta=beta).clone()
)JIT";

const auto if_script = R"JIT(
  def forward(self, a: Tensor, b: Tensor, x: bool):
    c = (a + b).relu().half().float()
    d = b * c
    if x:
      e = a.flatten().half() * b.flatten().half()
    else:
      e = a.flatten().half() + b.flatten().half()
    f = e.float().relu()
    g = {"d": d, "b": b}
    h = {"e": e, "f": f}
    return [g, h]
)JIT";
