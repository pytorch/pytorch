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
const auto abs_script = R"JIT(
  def forward(self, a):
    return a.abs().clone()
)JIT";

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
    return z.clone()
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

const auto reshape_inplace_script_1 = R"JIT(
  def forward(self, inp: Tensor, shape: List[int], flag: bool):
    if flag:
      a = inp + inp
      b = a.reshape(shape)
      c = b.sigmoid()
    else:
      a = inp * inp
      b = a.sigmoid_()
      c = b.reshape(shape)
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

const auto detach_script_0 = R"JIT(
  def forward(self, input: Tensor):
      a = input.detach()
      return input is a
)JIT";

const auto detach_script_1 = R"JIT(
  def forward(self, input: Tensor):
      a = input.detach()
      return a.clone()
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

const auto expand_as_script = R"JIT(
  def forward(self, input: Tensor, other:Tensor):
      a = input.expand_as(other)
      return a.clone()
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

const auto mul_tensor = R"JIT(
  def forward(self, a: Tensor, b: Tensor):
      return torch.mul(a, b).clone()
)JIT";

const auto mul_scalar = R"JIT(
  def forward(self, a: Tensor, b: int):
      return torch.mul(a, b).clone()
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

const auto nan_to_num_script = R"JIT(
  def forward(self, a: Tensor, nan: float, posinf: float, neginf: float):
      return torch.nan_to_num(a, nan, posinf, neginf).clone()
)JIT";

const auto stack_dim = R"JIT(
  def forward(self, a: Tensor, b: Tensor, dim: int):
      return torch.stack((a, b), dim = dim).clone()
)JIT";

const auto stack_three = R"JIT(
  def forward(self, a: Tensor, b: Tensor, c: Tensor):
      return torch.stack((a, b, c)).clone()
)JIT";

const auto relu_script = R"JIT(
  def forward(self, a: Tensor):
      return torch.relu(a).clone()
)JIT";

const auto tanh_script = R"JIT(
  def forward(self, a):
      return torch.tanh(a).clone()
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

const auto full_script = R"JIT(
  def forward(self,
              size: List[int],
              fill_value: int,
              dtype: Optional[int],
              layout: Optional[int],
              device: Optional[Device],
              pin_memory: Optional[bool]):
      a = torch.full(size,
                     fill_value,
                     dtype=dtype,
                     layout=layout,
                     device=device,
                     pin_memory=pin_memory)
      return (a.clone())
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

const auto linear_script = R"JIT(
  def forward(self, inp: Tensor, weights: Tensor, bias: Optional[Tensor]) -> Tensor:
      return torch.linear(inp, weights, bias).clone()
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

const auto var_cat_script = R"JIT(
  def forward(self, inp1: Tensor, inp2: Tensor, dim: int):
   return torch.cat([inp1, inp2], dim).clone()
)JIT";

const auto var_stack_script = R"JIT(
  def forward(self, inp1: Tensor, inp2: Tensor, dim: int):
   return torch.stack([inp1, inp2], dim).clone()
)JIT";

const auto isinstance_int_script = R"JIT(
  def forward(self, a: Any):
      return isinstance(a, int)
)JIT";

const auto isinstance_tensor_script = R"JIT(
  def forward(self, a: Any):
      return isinstance(a, torch.Tensor)
)JIT";

const auto isinstance_many_types_script = R"JIT(
  def forward(self, a: Any):
      return isinstance(a, (bool, int))
)JIT";

const auto typecheck_ir = R"IR(
graph(%a.1 : Tensor,
      %b.1 : Tensor):
  %t0 : Float(2, 2, strides=[2, 1], device=cpu), %t1 : Float(3, 3, strides=[3, 1]), %type_matched : bool = prim::TypeCheck[types=[Float(2, 2, strides=[2, 1], device=cpu), Float(3, 3, strides=[3, 1])]](%a.1, %b.1)
  return (%t0, %t1, %type_matched)
)IR";

const auto index_without_none_script = R"JIT(
  def forward(self, a: Tensor, idx: Tensor):
      return a[idx].clone()
)JIT";

const auto index_with_none_script = R"JIT(
  def forward(self, a: Tensor, idx: Tensor):
      return a[idx, None].clone()
)JIT";

const auto index_with_two_tensors_script = R"JIT(
  def forward(self, a: Tensor, idx_a: Tensor, idx_b: Tensor):
      return a[idx_a, idx_b].clone()
)JIT";

const auto clamp_min_int_script = R"JIT(
  def forward(self, a: Tensor, b: int):
      return torch.clamp_min(a, b).clone()
)JIT";

const auto clamp_min_float_script = R"JIT(
  def forward(self, a: Tensor, b: float):
      return torch.clamp_min(a, b).clone()
)JIT";

const auto argmin_script = R"JIT(
  def forward(self, a: Tensor):
      return torch.argmin(a).clone()
)JIT";

const auto argmin_with_dim_script = R"JIT(
  def forward(self, a: Tensor, dim: int):
      return torch.argmin(a, dim).clone()
)JIT";

const auto argmin_with_keep_dim_script = R"JIT(
  def forward(self, a: Tensor, dim: int):
      return torch.argmin(a, dim, True).clone()
)JIT";

const auto softmax_script = R"JIT(
  def forward(self, a: Tensor, dim: int):
      return torch.softmax(a, dim).clone()
)JIT";

const auto softmax_script_with_dtype = R"JIT(
  def forward(self, a: Tensor, dim: int, dtype: int):
      return torch.softmax(a, dim, dtype=dtype).clone()
)JIT";

const auto getitem_dict_tensor_script = R"JIT(
  def forward(self, key: Tensor):
      d = {key: 1}
      return d[key]
)JIT";

const auto getitem_dict_int_script = R"JIT(
  def forward(self, key: int):
      d = {key: 1}
      return d[key]
)JIT";

const auto getitem_dict_str_script = R"JIT(
  def forward(self, key: str):
      d = {key: 1}
      return d[key]
)JIT";

const auto getitem_list_int_script = R"JIT(
  def forward(self, idx: int):
      lst = [1, 2, 3]
      return lst[idx]
)JIT";

const auto getitem_list_tensor_script = R"JIT(
  def forward(self, tensor: Tensor, idx: int):
      lst = [tensor, tensor]
      return lst[idx]
)JIT";

const auto transpose_script = R"JIT(
  def forward(self, a: Tensor, dim1: int, dim2: int):
      return torch.transpose(a, dim1, dim2).clone()
)JIT";

const auto permute_script = R"JIT(
  def forward(self, a: Tensor, dims: List[int]):
      return torch.permute(a, dims).clone()
)JIT";

const auto slice_script = R"JIT(
  def forward(self, a: Tensor, dim: int, start: int, end: int, step: int):
    return a.slice(dim, start, end, step).clone()
)JIT";

const auto narrow_with_int_script = R"JIT(
  def forward(self, a: Tensor, dim: int, start: int, length: int):
      return a.narrow(dim, start, length).clone()
)JIT";

const auto two_tuple_unpack_script = R"JIT(
  def forward(self, tup: Tuple[Tensor, Tensor]):
      a, b = tup
      return (a, b)
)JIT";

const auto three_tuple_unpack_script = R"JIT(
  def forward(self, tup: Tuple[Tensor, Tensor, Tensor]):
      a, b, c = tup
      return (a, b, c)
)JIT";

const auto append_int_script = R"JIT(
  def forward(self, a: int):
      lst = [1, 2, 3]
      lst.append(a)
      return lst
)JIT";

const auto append_tensor_script = R"JIT(
  def forward(self, a: Tensor):
      lst = []
      lst.append(a)
      return lst
)JIT";

const auto nonzero_tensor = R"JIT(
  def forward(self, input: Tensor):
      a = torch.nonzero(input).clone()
      return (a)
)JIT";

const std::string quantize_script = R"IR(
  graph(%input: Tensor, %weights: Tensor):
      %scale: float = prim::Constant[value=1.]()
      %zero_point: int = prim::Constant[value=1]()
      %bias: None = prim::Constant()
      %packed_params = quantized::linear_prepack(%weights, %bias)
      %1254 = quantized::linear(%input, %packed_params, %scale, %zero_point)
      %1249: Tensor = aten::dequantize(%1254)
      return (%1249)
)IR";

const auto fmod_tensor = R"JIT(
  def forward(self, a: Tensor, b: Tensor):
      return torch.fmod(a, b).clone()
)JIT";

const auto fmod_scalar = R"JIT(
  def forward(self, a: Tensor, b: int):
      return torch.fmod(a, b).clone()
)JIT";

const std::string embedding_bag_byte_prepack_script = R"IR(
  graph(%input: Tensor):
      %none : None = prim::Constant()
      %output: Tensor = quantized::embedding_bag_byte_prepack(%input)
      %res: Tensor = aten::clone(%output, %none)
      return (%res)
)IR";

const auto linalg_norm_ord_scalar = R"JIT(
  def forward(self, a: Tensor, ord: int, dim: List[int], keepdim: bool, dtype: int):
      return torch.linalg_norm(a, ord, dim, keepdim, dtype=dtype).clone()
)JIT";

const auto linalg_norm_ord_str = R"JIT(
  def forward(self, a: Tensor, ord: str, dim: List[int], keepdim: bool, dtype: int):
      return torch.linalg_norm(a, ord, dim, keepdim, dtype=dtype).clone()
)JIT";

const std::string cat_script = R"IR(
  graph(%a: Tensor, %b: Tensor, %dim: int):
      %ten_list: Tensor[] = prim::ListConstruct(%a, %b)
      %1 : int = prim::Constant[value=0]()
      %2 : int = prim::Constant[value=1]()
      %3 : int = prim::Constant[value=1]()
      %ten_list2 : Tensor[] = aten::slice(%ten_list, %1, %2, %3)
      %ret: Tensor = aten::cat(%ten_list2, %dim)
      return (%ret)
)IR";

const auto cumsum_script = R"JIT(
   def forward(self, a: Tensor, dim: int):
      return torch.cumsum(a, dim).clone()
)JIT";

const auto cumsum_script_dtype = R"JIT(
   def forward(self, a: Tensor, dim: int, dtype: int):
      return torch.cumsum(a, dim, dtype=dtype).clone()
)JIT";

const std::string signed_log1p_script = R"IR(
  graph(%input):
      %0 : Tensor = aten::sign(%input)
      %1 : Tensor = aten::abs(%input)
      %2 : Tensor = aten::log1p(%1)
      %3 : Tensor = aten::mul(%0, %2)
      %none : NoneType = prim::Constant()
      %res : Tensor = aten::clone(%3, %none)
      return (%res)
)IR";

const auto getitem_immutable_input_dict_script = R"JIT(
  def forward(self, input: Dict[int, Tensor]):
      a = input[0]
      b = input[1]
      c = a + b
      return c.clone()
)JIT";

const auto getitem_mutable_input_dict_script = R"JIT(
  def forward(self, input: Dict[int, Tensor]):
      a = input[0]
      input[1] = a
      b = input[1]
      c = a + b
      return c.clone()
)JIT";

const auto var_tuple_unpack_script = R"JIT(
  def forward(self, input_0: Tuple[Tensor, Tensor], input_1: Tuple[int, int]):
      a, b = input_0
      c, d = input_1
      res = a * c + b * d
      return res.clone()
)JIT";

const auto var_tuple_unpack_not_applied_script = R"JIT(
  def forward(self, input_0: Tuple[Tensor, Tensor], input_1: Tuple[int, int]):
      a, b = input_0
      x = a + b
      c, d = input_1
      res = a * c + b * d + x
      return res.clone()
)JIT";
