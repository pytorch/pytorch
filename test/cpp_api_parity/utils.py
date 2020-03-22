from collections import namedtuple
import unittest

import torch
import torch.utils.cpp_extension
import torch.testing._internal.common_nn as common_nn

TorchNNModuleTestParams = namedtuple(
  'TorchNNModuleTestParams',
  [
    'module_name',
    'module_variant_name',
    'test_instance',
    'cpp_constructor_args',
    'arg_dict',
    'has_parity',
    'device',
    'cpp_tmp_folder',
  ]
)

TorchNNFunctionalTestParams = namedtuple(
  'TorchNNFunctionalTestParams',
  [
    'functional_name',
    'functional_variant_name',
    'test_instance',
    'cpp_function_call',
    'arg_dict',
    'has_parity',
    'device',
    'cpp_tmp_folder',
  ]
)

CppArg = namedtuple('CppArg', ['name', 'value'])

TORCH_NN_COMMON_TEST_HARNESS = """
#include <torch/script.h>

void write_ivalue_to_file(const torch::IValue& ivalue, const std::string& file_path) {
  auto bytes = torch::jit::pickle_save(ivalue);
  std::ofstream fout(file_path, std::ios::out | std::ios::binary);
  fout.write(bytes.data(), bytes.size());
  fout.close();
}

c10::Dict<std::string, torch::Tensor> load_dict_from_file(const std::string& file_path) {
  c10::Dict<std::string, torch::Tensor> arg_dict;
  auto arg_dict_module = torch::jit::load(file_path);
  for (const auto& p : arg_dict_module.named_buffers(/*recurse=*/false)) {
    arg_dict.insert(p.name, p.value);
  }
  return arg_dict;
}

// Generates rand tensor with non-equal values. This ensures that duplicate
// values won't be causing test failure for modules like MaxPooling.
// size should be small, otherwise randperm fails / long overflows.
torch::Tensor _rand_tensor_non_equal(torch::IntArrayRef size) {
  int64_t total = 1;
  for (int64_t elem : size) {
    total *= elem;
  }
  return torch::randperm(total).view(size).to(torch::kDouble);
}
"""

def compile_cpp_code_inline(name, cpp_sources, functions):
  cpp_module = torch.utils.cpp_extension.load_inline(
    name=name,
    cpp_sources=cpp_sources,
    functions=functions,
    verbose=False,
  )
  return cpp_module

def convert_to_list(python_input):
  if isinstance(python_input, torch.Tensor):
    return [python_input]
  else:
    return [tensor for tensor in python_input]

def set_python_tensors_requires_grad(python_tensors):
  return [tensor.requires_grad_(True) if tensor.dtype != torch.long else tensor for tensor in python_tensors]

def move_python_tensors_to_device(python_tensors, device):
  return [tensor.to(device) for tensor in python_tensors]

def has_test(unit_test_class, test_name):
  return hasattr(unit_test_class, test_name)

def add_test(unit_test_class, test_name, test_fn):
  if has_test(unit_test_class, test_name):
    raise RuntimeError("Found two tests with the same name: " + test_name)
  setattr(unit_test_class, test_name, test_fn)

def set_cpp_tensors_requires_grad(cpp_tensor_stmts, cpp_tensors):
  assert len(cpp_tensor_stmts) == len(cpp_tensors)
  return ['{}.requires_grad_(true)'.format(tensor_stmt) if tensor.dtype != torch.long else tensor_stmt \
    for tensor_stmt, (_, tensor) in zip(cpp_tensor_stmts, cpp_tensors)]

def move_cpp_tensors_to_device(cpp_tensor_stmts, device):
  return ['{}.to("{}")'.format(tensor_stmt, device) for tensor_stmt in cpp_tensor_stmts]

def is_criterion_test(test_instance):
  return isinstance(test_instance, common_nn.CriterionTest) or \
    isinstance(test_instance, common_nn.NewCriterionTest)

def compute_cpp_args_construction_stmts_and_forward_arg_symbols(test_params):
  device = test_params.device
  cpp_forward_args_symbols = []

  def add_cpp_forward_args(args):
    args_stmts = []
    for arg_name, _ in args:
      args_stmts.append('auto {} = arg_dict.at("{}")'.format(arg_name, arg_name))
      cpp_forward_args_symbols.append(arg_name)
    return args_stmts

  cpp_forward_input_args_stmts = move_cpp_tensors_to_device(set_cpp_tensors_requires_grad(add_cpp_forward_args(test_params.arg_dict['input']), test_params.arg_dict['input']), device)
  cpp_forward_target_args_stmts = move_cpp_tensors_to_device(add_cpp_forward_args(test_params.arg_dict['target']), device)
  cpp_forward_extra_args_stmts = move_cpp_tensors_to_device(add_cpp_forward_args(test_params.arg_dict['extra_args']), device)

  # Build the list of other arguments needed
  cpp_other_args_stmts = []
  for arg_name, _ in test_params.arg_dict['other']:
    cpp_other_args_stmts.append('auto {} = arg_dict.at("{}")'.format(arg_name, arg_name))
  cpp_other_args_stmts = move_cpp_tensors_to_device(cpp_other_args_stmts, device)

  cpp_args_construction_stmts = cpp_forward_input_args_stmts + cpp_forward_target_args_stmts + cpp_forward_extra_args_stmts + cpp_other_args_stmts

  return cpp_args_construction_stmts, cpp_forward_args_symbols

def serialize_arg_dict_as_script_module(arg_dict):
  arg_dict_flat = {
    arg_name: arg_value \
      for arg_name, arg_value in \
        arg_dict['input'] + \
        arg_dict['target'] + \
        arg_dict['extra_args'] + \
        arg_dict['other']
  }
  arg_dict_module = torch.nn.Module()
  for arg_name, arg_value in arg_dict_flat.items():
    assert isinstance(arg_value, torch.Tensor)
    arg_dict_module.register_buffer(arg_name, arg_value)

  return torch.jit.script(arg_dict_module)

# NOTE: any argument symbol used in `cpp_constructor_args` / `cpp_options_args` / `cpp_function_call`
# must have a mapping in `cpp_arg_symbol_map`.
#
# The mapping can take one of the following formats:
#
# 1. `argument_name` -> Python value
# 2. `argument_name` -> 'input' (which means `argument_name` in C++ will be bound to `test_instance._get_input()`)
#
# For example:
# ```
# def bceloss_weights_no_reduce_test():
#     t = torch.randn(15, 10).gt(0).double()
#     weights = torch.rand(10)
#     return dict(
#         fullname='BCELoss_weights_no_reduce',
#         constructor=wrap_functional(
#             lambda i: F.binary_cross_entropy(i, t.type_as(i),
#                                              weight=weights.type_as(i), reduction='none')),
#         cpp_function_call='F::binary_cross_entropy(i, t.to(i.options()), F::BinaryCrossEntropyFuncOptions().weight(weights.to(i.options())).reduction(torch::kNone))',
#         input_fn=lambda: torch.rand(15, 10).clamp_(2.8e-2, 1 - 2.8e-2),
#         cpp_arg_symbol_map={'i': 'input', 't': t, 'weights': weights},
#         reference_fn=lambda i, p, m: -(t * i.log() + (1 - t) * (1 - i).log()) * weights,
#     )
# ```
def compute_arg_dict(test_params_dict, test_instance):
  arg_dict = {
    'input': [],
    'target': [],
    'extra_args': [],
    'other': [],
  }

  def put_args_into_arg_dict(arg_type, arg_type_prefix, args):
    for i, arg in enumerate(args):
      arg_dict[arg_type].append(CppArg(name=arg_type_prefix+str(i), value=arg))

  put_args_into_arg_dict('input', 'i', convert_to_list(test_instance._get_input()))
  if is_criterion_test(test_instance):
    put_args_into_arg_dict('target', 't', convert_to_list(test_instance._get_target()))
  if test_instance.extra_args:
    put_args_into_arg_dict('extra_args', 'e', convert_to_list(test_instance.extra_args))

  cpp_arg_symbol_map = test_params_dict.get('cpp_arg_symbol_map', {})
  for arg_name, arg_value in cpp_arg_symbol_map.items():
    if isinstance(arg_value, str):
      if arg_value == 'input':
        arg_dict['other'].append(CppArg(name=arg_name, value=test_instance._get_input()))
      else:
        raise RuntimeError("`{}` has unsupported string value: {}".format(arg_name, arg_value))
    elif isinstance(arg_value, torch.Tensor):
      arg_dict['other'].append(CppArg(name=arg_name, value=arg_value))
    else:
      raise RuntimeError("`{}` has unsupported value: {}".format(arg_name, arg_value))

  return arg_dict

def skip_test_fn_if_needed(test_fn, test_params_dict, test_cuda, has_impl_parity, device):
  test_fn = unittest.skipIf(not test_params_dict.get('test_cpp_api_parity', True), "Excluded from C++ API parity tests")(test_fn)

  if device == 'cuda':
    test_fn = unittest.skipIf(not test_cuda, "CUDA unavailable")(test_fn)
    test_fn = unittest.skipIf(not test_params_dict.get('test_cuda', True), "Excluded from CUDA tests")(test_fn)

  # If `Implementation Parity` entry in parity table for this module is `No`,
  # we mark the test as expected failure.
  if not has_impl_parity:
    test_fn = unittest.expectedFailure(test_fn)

  return test_fn
