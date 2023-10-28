"""
  File "/data/users/willfeng/pytorch_yf225/test/test_yf225.py", line 81, in <module>
    compiled_m = torch.compile(m, backend="aot_eager" if device == "cpu" else "inductor", fullgraph=False, dynamic=False)
  File "/data/users/willfeng/pytorch_yf225/torch/__init__.py", line 1768, in compile
    return torch._dynamo.optimize(backend=backend, nopython=fullgraph, dynamic=dynamic, disable=disable)(model)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/eval_frame.py", line 347, in __call__
    new_mod = OptimizedModule(mod, self)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/eval_frame.py", line 180, in __init__
    self._initialize()
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/eval_frame.py", line 184, in _initialize
    if isinstance(self._orig_mod.forward, types.MethodType) and skipfiles.check(
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/skipfiles.py", line 380, in check
    return check_verbose(obj, allow_torch).skipped
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/skipfiles.py", line 368, in check_verbose
    traceback.print_stack()
obj: <code object forward at 0x7fdc06138b90, file "/data/users/willfeng/pytorch_yf225/test/test_yf225.py", line 63>

  File "/data/users/willfeng/pytorch_yf225/test/test_yf225.py", line 86, in <module>
    actual = compiled_m(x, y)
  File "/data/users/willfeng/pytorch_yf225/torch/nn/modules/module.py", line 1519, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/nn/modules/module.py", line 1528, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/eval_frame.py", line 410, in _fn
    return fn(*args, **kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/nn/modules/module.py", line 1519, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/nn/modules/module.py", line 1528, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/eval_frame.py", line 531, in catch_errors
    or skipfiles.check(frame.f_code)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/skipfiles.py", line 380, in check
    return check_verbose(obj, allow_torch).skipped
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/skipfiles.py", line 368, in check_verbose
    traceback.print_stack()
obj: <code object forward at 0x7fdc06138b90, file "/data/users/willfeng/pytorch_yf225/test/test_yf225.py", line 63>

  File "/data/users/willfeng/pytorch_yf225/test/test_yf225.py", line 86, in <module>
    actual = compiled_m(x, y)
  File "/data/users/willfeng/pytorch_yf225/torch/nn/modules/module.py", line 1519, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/nn/modules/module.py", line 1528, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/eval_frame.py", line 410, in _fn
    return fn(*args, **kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/nn/modules/module.py", line 1519, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/nn/modules/module.py", line 1528, in _call_impl
    return forward_call(*args, **kwargs)
  File "/data/users/willfeng/pytorch_yf225/test/test_yf225.py", line 63, in forward
    def forward(self, x, y):
  File "/data/users/willfeng/pytorch_yf225/test/test_yf225.py", line 49, in subfunc
    def subfunc(self, x, y):
  File "/data/users/willfeng/pytorch_yf225/test/test_yf225.py", line 50, in resume_in_subfunc
    z = torch.relu(x) + g1_mutation_tuple(x, y)[0]
  File "/data/users/willfeng/pytorch_yf225/test/test_yf225.py", line 51, in resume_in_subfunc
    z = z + g1_mutation_tensor(x, x)
  File "/data/users/willfeng/pytorch_yf225/test/test_yf225.py", line 52, in resume_in_subfunc
    z = z + g2(x, y)
  File "/data/users/willfeng/pytorch_yf225/test/test_yf225.py", line 54, in resume_in_subfunc
    z = z + g2_read_global_var(x, y)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/eval_frame.py", line 558, in catch_errors
    return callback(frame, cache_entry, hooks, frame_state)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/convert_frame.py", line 686, in _convert_frame
    result = inner_convert(frame, cache_entry, hooks, frame_state)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/convert_frame.py", line 148, in _fn
    return fn(*args, **kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/convert_frame.py", line 405, in _convert_frame_assert
    return _compile(
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/convert_frame.py", line 613, in _compile
    guarded_code = compile_inner(code, one_graph, hooks, transform)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/utils.py", line 221, in time_wrapper
    r = func(*args, **kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/convert_frame.py", line 530, in compile_inner
    out_code = transform_code_object(code, transform)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/bytecode_transformation.py", line 1028, in transform_code_object
    transformations(instructions, code_options)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/convert_frame.py", line 500, in transform
    tracer.run()
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/symbolic_convert.py", line 2117, in run
    super().run()
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/symbolic_convert.py", line 742, in run
    and self.step()
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/symbolic_convert.py", line 705, in step
    getattr(self, inst.opname)(inst)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/symbolic_convert.py", line 405, in wrapper
    return inner_fn(self, inst)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/symbolic_convert.py", line 1771, in CALL
    self.call_function(fn, args, kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/symbolic_convert.py", line 577, in call_function
    self.push(fn.call_function(self, args, kwargs))
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/variables/functions.py", line 307, in call_function
    return super().call_function(tx, args, kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/variables/functions.py", line 261, in call_function
    return super().call_function(tx, args, kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/variables/functions.py", line 90, in call_function
    return tx.inline_user_function_return(
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/symbolic_convert.py", line 613, in inline_user_function_return
    result = InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/symbolic_convert.py", line 2244, in inline_call
    return cls.inline_call_(parent, func, args, kwargs)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/symbolic_convert.py", line 2308, in inline_call_
    result = InliningInstructionTranslator.check_inlineable(func)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/symbolic_convert.py", line 2270, in check_inlineable
    result = skipfiles.check_verbose(func, allow_torch=True)
  File "/data/users/willfeng/pytorch_yf225/torch/_dynamo/skipfiles.py", line 368, in check_verbose
    traceback.print_stack()
obj: <code object f2 at 0x7fdc061b1fe0, file "/data/users/willfeng/pytorch_yf225/test/test_yf225.py", line 46>
"""

import torch
import torch._dynamo.config
from torch._inductor.utils import run_and_get_triton_code
from torch.testing import FileCheck
from torch._dynamo import disable

# TORCH_LOGS="+dynamo,aot,inductor" TORCH_COMPILE_DEBUG=1 python test/test_yf225.py

device = "cuda" if torch.cuda.is_available() else "cpu"

@disable()
def g1_mutation_tuple(d, e):
    d.relu_()
    return d, e

@disable()
def g1_mutation_tensor(d, e):
    d.relu_()
    return d + e

@disable()
def g2(a, b):
    return torch.cat(torch.chunk(a * b, 2))

global_a = torch.randn(4, 4, device=device)

@disable()
def g2_read_global_var(a, b):
    return torch.cat(torch.chunk(a * b.div(torch.selu(global_a)), 2))

def global3(a, b):
    return a + b


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1))  # torch.randn(4, 4))
        self.register_buffer('buf', torch.randn(1))  # torch.randn(4, 4))

    @disable()
    def f_read_param_mutate_param(self, c):
        self.buf.relu_()
        return c * c * self.weight

    def f2(self, x, y):
        return x + y

    def subfunc(self, x, y):
        z = torch.relu(x) + g1_mutation_tuple(x, y)[0]
        z = z + g1_mutation_tensor(x, x)
        z = z + g2(x, y)
        z = x + y
        z = z + g2_read_global_var(x, y)
        z = z + self.f_read_param_mutate_param(x)
        z = z + torch.tanh(self.weight)
        z = z + self.buf
        z = z + global_a
        z = z + self.f2(x, y)
        z = z + global3(x, y)
        return z

    def forward(self, x, y):
        x.relu_()
        self.buf.relu_()
        y = torch.cat(torch.chunk(y, 2))
        z = self.subfunc(x, y)
        return z


with (
    torch._dynamo.config.patch(
        dynamic_shapes=False,
        capture_dynamic_output_shape_ops=False,
        capture_scalar_outputs=False,
    ),
):
    torch._dynamo.reset()
    m = TestModule()
    m = m.to(device)
    compiled_m = torch.compile(m, backend="aot_eager" if device == "cpu" else "inductor", fullgraph=False, dynamic=False)
    x = torch.randn(4, 4, device=device)
    y = torch.randn(4, 4, device=device)

    # ref = m(x, y)
    actual = compiled_m(x, y)
    # assert torch.allclose(ref, actual)
