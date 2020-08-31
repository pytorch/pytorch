import torch
from torch.testing import FileCheck

from torch.testing._internal.common_utils import \
    (run_tests)
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, skipCPUIfNoLapack, skipCUDAIfNoMagma, onlyCPU)

# Information for generating an alias test
# NOTE: ending the alias_name with an underscore will interpret the test
#   as the test for an inplace method of that name
class AliasInfo(object):
    __slots__ = ['alias_name', 'alias_op', 'original_name', 'original_op',
                 'get_input', 'get_args', 'decorators']

    def __init__(self,
                 alias_name,  # the name of the alias
                 alias_op,  # the aliased op
                 original_name,  # the name of the original function
                 original_op,  # the original op
                 get_input,  # callable (device)->tensor that returns the first tensor argument
                 *,
                 get_args=lambda d: (),  # callable (device)->tuple that returns additional positional arguments
                 decorators=()):  # decorators to apply to the test
        self.alias_name = alias_name
        self.alias_op = alias_op
        self.original_name = original_name
        self.original_op = original_op
        self.get_input = get_input
        self.get_args = get_args
        self.decorators = decorators

alias_infos = (
    AliasInfo('absolute', torch.absolute, 'abs', torch.abs,
              lambda d: torch.randn(20, device=d)),
    AliasInfo('absolute_', torch.Tensor.absolute_, 'abs_', torch.Tensor.abs_,
              lambda d: torch.randn(20, device=d)),
    AliasInfo('clip', torch.clip, 'clamp', torch.clamp,
              lambda d: torch.randn(20, device=d), get_args=lambda d: (.4, .6)),
    AliasInfo('clip_', torch.Tensor.clip_, 'clamp_', torch.Tensor.clamp_,
              lambda d: torch.randn(20, device=d), get_args=lambda d: (.4, .6)),
    AliasInfo('linalg.det', torch.linalg.det, 'det', torch.det,
              lambda d: torch.randn(10, 10, device=d),
              decorators=(skipCPUIfNoLapack, skipCUDAIfNoMagma)),
    # NOTE: only runs on CPU because it leaks CUDA memory
    #   (see https://github.com/pytorch/pytorch/issues/43119)
    AliasInfo('outer', torch.outer, 'ger', torch.ger,
              lambda d: torch.randn(20, device=d), get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('arccosh', torch.arccosh, 'acosh', torch.acosh,
              lambda d: torch.randn(20, device=d) + 2),
    AliasInfo('arccosh_', torch.Tensor.arccosh_, 'acosh_', torch.Tensor.acosh_,
              lambda d: torch.randn(20, device=d) + 2),
    AliasInfo('arccos', torch.arccos, 'acos', torch.acos,
              lambda d: torch.randn(20, device=d)),
    AliasInfo('arccos_', torch.Tensor.arccos_, 'acos_', torch.Tensor.acos_,
              lambda d: torch.randn(20, device=d)),
    AliasInfo('arcsin', torch.arcsin, 'asin', torch.asin,
              lambda d: torch.randn(20, device=d)),
    AliasInfo('arcsin_', torch.Tensor.arcsin_, 'asin_', torch.Tensor.asin_,
              lambda d: torch.randn(20, device=d)),
    AliasInfo('arctan', torch.arctan, 'atan', torch.atan,
              lambda d: torch.randn(20, device=d)),
    AliasInfo('arctan_', torch.Tensor.arctan_, 'atan_', torch.Tensor.atan_,
              lambda d: torch.randn(20, device=d)),
    AliasInfo('fix', torch.fix, 'trunc', torch.trunc,
              lambda d: 10 * torch.randn(20, device=d)),
    AliasInfo('fix_', torch.Tensor.fix_, 'trunc_', torch.Tensor.trunc_,
              lambda d: 10 * torch.randn(20, device=d)),
    AliasInfo('negative', torch.negative, 'neg', torch.neg,
              lambda d: 10 * torch.randn(20, device=d)),
    AliasInfo('negative_', torch.Tensor.negative_, 'neg_', torch.Tensor.neg_,
              lambda d: 10 * torch.randn(20, device=d)),
    AliasInfo('arcsinh', torch.arcsinh, 'asinh', torch.asinh,
              lambda d: torch.randn(20, device=d)),
    AliasInfo('arcsinh_', torch.Tensor.arcsinh_, 'asinh_', torch.Tensor.asinh_,
              lambda d: torch.randn(20, device=d)),
    AliasInfo('arctanh', torch.arctanh, 'atanh', torch.atanh,
              lambda d: torch.clamp(torch.randn(20, device=d), -1, 1)),
    AliasInfo('arctanh_', torch.Tensor.arctanh_, 'atanh_', torch.Tensor.atanh_,
              lambda d: torch.clamp(torch.randn(20, device=d), -1, 1)),
    AliasInfo('subtract', torch.subtract, 'sub', torch.sub,
              lambda d: torch.randn(20, device=d),
              get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('subtract_', torch.Tensor.subtract_, 'sub_', torch.Tensor.sub_,
              lambda d: torch.randn(20, device=d),
              get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
)

# Placeholder test class for validating that aliases are correctly
#   translated when scripted and traced
class TestOpNormalization(JitTestCase):
    pass

# Generates alias tests and adds them to the specified class (cls)
def create_alias_tests(cls):
    for info in alias_infos:

        # Tests that the JIT remaps aliases to their original ops
        def _test_jit_op_alias_normalization(self, device, info=info):
            tensor = torch.tensor
            op = info.alias_op
            is_inplace = info.alias_name.endswith('_')

            # Checks that scripting converts aliases
            # NOTE: the code to test scripting must be generated since
            #   scripting does not support splatting args or directly
            #   calling torch.Tensor methods. The following
            #   splats args after the first tensor by inlining them as constants.
            if is_inplace:
                fn_template = '''
                    def _fn(t):
                        return t.{alias_name}({args})
                '''
                arg_string = ', '.join((str(arg) for arg in info.get_args(device)))
                script = fn_template.format(alias_name=info.alias_name, args=arg_string)
            else:
                fn_template = '''
                    def _fn(t):
                        return op(t{args})
                '''
                arg_string = ", " + ', '.join((str(arg) for arg in info.get_args(device)))
                script = fn_template.format(args=arg_string)

            # Compiles script
            scripted = torch.jit.CompilationUnit(script)._fn

            # Acquires and checks the graph remaps the alias
            inp = info.get_input(device)
            scripted(inp.clone())
            graph = scripted.graph_for(inp.clone())
            FileCheck().check(info.original_name).check_not(info.alias_name).run(graph)

            # Checks that tracing converts aliases
            # NOTE: tracing has no problem splatting args
            args = info.get_args(device)

            def _fn(t, info=info, args=args):
                return info.alias_op(t, *args)

            traced = torch.jit.trace(_fn, (inp.clone(),))
            traced(inp.clone())
            graph = traced.graph_for(inp.clone())
            FileCheck().check(info.original_name).check_not(info.alias_name).run(graph)

        # Applies decorators
        for decorator in info.decorators:
            _test_jit_op_alias_normalization = decorator(_test_jit_op_alias_normalization)

        test_name = "test_jit_op_alias_normalization_" + info.alias_name
        setattr(cls, test_name, _test_jit_op_alias_normalization)

        # Tests that the alias functions perform the same operation as the original
        def _test_alias_computation(self, device, info=info):
            alias_op = info.alias_op
            original_op = info.original_op

            inp = info.get_input(device)
            args = info.get_args(device)

            alias_input = inp.clone()
            alias_result = alias_op(alias_input, *args)

            original_input = inp.clone()
            original_result = alias_op(original_input, *args)

            self.assertEqual(alias_input, original_input, atol=0, rtol=0)
            self.assertEqual(alias_result, original_result, atol=0, rtol=0)

        # Applies decorators
        for decorator in info.decorators:
            _test_alias_computation = decorator(_test_alias_computation)

        test_name = "test_alias_computation_" + info.alias_name
        setattr(cls, test_name, _test_alias_computation)


create_alias_tests(TestOpNormalization)
instantiate_device_type_tests(TestOpNormalization, globals())

if __name__ == '__main__':
    run_tests()
