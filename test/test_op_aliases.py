import torch
from torch.testing import FileCheck

from torch.testing._internal.common_utils import \
    (run_tests)
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, skipCPUIfNoLapack, skipCUDAIfNoMagma, onlyCPU)
from collections.abc import Sequence

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
    AliasInfo('linalg_det', torch.linalg.det, 'det', torch.det,
              lambda d: torch.randn(10, 10, device=d),
              decorators=(skipCPUIfNoLapack, skipCUDAIfNoMagma)),
    # NOTE: only runs on CPU because it leaks CUDA memory
    #   (see https://github.com/pytorch/pytorch/issues/43119)
    AliasInfo('ger', torch.ger, 'outer', torch.outer,
              lambda d: torch.randn(20, device=d), get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('subtract', torch.subtract, 'sub', torch.sub,
              lambda d: torch.randn(20, device=d),
              get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('subtract_', torch.Tensor.subtract_, 'sub_', torch.Tensor.sub_,
              lambda d: torch.randn(20, device=d),
              get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('greater_equal', torch.greater_equal, 'ge', torch.ge,
              lambda d: torch.randn(20, device=d),
              get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('greater_equal_', torch.Tensor.greater_equal_, 'ge_', torch.Tensor.ge_,
              lambda d: torch.randn(20, device=d),
              get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('greater', torch.greater, 'gt', torch.gt,
              lambda d: torch.randn(20, device=d),
              get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('greater_', torch.Tensor.greater_, 'gt_', torch.Tensor.gt_,
              lambda d: torch.randn(20, device=d),
              get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('less_equal', torch.less_equal, 'le', torch.le,
              lambda d: torch.randn(20, device=d),
              get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('less_equal_', torch.Tensor.less_equal_, 'le_', torch.Tensor.less_equal_,
              lambda d: torch.randn(20, device=d),
              get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('less', torch.less, 'lt', torch.lt,
              lambda d: torch.randn(20, device=d),
              get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('less_', torch.Tensor.less_, 'lt_', torch.Tensor.lt_,
              lambda d: torch.randn(20, device=d),
              get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('not_equal', torch.not_equal, 'ne', torch.ne,
              lambda d: torch.randn(20, device=d),
              get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('not_equal_', torch.Tensor.not_equal_, 'ne_', torch.Tensor.ne_,
              lambda d: torch.randn(20, device=d),
              get_args=lambda d: (torch.randn(20, device=d),),
              decorators=(onlyCPU,)),
    # NOTE: only runs on CPU because it leaks CUDA memory
    #   (see https://github.com/pytorch/pytorch/issues/43119)
    AliasInfo('divide', torch.divide, 'div', torch.div,
              lambda d: torch.randn(20, device=d), get_args=lambda d: (torch.rand(20, device=d) + .1,),
              decorators=(onlyCPU,)),
    AliasInfo('divide_', torch.Tensor.divide_, 'div_', torch.Tensor.div_,
              lambda d: torch.randn(20, device=d), get_args=lambda d: (torch.rand(20, device=d) + .1,),
              decorators=(onlyCPU,)),
    # NOTE: only runs on CPU because it leaks CUDA memory
    #   (see https://github.com/pytorch/pytorch/issues/43119)
    AliasInfo('multiply', torch.multiply, 'mul', torch.mul,
              lambda d: torch.randn(20, device=d), get_args=lambda d: (torch.rand(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('multiply_', torch.Tensor.multiply_, 'mul_', torch.Tensor.mul_,
              lambda d: torch.randn(20, device=d), get_args=lambda d: (torch.rand(20, device=d),),
              decorators=(onlyCPU,)),
    AliasInfo('true_divide', torch.true_divide, 'div', torch.div,
              lambda d: torch.randn(20, device=d), get_args=lambda d: (torch.rand(20, device=d) + .1,),
              decorators=(onlyCPU,)),
    AliasInfo('true_divide_', torch.Tensor.true_divide_, 'div_', torch.Tensor.div_,
              lambda d: torch.randn(20, device=d), get_args=lambda d: (torch.rand(20, device=d) + .1,),
              decorators=(onlyCPU,)),
    AliasInfo('swapdims', torch.swapdims, 'transpose', torch.transpose,
              lambda d: torch.randn(20, 3, 2, 1, device=d), get_args=lambda d: (3, 1)),
    AliasInfo('swapdims_', torch.Tensor.swapdims_, 'transpose_', torch.Tensor.transpose_,
              lambda d: torch.randn(20, 3, 2, 1, device=d), get_args=lambda d: (3, 1)),
    AliasInfo('swapaxes', torch.swapaxes, 'transpose', torch.transpose,
              lambda d: torch.randn(20, 3, 2, 1, device=d), get_args=lambda d: (3, 1)),
    AliasInfo('swapaxes_', torch.Tensor.swapaxes_, 'transpose_', torch.Tensor.transpose_,
              lambda d: torch.randn(20, 3, 2, 1, device=d), get_args=lambda d: (3, 1)),
    AliasInfo('row_stack', torch.row_stack, 'vstack', torch.vstack,
              lambda d: ((torch.randn(20, device=d), torch.randn(20, device=d)))),
    AliasInfo('moveaxis', torch.moveaxis, 'movedim', torch.movedim,
              lambda d: torch.randn(20, 3, 2, 1, device=d), get_args=lambda d: (3, 1)),
)

# Placeholder test class for validating that aliases are correctly
#   translated when scripted and traced
class TestOpNormalization(JitTestCase):
    pass


# Clone input tensor and sequence of Tensors
def clone_inp(inp):
    if isinstance(inp, Sequence):
        return list(map(torch.clone, inp))
    else:
        return inp.clone()

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
                is_input_tensor_list = isinstance(info.get_input(device), Sequence)
                # For sequence of Tensors, annotate the type to be List[Tensor]
                if is_input_tensor_list:
                    fn_template = '''
                    def _fn(t: List[Tensor]):
                        return op(t{args})
                    '''
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
            scripted(clone_inp(inp))
            graph = scripted.graph_for(clone_inp(inp))
            FileCheck().check(info.original_name).check_not(info.alias_name).run(graph)

            # Checks that tracing converts aliases
            # NOTE: tracing has no problem splatting args
            args = info.get_args(device)

            def _fn(t, info=info, args=args):
                return info.alias_op(t, *args)

            traced = torch.jit.trace(_fn, (clone_inp(inp),))
            traced(clone_inp(inp))
            graph = traced.graph_for(clone_inp(inp))
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

            alias_input = clone_inp(inp)
            alias_result = alias_op(alias_input, *args)

            original_input = clone_inp(inp)
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
