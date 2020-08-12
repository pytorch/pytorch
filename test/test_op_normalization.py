import torch
from torch.testing import FileCheck

from torch.testing._internal.common_utils import \
    (run_tests)
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, skipCPUIfNoLapack, skipCUDAIfNoMagma)

# Information for generating an alias test
# NOTE: ending the alias_name with an underscore will interpret the test
#   as the test for an inplace method of that name
class AliasInfo(object):
    __slots__ = ['alias_name', 'alias_op', 'original_name', 'input', 'args', 'decorators']

    def __init__(self,
                 alias_name,  # the name of the alias
                 alias_op,  # the aliased op
                 original_name,  # the name of the original function
                 input,  # the first tensor argument to the op
                 *,
                 args=(),  # tuple of additional positional arguments
                 decorators=()):  # decorators to apply to the test
        self.alias_name = alias_name
        self.alias_op = alias_op
        self.original_name = original_name
        self.input = input
        self.args = args
        self.decorators = decorators

alias_infos = (
    AliasInfo('absolute', torch.absolute, 'abs',
              torch.randn(20) - .5),
    AliasInfo('absolute_', torch.Tensor.absolute_, 'abs_',
              torch.randn(20) - .5),
    AliasInfo('clip', torch.clip, 'clamp',
              torch.randn(20), args=(.4, .6)),
    AliasInfo('clip_', torch.Tensor.clip_, 'clamp_',
              torch.randn(20), args=(.4, .6)),
    AliasInfo('linalg.det', torch.linalg.det, 'det',
              torch.randn(10, 10), decorators=(skipCPUIfNoLapack, skipCUDAIfNoMagma)),
    AliasInfo('outer', torch.outer, 'ger',
              torch.randn(20), args=(torch.randn(20),))
)

# Placeholder test class for validating that aliases are correctly
#   translated when scripted and traced
class TestOpNormalization(JitTestCase):
    pass

# Generates alias tests and adds them to the specified class (cls)
def create_alias_tests(cls):
    for info in alias_infos:

        @torch.no_grad()
        def _test(self, device, info=info):
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
                arg_string = ', '.join((str(arg) for arg in info.args))
                script = fn_template.format(alias_name=info.alias_name, args=arg_string)
            else:
                fn_template = '''
                    def _fn(t):
                        return op(t{args})
                '''
                arg_string = ", " + ', '.join((str(arg) for arg in info.args))
                script = fn_template.format(args=arg_string)

            # Compiles script
            scripted = torch.jit.CompilationUnit(script)._fn

            # Acquires and checks the graph remaps the alias
            scripted(info.input)
            graph = scripted.graph_for(info.input)
            FileCheck().check(info.original_name).check_not(info.alias_name).run(graph)

            # Checks that tracing converts aliases
            # NOTE: tracing has no problem splatting args
            def _fn(t):
                return info.alias_op(t, *info.args)

            traced = torch.jit.trace(_fn, (info.input,))
            traced(info.input)
            graph = traced.graph_for(info.input)
            FileCheck().check(info.original_name).check_not(info.alias_name).run(graph)


        # Applies decorators
        for decorator in info.decorators:
            _test = decorator(_test)

        test_name = "test_alias_" + info.alias_name
        setattr(cls, test_name, _test)

create_alias_tests(TestOpNormalization)
instantiate_device_type_tests(TestOpNormalization, globals())

if __name__ == '__main__':
    run_tests()
