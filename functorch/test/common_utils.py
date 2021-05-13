import itertools
import types
import functools
from collections import namedtuple

"""
Usage:

class MyTestCase(TestCase):
    @parameterized('param', {'abs': torch.abs, 'cos': torch.cos})
    def test_single_param(self, param):
        pass

    @parameterized('param1', {'sin': torch.sin, 'tan': torch.tan})
    @parameterized('param2', {'abs': torch.abs, 'cos': torch.cos})
    def test_multiple_param(self, param1, param2):
        pass

# The following creates:
# - MyTestCase.test_single_param_abs
# - MyTestCase.test_single_param_cos
# - MyTestCase.test_multiple_param_abs_sin
# - MyTestCase.test_multiple_param_cos_sin
# - MyTestCase.test_multiple_param_abs_tan
# - MyTestCase.test_multiple_param_cos_tan
instantiate_parameterized_methods(MyTestCase)

# This is also composable with PyTorch testing's instantiate_device_type_tests
# Make sure the param is after the device arg
class MyDeviceSpecificTest(TestCase):
    @parameterized('param', {'abs': torch.abs, 'cos': torch.cos})
    def test_single_param(self, device, param):
        pass

# The following creates:
# - MyDeviceSpecificTestCPU.test_single_param_abs_cpu
# - MyDeviceSpecificTestCPU.test_single_param_cos_cpu
# - MyDeviceSpecificTestCUDA.test_single_param_abs_cuda
# - MyDeviceSpecificTestCUDA.test_single_param_cos_cpu
instantiate_parameterized_methods(MyDeviceSpecificTest)
instantiate_device_type_tests(MyDeviceSpecificTest, globals())

# !!!!! warning !!!!!
# 1. The method being parameterized over MUST NOT HAVE A DOCSTRING. We'll
# error out nicely if this happens.
# 2. All other decorators MUST USE functools.wraps (they must propagate the docstring)
# `@parameterized` works by storing some metadata in place of the docstring.
# This takes advantage of how other decorators work (other decorators usually
# propagate the docstring via functools.wrap).
# 3. We might not compose with PyTorch testing's @dtypes and @precision
# decorators. But that is easily fixable. TODO.
# I think this composes with PyTorch testing's instantiate_device_type_tests.
"""

PARAM_META = '_torch_parameterized_meta'

class ParamMeta():
    def __init__(self):
        self.stack = []

    def push(self, elt):
        self.stack.append(elt)

    def pop(self, elt):
        return self.stack.pop()

def has_param_meta(method):
    param_meta = getattr(method, '__doc__', None)
    return param_meta is not None and isinstance(param_meta, ParamMeta)

def get_param_meta(method):
    param_meta = getattr(method, '__doc__', None)
    if param_meta is None:
        method.__doc__ = ParamMeta()
    if not isinstance(method.__doc__, ParamMeta):
        raise RuntimeError('Tried to use @parameterized on a method that has '
                           'a docstring. This is not supported. Please remove '
                           'the docstring.')
    return method.__doc__

def parameterized(arg_name, case_dict):
    def decorator(fn):
        param_meta = get_param_meta(fn)
        param_meta.push((arg_name, case_dict))
        return fn
    return decorator

def _set_parameterized_method(test_base, fn, instantiated_cases, extension_name):
    new_name = f'{fn.__name__}_{extension_name}'

    def wrapped(self, *args, **kwargs):
        for arg_name, case in instantiated_cases:
            kwargs[arg_name] = case
        return fn(self, *args, **kwargs)

    wrapped.__name__ = new_name
    setattr(test_base, new_name, wrapped)

def to_tuples(dct):
    return [(k, v) for k, v in dct.items()]

def instantiate_parameterized_methods(test_base):
    allattrs = tuple(dir(test_base))
    for attr_name in allattrs:
        attr = getattr(test_base, attr_name)
        if not has_param_meta(attr):
            continue

        param_meta = get_param_meta(attr)
        arg_names, case_dicts = zip(*param_meta.stack)
        case_dicts = [to_tuples(cd) for cd in case_dicts]
        for list_of_name_and_case in itertools.product(*case_dicts):
            case_names, cases = zip(*list_of_name_and_case)
            extension_name = '_'.join(case_names)
            instantiated_cases = list(zip(arg_names, cases))
            _set_parameterized_method(test_base, attr, instantiated_cases, extension_name)
        # Remove the base fn from the testcase
        delattr(test_base, attr_name)
