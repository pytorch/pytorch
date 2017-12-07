import os
import sys
import glob

source_files = set(['.py', '.cpp', '.h'])


def all_generator_source():
    r = []
    for directory, _, filenames in os.walk('tools'):
        for f in filenames:
            if os.path.splitext(f)[1] in source_files:
                full = os.path.join(directory, f)
                r.append(full)
    return sorted(r)


inputs = [
    'torch/csrc/generic/TensorMethods.cwrap',
    'torch/lib/tmp_install/share/ATen/Declarations.yaml',
    'tools/autograd/derivatives.yaml',
] + glob.glob('torch/csrc/generic/methods/*.cwrap')

outputs = [
    'torch/csrc/autograd/generated/Functions.cpp',
    'torch/csrc/autograd/generated/Functions.h',
    'torch/csrc/autograd/generated/python_functions.cpp',
    'torch/csrc/autograd/generated/python_functions.h',
    'torch/csrc/autograd/generated/python_nn_functions.cpp',
    'torch/csrc/autograd/generated/python_nn_functions.h',
    'torch/csrc/autograd/generated/python_nn_functions_dispatch.h',
    'torch/csrc/autograd/generated/python_variable_methods.cpp',
    'torch/csrc/autograd/generated/python_variable_methods_dispatch.h',
    'torch/csrc/autograd/generated/VariableType.cpp',
    'torch/csrc/autograd/generated/VariableType.h',
    'torch/csrc/jit/generated/aten_dispatch.cpp',
    'torch/csrc/jit/generated/aten_dispatch.h',
]


def generate_code_ninja(w):
    all_inputs = all_generator_source() + inputs
    cmd = "{} {}".format(sys.executable, 'tools/setup_helpers/generate_code.py')
    w.writer.build(
        outputs, 'do_cmd', all_inputs,
        variables={
            'cmd': cmd,
        })


def generate_code(ninja_global=None):
    # if ninja is enabled, we just register this file as something
    # ninja will need to call if needed
    if ninja_global is not None:
        return generate_code_ninja(ninja_global)

    # cwrap depends on pyyaml, so we can't import it earlier
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(root)
    from tools.cwrap import cwrap
    from tools.cwrap.plugins.THPPlugin import THPPlugin
    from tools.cwrap.plugins.ArgcountSortPlugin import ArgcountSortPlugin
    from tools.cwrap.plugins.AutoGPU import AutoGPU
    from tools.cwrap.plugins.BoolOption import BoolOption
    from tools.cwrap.plugins.KwargsPlugin import KwargsPlugin
    from tools.cwrap.plugins.NullableArguments import NullableArguments

    from tools.cwrap.plugins.WrapDim import WrapDim
    from tools.cwrap.plugins.AssertNDim import AssertNDim

    from tools.cwrap.plugins.Broadcast import Broadcast
    from tools.cwrap.plugins.ProcessorSpecificPlugin import ProcessorSpecificPlugin
    from tools.autograd.gen_variable_type import gen_variable_type
    from tools.jit.gen_jit_dispatch import gen_jit_dispatch
    thp_plugin = THPPlugin()

    cwrap('torch/csrc/generic/TensorMethods.cwrap', plugins=[
        ProcessorSpecificPlugin(), BoolOption(), thp_plugin,
        AutoGPU(condition='IS_CUDA'), ArgcountSortPlugin(), KwargsPlugin(),
        AssertNDim(), WrapDim(), Broadcast()
    ])
    # Build ATen based Variable classes
    autograd_gen_dir = 'torch/csrc/autograd/generated'
    jit_gen_dir = 'torch/csrc/jit/generated'
    for d in (autograd_gen_dir, jit_gen_dir):
        if not os.path.exists(d):
            os.mkdir(d)
    gen_variable_type(
        'torch/lib/tmp_install/share/ATen/Declarations.yaml',
        autograd_gen_dir)
    gen_jit_dispatch(
        'torch/lib/tmp_install/share/ATen/Declarations.yaml',
        jit_gen_dir)

# called from ninja
if __name__ == "__main__":
    generate_code(None)
