import torch
import yaml
from math_ops import MathOpsModule
from nn_ops import NNOpsModule
from sampling_ops import SamplingOpsModule
from tensor_ops import TensorOpsModule

path = "/Users/linbin/model_operators.yaml"

production_ops = []
with open(path) as input_yaml_file:
    ops_info = yaml.safe_load(input_yaml_file)
    production_ops = ops_info["root_operators"]
    print(production_ops)


def scriptAndSave(module, name):
    script_module = torch.jit.script(module)
    # optimized_module = optimize_for_mobile(script_module)
    ops = torch.jit.export_opnames(script_module)
    # optimized_ops = torch.jit.export_opnames(optimized_module)
    print(ops)
    script_module._save_for_lite_interpreter(name)
    script_module()
    print("model saved.")
    return ops


ops = [
    scriptAndSave(MathOpsModule(), "math_ops.ptl"),
    scriptAndSave(TensorOpsModule(), "tensor_ops.ptl"),
    scriptAndSave(NNOpsModule(), "nn_ops.ptl"),
    scriptAndSave(SamplingOpsModule(), "sampling_ops.ptl"),
]
# print(ops)

diff_ops = set(production_ops) - set().union(*ops)
print(f"{len(diff_ops)}/{len(production_ops)} production ops not covered")
print(diff_ops)
