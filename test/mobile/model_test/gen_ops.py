"""
This is a script to aggregate production ops from
fbsource/xplat/pytorch_models/build/all_mobile_model_configs.yaml.
Specify the file path in the first argument. The results will be dump to model_ops.yaml
"""

import sys
import yaml

root_operators = set()
traced_operators = set()
kernel_metadata = {}

with open(sys.argv[1]) as input_yaml_file:
    model_infos = yaml.safe_load(input_yaml_file)
    for info in model_infos:
        root_operators.update(info["root_operators"])
        traced_operators.update(info["traced_operators"])
        # merge dtypes for each kernel
        for kernal, dtypes in info["kernel_metadata"].items():
            new_dtypes = dtypes + (kernel_metadata[kernal] if kernal in kernel_metadata else [])
            kernel_metadata[kernal] = list(set(new_dtypes))


# Only test these built-in ops. No custom ops or non-CPU ops.
namespaces = ["aten", "prepacked", "prim", "quantized"]
root_operators = sorted(list([x for x in root_operators if x.split("::")[0] in namespaces]))
traced_operators = sorted(list([x for x in traced_operators if x.split("::")[0] in namespaces]))

out_path = "test/mobile/model_test/model_ops.yaml"
with open(out_path, "w") as f:
    yaml.safe_dump({"root_operators": root_operators, "traced_operators": traced_operators, "kernel_metadata": kernel_metadata}, f)
    f.close()
