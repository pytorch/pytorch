#!/usr/bin/env python3
import os
import io

'''
 This script helps add corresponding fx graph operations before each kernel in cpp output code.

 TODO: Support the case where the operator orders of output code and fx graph are inconsistent. Raise an error now.
'''

# Skip lines for non-operators. Feel free to add more if needed.
skip_pattern = {
    "    cpp_fused_",
    " = args",
    "args.clear()",
    "    del ",
    "= buf",
    " = as_strided(",
    " = empty_strided(",
    "    assert_size_stride("}

# For operations which are excluded from kernels, map their names from cpp to fx graph. Feel free to add more if needed.
cpp_2_graph_operation_dict = {
    "aten.convolution": "torch.ops.aten.convolution",
    "extern_kernels.convolution": "torch.ops.aten.convolution.default",
    "torch.ops.mkldnn._convolution_pointwise.binary": "torch.ops.mkldnn._convolution_pointwise.binary",
    "aten.bmm": "aten.bmm",
    "extern_kernels.bmm": "torch.ops.aten.bmm.default",
    "torch.ops.mkl._mkl_linear": "torch.ops.mkl._mkl_linear.default"
}

# Operators of fx graph in the above dict
graph_operation_list = [
    "torch.ops.aten.convolution",
    "torch.ops.aten.convolution.default",
    "torch.ops.mkldnn._convolution_pointwise.binary",
    "aten.bmm",
    "torch.ops.aten.bmm.default",
    "torch.ops.mkl._mkl_linear.default"
]


def match_output_code_with_fx_graph(dbg_log):
    # runnable_graph_file, output_code_file
    debug_trace_flag = "Debug trace"

    lines = io.StringIO(dbg_log).readlines()
    for i in range(0, len(lines)):
        line = lines[i]
        if debug_trace_flag in line:
            line_items = line.split(": ")
            assert(len(line_items) == 2)
            debug_path = line_items[1].strip()
            assert(debug_path.endswith("debug"))
            assert(os.path.exists(debug_path))
            return make_graph_matching(debug_path)


def make_graph_matching(debug_path):
    runnable_graph_file = os.path.join(debug_path, "fx_graph_runnable.py")
    output_code_file = os.path.join(debug_path, "output_code.py")
    assert(os.path.exists(runnable_graph_file))
    assert(os.path.exists(output_code_file))
    matched_file = os.path.join(debug_path, "matched_output_code.py")

    with open(runnable_graph_file, 'r') as graph_f, \
            open(output_code_file, 'r') as cpp_f:
        graph_lines = graph_f.readlines()
        cpp_lines = cpp_f.readlines()

    graph_line_id = 0
    while "def forward" not in graph_lines[graph_line_id]:
        graph_line_id += 1
    graph_line_id += 1

    cpp_line_id = 0
    while "call" not in cpp_lines[cpp_line_id]:
        cpp_line_id += 1
    cpp_line_id += 1

    kernel_2_python_dict = {}
    graph_line_container = []
    kernel_name = ""

    while cpp_line_id < len(cpp_lines) and graph_line_id < len(graph_lines):
        match_substr = ""
        while cpp_line_id < len(cpp_lines) and match_substr == "":
            match_flag = False
            for substr in cpp_2_graph_operation_dict:
                if substr in cpp_lines[cpp_line_id]:
                    match_substr = substr
                    match_flag = True
                    break
            if not match_flag:
                if "cpp_fused_" in cpp_lines[cpp_line_id]:
                    kernel_name = cpp_lines[cpp_line_id][0:cpp_lines[cpp_line_id].find('(')].strip()
                elif all(pt not in cpp_lines[cpp_line_id] for pt in skip_pattern):
                    raise Exception(f"Please add missing pattern in 'skip_pattern' or 'cpp_2_graph_operation_dict' "
                                    f"for cpp #line{cpp_line_id} {cpp_lines[cpp_line_id].strip()}.")
            cpp_line_id += 1
            if "return" in cpp_lines[cpp_line_id]:
                break
        if "return" in cpp_lines[cpp_line_id] or cpp_line_id >= len(cpp_lines):
            break
        while graph_line_id < len(graph_lines) and cpp_2_graph_operation_dict[match_substr] not in graph_lines[graph_line_id]:
            if any(op in graph_lines[graph_line_id] for op in graph_operation_list):
                raise Exception(f"Dismatch for cpp kernel {kernel_name} and "
                                f"graph #line{graph_line_id + 1} {graph_lines[graph_line_id].strip()}.")
            graph_line_container.append('#line ' + str(graph_line_id + 1) + ': ' + graph_lines[graph_line_id].strip() + '\n')
            graph_line_id += 1
        if kernel_name != "":
            kernel_2_python_dict[kernel_name] = graph_line_container

        graph_line_container = []
        kernel_name = ""
        cpp_line_id += 1
        graph_line_id += 1
    while graph_line_id < len(graph_lines) and "return" not in graph_lines[graph_line_id]:
        graph_line_container.append('#line ' + str(graph_line_id + 1) + ': ' + graph_lines[graph_line_id].strip() + '\n')
        graph_line_id += 1

    if kernel_name != "" and len(graph_line_container) > 0:
        kernel_2_python_dict[kernel_name] = graph_line_container

    with open(matched_file, 'w') as matched_f:
        for cpp_line_id in range(len(cpp_lines)):
            if "async_compile.cpp" in cpp_lines[cpp_line_id]:
                kernel_name = cpp_lines[cpp_line_id].split(" =", 1)[0]
                for v in kernel_2_python_dict[kernel_name]:
                    matched_f.write(v)
            matched_f.write(cpp_lines[cpp_line_id])

    print("\nThe matched output_code is written to {}".format(matched_file))
    return matched_file
