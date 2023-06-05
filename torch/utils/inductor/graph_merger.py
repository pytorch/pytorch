#!/usr/bin/env python3
import os
import io
import shutil
from torch._inductor import codecache

def merge_fx_graphs(dbg_log, dbg_code_dst_name="graph_merger"):
    debug_trace_flag = "Debug trace"
    cache_dir = codecache.cache_dir()
    dbg_code_dst_dir = os.path.join(cache_dir, dbg_code_dst_name)
    if not os.path.exists(dbg_code_dst_dir):
        os.makedirs(dbg_code_dst_dir, exist_ok=True)
    dbg_code_dst_single_readable_py = os.path.join(dbg_code_dst_dir, "fx_graph_readable_merged_to_one.py")

    lines = io.StringIO(dbg_log).readlines()
    graph_idx = 0

    with open(dbg_code_dst_single_readable_py, "w") as all_readable_py:
        for i in range(0, len(lines)):
            line = lines[i]
            if debug_trace_flag in line:
                line_items = line.split(": ")
                assert(len(line_items) == 2)
                current_graph_gen_code_path = line_items[1].strip()
                assert(current_graph_gen_code_path.endswith("debug"))
                assert(os.path.exists(current_graph_gen_code_path))
                print("\nGenerated code of graph {} is placed at {}".format(str(graph_idx), current_graph_gen_code_path))

                output_path_with_graph_idx = os.path.join(dbg_code_dst_dir, str(graph_idx) + ".debug")
                print("Copy recursively from {} to {}".format(current_graph_gen_code_path, output_path_with_graph_idx))
                shutil.copytree(current_graph_gen_code_path, output_path_with_graph_idx, dirs_exist_ok=True)
                # Recursively enumerate sub dir and files and rename them to graph_idx
                print("Copy the generated code of graph {} to {}".format(str(graph_idx), output_path_with_graph_idx))

                with open(os.path.join(output_path_with_graph_idx, "fx_graph_readable.py")) as graph_readable:
                    all_readable_py.writelines("GRAPH_INDEX:{}\n".format(str(graph_idx)))
                    content = graph_readable.readlines()
                    all_readable_py.writelines(content)
                    all_readable_py.writelines("\n")

                graph_idx += 1

    print("\nThe merged graph is written to {}".format(dbg_code_dst_single_readable_py))
    return dbg_code_dst_single_readable_py
